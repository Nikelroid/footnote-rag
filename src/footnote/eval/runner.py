"""Run a system over the questions, then grade it, keeping everything needed to audit the numbers.

Answering and grading are separate passes. An answer is paid for once and can be
graded again when the judge changes, and grades from different judge versions are
never mixed. A request that fails for a reason outside the model goes to
errors.jsonl and is never scored as a wrong answer.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .. import __version__, config
from ..data.financebench import Question
from ..ledger import BudgetExceeded
from ..llm import LLM, ServedModelMismatch, SpendLimitReached
from ..pricing import Usage
from ..systems.base import CaseTimeout, RunContext, System
from . import numeric
from .judge import Judge, JudgeError, Verdict
from .metrics import retrieval_scores

# How the last reply of a question ended decides what happens to the answer.
# "truncated" and "overflow" answers are counted and shown, and are not graded.
STOP_STATUS = {
    "end_turn": "ok",
    "max_tokens": "truncated",
    "refusal": "blocked",
    "model_context_window_exceeded": "overflow",
}
UNGRADED = ("truncated", "overflow")

# This many failures in a row means the setup is broken, not the questions.
MAX_FAILURES_IN_A_ROW = 5


class RunLocked(RuntimeError):
    pass


class RunAborted(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class _Appender:
    """Append one JSON row at a time, so a crash never costs the rows already written."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()

    def write(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock, self._path.open("a") as f:
            f.write(line + "\n")
            f.flush()


@contextmanager
def _locked(run_dir: Path):
    """Only one process may write to a run. Two at once would answer, and pay for, every question twice."""
    with (run_dir / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            raise RunLocked(f"{run_dir.name} is being written by another process") from err
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class _Watch:
    """Shared by the workers of one pass: a stop flag, and a count of failures in a row."""

    def __init__(self):
        self.stop = threading.Event()
        self.reason: BaseException | None = None
        self._lock = threading.Lock()
        self._failures = 0

    def succeeded(self) -> None:
        with self._lock:
            self._failures = 0

    def failed(self, kind: str, err: BaseException) -> None:
        with self._lock:
            self._failures += 1
            fatal = kind in ("budget", "spend_limit")
            if fatal or self._failures >= MAX_FAILURES_IN_A_ROW:
                if self.reason is None:
                    self.reason = err if fatal else RunAborted(f"{self._failures} failures in a row, the last one: {err}")
                self.stop.set()


def _run_all(tasks: list, worker: Callable, concurrency: int, watch: _Watch) -> None:
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(worker, *task) for task in tasks]
        try:
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            watch.stop.set()  # requests already sent finish. Nothing new is started.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    if watch.reason is not None:
        raise watch.reason


def run_dir_for(system: System, questions: list[Question], runs_dir: Path | None = None) -> Path:
    """A run is named after what defines it, so the same command resumes the same run."""
    spec = {
        "system": system.describe(),
        "question_ids": sorted(q.id for q in questions),
        "footnote_version": __version__,
    }
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:8]
    return (runs_dir or config.RUNS_DIR) / f"{system.name}-{digest}"


def _classify(err: BaseException) -> str:
    if isinstance(err, BudgetExceeded):
        return "budget"
    if isinstance(err, SpendLimitReached):
        return "spend_limit"
    if isinstance(err, CaseTimeout):
        return "timeout"
    if isinstance(err, ServedModelMismatch):
        return "served_model_mismatch"
    if type(err).__module__.startswith(("anthropic", "google.genai")):
        return "api_error"
    return "harness_error"


def _write_config(run_dir: Path, system: System, questions: list[Question], reps: int, line: str) -> None:
    path = run_dir / "config.json"
    if path.exists():
        cfg = json.loads(path.read_text())
    else:
        cfg = {
            "system": system.describe(),
            "question_ids": sorted(q.id for q in questions),
            "footnote_version": __version__,
            "created": _now(),
            "budget_line": line,
            "reps": 0,
        }
    # The report checks every question and run named here, so a half-finished run cannot pass as a full one.
    cfg["reps"] = max(cfg.get("reps", 0), reps)
    path.write_text(json.dumps(cfg, indent=1))


def answer_run(
    system: System,
    questions: list[Question],
    llm: LLM,
    *,
    reps: int = config.DEFAULT_REPS,
    line: str = "development",
    runs_dir: Path | None = None,
    concurrency: int = 4,
    case_timeout_s: float = 300.0,
    log: Callable[[str], None] = print,
) -> Path:
    run_dir = run_dir_for(system, questions, runs_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with _locked(run_dir):
        _write_config(run_dir, system, questions, reps, line)
        answers = _Appender(run_dir / "answers.jsonl")
        errors = _Appender(run_dir / "errors.jsonl")
        traces = run_dir / "traces"
        traces.mkdir(exist_ok=True)
        done = {(row["case_id"], row["rep"]) for row in _read_jsonl(run_dir / "answers.jsonl")}
        todo = [(q, rep) for q in questions for rep in range(reps) if (q.id, rep) not in done]
        watch = _Watch()

        def one(q: Question, rep: int) -> None:
            if watch.stop.is_set():
                return
            ctx = RunContext(llm=llm, line=line, run_id=run_dir.name, deadline=time.monotonic() + case_timeout_s)
            try:
                out = system.answer(q, ctx)
                if not ctx.calls:
                    raise RuntimeError("the system returned an answer without calling the model")
                last = ctx.calls[-1]
                status = STOP_STATUS.get(last.stop_reason or "")
                if status is None:
                    raise RuntimeError(f"unexpected stop reason {last.stop_reason!r}")
            except Exception as err:  # every failure is classified and recorded
                kind = _classify(err)
                errors.write(
                    {
                        "case_id": q.id,
                        "rep": rep,
                        "stage": "answer",
                        "class": kind,
                        "error": f"{type(err).__name__}: {err}"[:500],
                        "calls": [c.to_record() for c in ctx.calls],
                        "ts": _now(),
                    }
                )
                watch.failed(kind, err)
                return
            watch.succeeded()
            usage = Usage()
            for call in ctx.calls:
                usage = usage + call.usage
            # A chunk may span pages. Every page it covers counts as retrieved at the chunk's rank.
            ranked = [
                (hit["doc"], page) for hit in out.retrieved for page in range(hit["page"], hit.get("end_page", hit["page"]) + 1)
            ]
            (traces / f"{q.id}_rep{rep}.json").write_text(json.dumps(ctx.trace, ensure_ascii=False, indent=1))
            answers.write(
                {
                    "case_id": q.id,
                    "rep": rep,
                    "split": q.split,
                    "tags": q.tags,
                    "question": q.question,
                    "gold": q.answer,
                    "answer": out.answer,
                    "status": status,
                    "stop_reason": last.stop_reason,
                    "model": last.model,
                    "usage": usage.to_dict(),
                    "thinking_tokens": sum(c.thinking_tokens for c in ctx.calls),
                    "cost_usd": sum(c.cost_usd for c in ctx.calls),
                    "latency_s": round(sum(c.latency_s for c in ctx.calls), 3),
                    "attempts": sum(c.attempts for c in ctx.calls),
                    "n_calls": len(ctx.calls),
                    "calls": [c.to_record() for c in ctx.calls],
                    "retrieved": out.retrieved,
                    "citations": out.citations,
                    # A system that searches and finds nothing scores zero. It is not left out of the average.
                    "retrieval": retrieval_scores(q.gold_pages, ranked) if system.retrieves else None,
                    "ts": _now(),
                }
            )

        log(f"{run_dir.name}: {len(todo)} to run, {len(done)} already done")
        _run_all(todo, one, concurrency, watch)
    return run_dir


def grades_path(run_dir: Path, grader_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in grader_id)
    return run_dir / "grades" / f"{safe}.jsonl"


def grade_run(
    run_dir: Path,
    questions: dict[str, Question],
    judge: Judge,
    *,
    concurrency: int = 4,
    only_rep: int | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Grade the answers of a run. With `only_rep`, grade that one repeat of every question, as a sample."""
    path = grades_path(run_dir, judge.grader_id)
    with _locked(run_dir):
        grades = _Appender(path)
        path.touch()  # a run with nothing to grade still has a grades file, so the report can say so
        errors = _Appender(run_dir / "errors.jsonl")
        done = {(row["case_id"], row["rep"]) for row in _read_jsonl(path)}
        rows = [r for r in _read_jsonl(run_dir / "answers.jsonl") if (r["case_id"], r["rep"]) not in done]
        todo = [(r,) for r in rows if r["status"] not in UNGRADED and only_rep in (None, r["rep"])]
        watch = _Watch()

        def one(row: dict) -> None:
            if watch.stop.is_set():
                return
            q = questions[row["case_id"]]
            try:
                if row["status"] == "blocked":
                    verdict = Verdict("refused", "The API stopped the reply with a refusal.", "rule")
                else:
                    verdict = judge.grade(q.question, q.answer, q.justification, row["answer"], run_id=run_dir.name)
            except Exception as err:
                kind = "judge_error" if isinstance(err, JudgeError) else _classify(err)
                errors.write(
                    {"case_id": row["case_id"], "rep": row["rep"], "stage": "grade", "class": kind, "error": f"{type(err).__name__}: {err}"[:500], "ts": _now()}
                )
                watch.failed(kind, err)
                return
            watch.succeeded()
            record = {"case_id": row["case_id"], "rep": row["rep"], "correct": int(verdict.label == "correct")}
            record.update(verdict.to_record())
            record["numeric"] = numeric.grade(q.answer, row["answer"], q.question).to_record()
            record["ts"] = _now()
            grades.write(record)

        log(f"{run_dir.name}: grading {len(todo)} answers with {judge.grader_id}, {len(done)} already graded")
        _run_all(todo, one, concurrency, watch)
    return path
