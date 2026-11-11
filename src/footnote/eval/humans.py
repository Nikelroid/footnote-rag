"""Checks against the 2,400 answers that the FinanceBench authors labelled by hand.

Three uses. Recounting their table proves the loader and the label mapping are
right. Scoring the numeric check against their labels costs nothing. Scoring the
model judge against their labels is the test it has to pass before its grades
are used.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from .. import config
from ..data.financebench import HumanLabel, Question
from ..ledger import BudgetExceeded
from ..llm import SpendLimitReached
from . import numeric
from .judge import LABELS, Judge, JudgeError
from .metrics import cohen_kappa, confusion, wilson_interval
from .runner import _Appender, _read_jsonl


def paper_table(labels: list[HumanLabel]) -> dict[tuple[str, str], Counter]:
    table: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for h in labels:
        table[(h.model_name, h.eval_mode)][h.label] += 1
    return dict(table)


def numeric_agreement(labels: list[HumanLabel], questions: dict[str, Question], split: str | None = None) -> dict:
    """How often the numeric check and the human label agree on whether an answer is correct."""
    table = [[0, 0], [0, 0]]  # rows: human not correct, human correct. columns: no match, match.
    for h in labels:
        q = questions[h.question_id]
        if split is not None and q.split != split:
            continue
        verdict = numeric.grade(q.answer, h.model_answer, q.question)
        if verdict.applicable:
            table[int(h.label == "correct")][int(bool(verdict.match))] += 1
    n = sum(map(sum, table))
    agree = table[0][0] + table[1][1]
    lo, hi = wilson_interval(agree, n)
    return {"n": n, "agreement": agree / n if n else None, "lo": lo, "hi": hi, "kappa": cohen_kappa(table), "table": table}


def sample_labels(
    labels: list[HumanLabel], questions: dict[str, Question], n: int, split: str | None, seed: int = config.SPLIT_SEED
) -> list[HumanLabel]:
    """A fixed random sample, spread evenly over the sixteen model configurations."""
    pool: dict[tuple[str, str], list[HumanLabel]] = defaultdict(list)
    for h in labels:
        if split is None or questions[h.question_id].split == split:
            pool[(h.model_name, h.eval_mode)].append(h)
    rng = random.Random(seed)
    per_group = max(1, n // len(pool))
    picked: list[HumanLabel] = []
    for key in sorted(pool):
        group = sorted(pool[key], key=lambda h: h.question_id)
        rng.shuffle(group)
        picked.extend(group[:per_group])
    return picked


def judge_agreement(
    judge: Judge,
    sample: list[HumanLabel],
    questions: dict[str, Question],
    out_dir: Path,
    log: Callable[[str], None] = print,
) -> dict:
    """Grade the sample with the judge and compare with the human labels. Resumable."""
    rows_path = out_dir / "rows.jsonl"
    rows = _Appender(rows_path)
    done = {(r["question_id"], r["model_name"], r["eval_mode"]) for r in _read_jsonl(rows_path)}
    failures = 0
    for h in sample:
        key = (h.question_id, h.model_name, h.eval_mode)
        if key in done:
            continue
        q = questions[h.question_id]
        try:
            verdict = judge.grade(q.question, q.answer, q.justification, h.model_answer, run_id=out_dir.name)
        except (BudgetExceeded, SpendLimitReached):
            raise
        except JudgeError as err:
            failures += 1
            log(f"  judge failed on {key}: {err}")
            continue
        rows.write(
            {
                "question_id": h.question_id,
                "model_name": h.model_name,
                "eval_mode": h.eval_mode,
                "split": q.split,
                "human": h.label,
                "judge": verdict.label,
                "reason": verdict.reason,
                "grader": verdict.grader,
                "cost_usd": verdict.response.cost_usd if verdict.response else 0.0,
            }
        )
    graded = _read_jsonl(rows_path)
    summary = summarize_agreement(graded)
    summary["judge_failures"] = failures
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def summarize_agreement(rows: list[dict]) -> dict:
    table = confusion([r["human"] for r in rows], [r["judge"] for r in rows], LABELS)
    n = len(rows)
    agree = sum(table[i][i] for i in range(len(LABELS)))
    lo, hi = wilson_interval(agree, n)
    per_label = {}
    for i, label in enumerate(LABELS):
        human_total = sum(table[i])
        judge_total = sum(row[i] for row in table)
        per_label[label] = {
            "recall": table[i][i] / human_total if human_total else None,
            "precision": table[i][i] / judge_total if judge_total else None,
            "human_n": human_total,
        }
    human_correct = sum(table[0]) / n if n else None
    judge_correct = sum(row[0] for row in table) / n if n else None
    return {
        "n": n,
        "agreement": agree / n if n else None,
        "lo": lo,
        "hi": hi,
        "kappa": cohen_kappa(table),
        "labels": list(LABELS),
        "table": table,
        "per_label": per_label,
        # The judge matters through the accuracy it reports, so its bias on that number is shown directly.
        "accuracy_by_humans": human_correct,
        "accuracy_by_judge": judge_correct,
        "cost_usd": sum(r["cost_usd"] for r in rows),
        "disagreements": [r for r in rows if r["human"] != r["judge"]],
    }


def audit(questions: list[Question], labels: list[HumanLabel], manifest: dict) -> dict:
    """Checks over the whole question set: duplicates, balance, missing fields and evidence."""
    texts = Counter(q.question.strip().lower() for q in questions)
    pdfs = {f["path"] for f in manifest["files"] if f["path"].startswith("pdfs/")}
    docs = sorted({q.doc_name for q in questions})
    numeric_refs = [q for q in questions if numeric.parse_gold(q.answer)]
    return {
        "questions": len(questions),
        "distinct_ids": len({q.id for q in questions}),
        "duplicate_questions": sum(1 for count in texts.values() if count > 1),
        "by_type": dict(Counter(q.question_type for q in questions)),
        "by_split": dict(Counter(q.split for q in questions)),
        "split_by_type": {s: dict(Counter(q.question_type for q in questions if q.split == s)) for s in ("dev", "test")},
        "companies": len({q.company for q in questions}),
        "filings": len(docs),
        "filings_missing_pdf": [d for d in docs if f"pdfs/{d}.pdf" not in pdfs],
        "filings_in_repo": len(pdfs),
        "pdf_bytes_needed": sum(f["size"] for f in manifest["files"] if f["path"] in {f"pdfs/{d}.pdf" for d in docs}),
        "pdf_bytes_all": sum(f["size"] for f in manifest["files"] if f["path"].startswith("pdfs/")),
        "empty_answers": sum(1 for q in questions if not q.answer.strip()),
        "missing_justification": sum(1 for q in questions if not q.justification),
        "no_evidence": sum(1 for q in questions if not q.evidence),
        "evidence_pages": dict(Counter(len(q.evidence) for q in questions)),
        "evidence_in_other_filing": sum(1 for q in questions for e in q.evidence if e.doc_name != q.doc_name),
        "single_number_references": len(numeric_refs),
        "answer_chars": {"min": min(len(q.answer) for q in questions), "max": max(len(q.answer) for q in questions)},
        "human_labels": len(labels),
        "human_label_counts": dict(Counter(h.label for h in labels)),
        "human_labels_without_question": sum(1 for h in labels if h.question_id not in {q.id for q in questions}),
    }
