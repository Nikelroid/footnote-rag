"""Turn raw rows into the results table. Every number here is recomputed from the rows on disk.

A table is only printed for a complete run: every question, every repeat, every
answer graded, all by one judge. Anything else raises, unless the caller asks for
a partial table, and then the gaps are printed next to the numbers.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..pricing import Usage, cost_usd
from .metrics import cluster_bootstrap, cohen_kappa, confusion, flip_rate, paired_difference, wilson_interval
from .runner import UNGRADED, _read_jsonl, grades_path

OUTCOMES = ("correct", "incorrect", "refused")


def answer_cost(row: dict) -> float:
    """What an answer costs at the prices in `pricing.py`, counted again from the tokens of its calls.

    The token counts are what the API reported. The dollars stored next to them are
    right only if the price table was right on the day of the run, and for Gemini it
    was not. Counting from the tokens lets a corrected price reach every table.
    """
    return sum(cost_usd(call["model"], Usage(**call["usage"])) for call in row["calls"])


class IncompleteRun(ValueError):
    pass


@dataclass
class RunData:
    run_dir: Path
    config: dict
    answers: list[dict]
    grades: dict[tuple[str, int], dict]
    grader: str
    errors: list[dict]

    @property
    def name(self) -> str:
        return self.config["system"]["name"]


def _unique(rows: list[dict], what: str, run: str) -> dict[tuple[str, int], dict]:
    counts = Counter((r["case_id"], r["rep"]) for r in rows)
    repeated = [key for key, n in counts.items() if n > 1]
    if repeated:
        raise ValueError(f"{run}: {len(repeated)} {what} appear more than once, for example {repeated[0]}. The run was written twice.")
    return {(r["case_id"], r["rep"]): r for r in rows}


def load_run(run_dir: Path, grader_file: str | None = None) -> RunData:
    run_dir = Path(run_dir)
    files = sorted((run_dir / "grades").glob("*.jsonl")) if (run_dir / "grades").exists() else []
    if grader_file is not None:
        # The full file name works, and so does the judge model alone, such as claude-haiku-5-5.
        files = [f for f in files if f.stem == grader_file or f.stem.endswith(f"_{grader_file}")]
    if len(files) != 1:
        raise ValueError(f"{run_dir.name}: expected one grades file, found {[f.stem for f in files]}. Name the one to use.")
    answers = _read_jsonl(run_dir / "answers.jsonl")
    _unique(answers, "answers", run_dir.name)
    return RunData(
        run_dir=run_dir,
        config=json.loads((run_dir / "config.json").read_text()),
        answers=answers,
        grades=_unique(_read_jsonl(files[0]), "grades", run_dir.name),
        grader=files[0].stem,
        errors=_read_jsonl(run_dir / "errors.jsonl"),
    )


def completeness(run: RunData) -> dict:
    """Compare what is on disk with what the run was asked to produce."""
    reps = run.config.get("reps") or (max(r["rep"] for r in run.answers) + 1 if run.answers else 0)
    expected = {(case, rep) for case in run.config["question_ids"] for rep in range(reps)}
    answered = {(r["case_id"], r["rep"]) for r in run.answers}
    gradable = {(r["case_id"], r["rep"]) for r in run.answers if r["status"] not in UNGRADED}
    return {
        "expected": len(expected),
        "missing_answers": len(expected - answered),
        "ungraded": len(gradable - set(run.grades)),
        "not_gradable": len(answered - gradable),
        "unexpected_rows": len(answered - expected) + len(set(run.grades) - answered),
    }


def outcome_values(run: RunData, label: str, rows: list[dict] | None = None) -> dict[str, list[float]]:
    """For each question, one 0 or 1 per graded repeat."""
    values: dict[str, list[float]] = {}
    for row in run.answers if rows is None else rows:
        grade = run.grades.get((row["case_id"], row["rep"]))
        if grade is not None:
            values.setdefault(row["case_id"], []).append(1.0 if grade["label"] == label else 0.0)
    return values


def summarize(run: RunData, allow_partial: bool = False) -> dict:
    if not run.answers:
        raise IncompleteRun(f"{run.run_dir.name} has no answers")
    for row in run.answers:
        # A zero here is a broken runner, not a measurement.
        if row["n_calls"] < 1 or row["cost_usd"] <= 0 or row["latency_s"] <= 0:
            raise ValueError(f"{run.run_dir.name}: row {row['case_id']} rep {row['rep']} has an empty cost, latency or call count")
    gaps = completeness(run)
    if gaps["unexpected_rows"]:
        raise ValueError(f"{run.run_dir.name}: {gaps['unexpected_rows']} rows do not belong to this run")
    if (gaps["missing_answers"] or gaps["ungraded"]) and not allow_partial:
        raise IncompleteRun(
            f"{run.run_dir.name}: {gaps['missing_answers']} of {gaps['expected']} answers are missing and {gaps['ungraded']} "
            "answers are not graded. Finish the run, or ask for a partial table."
        )
    correct = outcome_values(run, "correct")
    out: dict = {
        "run": run.run_dir.name,
        "system": run.name,
        "model": run.config["system"].get("model"),
        "grader": run.grader,
        "n_cases": len(run.config["question_ids"]),
        "n_cases_scored": len(correct),
        "reps": run.config.get("reps") or max(r["rep"] for r in run.answers) + 1,
        "n_answers": len(run.answers),
        "n_graded": len(run.grades),
        **{f"n_{key}": value for key, value in gaps.items()},
    }
    for label in OUTCOMES:
        out[label] = cluster_bootstrap(outcome_values(run, label))
    out["flip_rate"] = flip_rate(correct)
    out["by_split"] = {
        split: cluster_bootstrap(outcome_values(run, "correct", [r for r in run.answers if r["split"] == split]))
        for split in sorted({r["split"] for r in run.answers})
    }
    out["by_type"] = {
        kind: cluster_bootstrap(outcome_values(run, "correct", [r for r in run.answers if r["tags"][0] == kind]))
        for kind in sorted({r["tags"][0] for r in run.answers})
    }
    costs = np.array([answer_cost(r) for r in run.answers])
    latency = np.array([r["latency_s"] for r in run.answers])
    judge_costs = [g["judge"]["cost_usd"] for g in run.grades.values() if "judge" in g]
    out["answer_cost_per_question_usd"] = float(costs.mean())
    out["answer_cost_total_usd"] = float(costs.sum())
    out["judge_cost_per_answer_usd"] = float(np.mean(judge_costs)) if judge_costs else 0.0
    out["judge_cost_total_usd"] = float(np.sum(judge_costs)) if judge_costs else 0.0
    out["latency_median_s"] = float(np.median(latency))
    out["latency_p90_s"] = float(np.quantile(latency, 0.9))
    out["prompt_tokens_mean"] = float(
        np.mean([sum(v for k, v in r["usage"].items() if k != "output_tokens") for r in run.answers])
    )
    out["output_tokens_mean"] = float(np.mean([r["usage"]["output_tokens"] for r in run.answers]))
    scored = [r["retrieval"] for r in run.answers if r.get("retrieval")]
    out["retrieval"] = {k: float(np.mean([s[k] for s in scored])) for k in scored[0]} if scored else None
    checked = [g for g in run.grades.values() if g["numeric"]["applicable"]]
    disagree = sorted((g["case_id"], g["rep"]) for g in checked if bool(g["numeric"]["match"]) != bool(g["correct"]))
    out["numeric_check"] = {
        "n": len(checked),
        "agree": (1 - len(disagree) / len(checked)) if checked else None,
        "disagreements": [list(key) for key in disagree],  # the answers to read by hand
    }
    return out


def same_grader(runs: list[RunData]) -> str:
    """Scores from different judges are not comparable, so a table or a comparison needs one judge for all runs."""
    graders = {run.grader for run in runs}
    if len(graders) != 1:
        raise ValueError(f"the runs were graded by different judges: {sorted(graders)}. Grade them with one judge first.")
    return graders.pop()


def judge_overlap(run_dirs: list[Path], reference: str, candidate: str) -> dict:
    """Compare two judges on the answers both of them graded. Rows are the reference judge, columns the candidate.

    The human labels test a judge on other models' answers. This tests it on the
    system's own answers, where a judge from the same model could be too kind.
    """
    rows = []
    for run_dir in map(Path, run_dirs):
        ref = {(g["case_id"], g["rep"]): g for g in _read_jsonl(grades_path(run_dir, reference))}
        cand = {(g["case_id"], g["rep"]): g for g in _read_jsonl(grades_path(run_dir, candidate))}
        for key in sorted(set(ref) & set(cand)):
            if "judge" not in ref[key] or "judge" not in cand[key]:
                continue  # settled by a rule, such as an empty answer. Neither model saw it.
            rows.append(
                {
                    "run": run_dir.name,
                    "case_id": key[0],
                    "rep": key[1],
                    "reference": ref[key]["label"],
                    "candidate": cand[key]["label"],
                    "reference_reason": ref[key]["reason"],
                    "candidate_reason": cand[key]["reason"],
                }
            )
    table = confusion([r["reference"] for r in rows], [r["candidate"] for r in rows], OUTCOMES)
    n = len(rows)
    agree = sum(table[i][i] for i in range(len(OUTCOMES)))
    lo, hi = wilson_interval(agree, n)
    return {
        "reference": reference,
        "candidate": candidate,
        "runs": sorted({r["run"] for r in rows}),
        "n": n,
        "agreement": agree / n if n else None,
        "lo": lo,
        "hi": hi,
        "kappa": cohen_kappa(table),
        "labels": list(OUTCOMES),
        "table": table,
        "correct_by_reference": sum(table[0]) / n if n else None,
        "correct_by_candidate": sum(row[0] for row in table) / n if n else None,
        "disagreements": [r for r in rows if r["reference"] != r["candidate"]],
    }


def compare(a: RunData, b: RunData) -> dict:
    same_grader([a, b])
    result = paired_difference(outcome_values(a, "correct"), outcome_values(b, "correct"))
    result.update({"a": a.name, "b": b.name})
    return result


def breakdown(floor: RunData, system: RunData) -> dict:
    """Two things the results table does not show.

    Does the system only work where the model already knew the answer? And are its
    misses search misses, or does it miss with the evidence page in front of it?
    """
    same_grader([floor, system])
    known = {case: sum(v) / len(v) for case, v in outcome_values(floor, "correct").items()}
    scores = outcome_values(system, "correct")
    groups = {"never": lambda s: s == 0, "sometimes": lambda s: 0 < s < 1, "always": lambda s: s == 1}
    by_memory = {}
    for name, belongs in groups.items():
        cases = [case for case, share in known.items() if belongs(share) and case in scores]
        by_memory[name] = {"questions": len(cases), "correct": float(np.mean([np.mean(scores[c]) for c in cases])) if cases else None}
    read = {True: [0, 0], False: [0, 0]}  # evidence page read or not: [answers, correct answers]
    for row in system.answers:
        grade = system.grades.get((row["case_id"], row["rep"]))
        if grade is None or not row.get("retrieval"):
            continue
        tally = read[row["retrieval"]["hit@10"] > 0]
        tally[0] += 1
        tally[1] += grade["label"] == "correct"
    misses = (read[True][0] - read[True][1]) + (read[False][0] - read[False][1])
    return {
        "floor": floor.name,
        "system": system.name,
        "by_memory": by_memory,
        "evidence_read": {"answers": read[True][0], "correct": read[True][1]},
        "evidence_not_read": {"answers": read[False][0], "correct": read[False][1]},
        "misses": misses,
        "misses_with_evidence_read": read[True][0] - read[True][1],
    }


def render_breakdown(b: dict) -> str:
    m, hit, miss = b["by_memory"], b["evidence_read"], b["evidence_not_read"]
    lines = [
        f"**Does the {b['system']} system only work where the model already knew the answer?**",
        "",
        "| Without documents, the model answers the question correctly | Questions | Correct with the system |",
        "|---|---|---|",
    ]
    for name, label in (("never", "in none of its runs"), ("sometimes", "in some of its runs"), ("always", "in all of its runs")):
        share = "n/a" if m[name]["correct"] is None else f"{100 * m[name]['correct']:.1f}%"
        lines.append(f"| {label} | {m[name]['questions']} | {share} |")
    lines += [
        "",
        "**Are its misses search misses?**",
        "",
        "| The evidence page was among the pages read | Answers | Correct |",
        "|---|---|---|",
        f"| yes | {hit['answers']} | {100 * hit['correct'] / max(1, hit['answers']):.1f}% |",
        f"| no | {miss['answers']} | {100 * miss['correct'] / max(1, miss['answers']):.1f}% |",
        "",
        f"{b['misses_with_evidence_read']} of the {b['misses']} answers that were not correct were given with the evidence page read.",
    ]
    return "\n".join(lines) + "\n"


def _pct(stat: dict) -> str:
    return f"{100 * stat['mean']:.1f}% ({100 * stat['lo']:.1f} to {100 * stat['hi']:.1f})"


def render(summaries: list[dict], comparisons: list[dict]) -> str:
    graders = sorted({s["grader"] for s in summaries})
    if len(graders) > 1:
        raise ValueError(f"the runs were graded by different judges: {graders}")
    lines = [
        "| System | Correct | Wrong | Refused | Flips | Evidence page read | $ per question | Median s | Graded answers |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        # No system hands the model more than 10 pages, so this is whether the evidence page was among the pages it read.
        hit = f"{100 * s['retrieval']['hit@10']:.1f}%" if s["retrieval"] else "n/a"
        flips = "n/a" if s["flip_rate"] != s["flip_rate"] else f"{100 * s['flip_rate']:.1f}%"
        lines.append(
            f"| {s['system']} | {_pct(s['correct'])} | {_pct(s['incorrect'])} | {_pct(s['refused'])} | {flips} | {hit} "
            f"| {s['answer_cost_per_question_usd']:.5f} | {s['latency_median_s']:.1f} | {s['n_graded']} of {s['n_expected']} |"
        )
    lines.append("")
    lines.append(f"Judge: {graders[0]}. Intervals are 95%, from resampling questions. Flips is the share of questions whose repeated runs disagree.")
    for s in summaries:
        splits = ", ".join(f"{name} {_pct(stat)}" for name, stat in s["by_split"].items() if stat["n_cases"])
        notes = [f"correct by split: {splits}"] if splits else []
        if s["numeric_check"]["n"]:
            notes.append(f"numeric check agrees with the judge on {100 * s['numeric_check']['agree']:.1f}% of {s['numeric_check']['n']} answers")
        if s["n_missing_answers"] or s["n_ungraded"]:
            notes.append(f"PARTIAL: {s['n_missing_answers']} answers missing and {s['n_ungraded']} not graded")
        if s["n_not_gradable"]:
            notes.append(f"{s['n_not_gradable']} answers were cut off or too long for the model, and are left out")
        lines.append(f"- {s['system']}: " + ", ".join(notes) + ".")
    if comparisons:
        lines += ["", "| Comparison | Change in correct | 95% interval | p | Questions |", "|---|---|---|---|---|"]
        for c in comparisons:
            lines.append(
                f"| {c['a']} to {c['b']} | {100 * c['diff']:+.1f} points | {100 * c['lo']:+.1f} to {100 * c['hi']:+.1f} "
                f"| {c['p_value']:.4f} | {c['n_cases']} |"
            )
    return "\n".join(lines) + "\n"
