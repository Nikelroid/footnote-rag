"""Load FinanceBench: the 150 public questions, the document table and the human-labelled model answers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .. import config

LABELS = {"Correct Answer": "correct", "Incorrect Answer": "incorrect", "Refusal": "refused"}


@dataclass(frozen=True)
class Evidence:
    doc_name: str
    page: int  # zero-indexed page of the PDF, as the dataset gives it
    text: str
    full_page_text: str


@dataclass(frozen=True)
class Question:
    id: str
    company: str
    doc_name: str
    question_type: str
    question_reasoning: str | None
    question: str
    answer: str
    justification: str | None
    evidence: tuple[Evidence, ...]
    doc_type: str | None
    doc_period: str | None
    split: str  # "dev" or "test"

    @property
    def tags(self) -> list[str]:
        return [self.question_type, self.company, self.doc_type or "unknown"]

    @property
    def gold_pages(self) -> set[tuple[str, int]]:
        return {(e.doc_name, e.page) for e in self.evidence}


@dataclass(frozen=True)
class HumanLabel:
    """One model answer from the FinanceBench paper, with the label its authors gave it."""

    question_id: str
    model_name: str
    eval_mode: str
    question: str
    gold_answer: str
    model_answer: str
    label: str  # correct | incorrect | refused


class MissingData(RuntimeError):
    """A benchmark file is needed and has not been downloaded yet."""


FETCH_HINT = "Run `footnote data fetch --pdfs needed` first."


def default_root() -> Path:
    return config.RAW_DIR / "financebench"


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        raise MissingData(f"{path} is not there yet. {FETCH_HINT}")
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def assign_splits(strata: dict[str, list[str]], seed: int = config.SPLIT_SEED, dev_fraction: float = config.DEV_FRACTION) -> dict[str, str]:
    """A fixed random split, drawn inside each stratum. It never looks at any score."""
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        rng.shuffle(ids)
        n_dev = round(len(ids) * dev_fraction)
        for i, qid in enumerate(ids):
            out[qid] = "dev" if i < n_dev else "test"
    return out


def load_documents(root: Path | None = None) -> dict[str, dict]:
    root = root or default_root()
    return {row["doc_name"]: row for row in _rows(root / "data" / "financebench_document_information.jsonl")}


def load_questions(root: Path | None = None) -> list[Question]:
    root = root or default_root()
    rows = _rows(root / "data" / "financebench_open_source.jsonl")
    docs = load_documents(root)
    strata: dict[str, list[str]] = {}
    for row in rows:
        strata.setdefault(row["question_type"], []).append(row["financebench_id"])
    splits = assign_splits(strata)
    questions = []
    for row in rows:
        doc = docs.get(row["doc_name"], {})
        questions.append(
            Question(
                id=row["financebench_id"],
                company=row["company"],
                doc_name=row["doc_name"],
                question_type=row["question_type"],
                question_reasoning=row.get("question_reasoning"),
                question=row["question"],
                answer=row["answer"],
                justification=row.get("justification"),
                evidence=tuple(
                    Evidence(
                        doc_name=e["doc_name"],
                        page=int(e["evidence_page_num"]),
                        text=e["evidence_text"],
                        full_page_text=e["evidence_text_full_page"],
                    )
                    for e in row["evidence"]
                ),
                doc_type=doc.get("doc_type"),
                doc_period=str(doc["doc_period"]) if doc.get("doc_period") is not None else None,
                split=splits[row["financebench_id"]],
            )
        )
    return sorted(questions, key=lambda q: q.id)


def load_human_labels(root: Path | None = None) -> list[HumanLabel]:
    root = root or default_root()
    paths = sorted((root / "results").glob("*.jsonl"))
    if not paths:
        raise MissingData(f"There are no labelled answers in {root / 'results'} yet. {FETCH_HINT}")
    out = []
    for path in paths:
        for row in _rows(path):
            out.append(
                HumanLabel(
                    question_id=row["financebench_id"],
                    model_name=row["model_name"],
                    eval_mode=row["eval_mode"],
                    question=row["question"],
                    gold_answer=str(row["gold_answer"]),
                    # One answer in the files is the bare number 0, stored as an integer.
                    model_answer="" if row["model_answer"] is None else str(row["model_answer"]),
                    label=LABELS[row["label"]],
                )
            )
    return out
