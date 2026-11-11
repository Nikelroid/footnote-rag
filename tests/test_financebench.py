"""Checks on the real benchmark files. Skipped until `footnote data fetch` has been run."""

import hashlib

import pytest

from footnote.data import download
from footnote.data.financebench import default_root, load_human_labels, load_questions

pytestmark = pytest.mark.skipif(
    not (default_root() / "data" / "financebench_open_source.jsonl").exists(), reason="benchmark data not downloaded"
)


def test_downloaded_files_match_the_pinned_manifest():
    manifest = download.load_manifest()
    assert download.verify(manifest, download.core_entries(manifest)) == []


def test_question_set_shape():
    from footnote.eval.humans import audit

    report = audit(load_questions(), load_human_labels(), download.load_manifest())
    assert report["questions"] == report["distinct_ids"] == 150 and report["duplicate_questions"] == 0
    assert report["by_type"] == {"metrics-generated": 50, "domain-relevant": 50, "novel-generated": 50}
    assert report["filings"] == 84 and report["companies"] == 32
    assert report["filings_missing_pdf"] == [] and report["no_evidence"] == 0 and report["empty_answers"] == 0
    assert report["evidence_in_other_filing"] == 0
    assert report["human_labels"] == 2400 and report["human_labels_without_question"] == 0


def test_split_is_stratified_and_frozen():
    questions = load_questions()
    dev = sorted(q.id for q in questions if q.split == "dev")
    assert len(dev) == 51 and len(questions) - len(dev) == 99
    for kind in ("metrics-generated", "domain-relevant", "novel-generated"):
        assert sum(1 for q in questions if q.split == "dev" and q.question_type == kind) == 17
    # If this changes, dev and test results from before and after cannot be compared.
    assert hashlib.sha256("\n".join(dev).encode()).hexdigest()[:16] == "0874fb193fd56cef"


def test_human_labels_reproduce_the_published_table():
    from footnote.eval.humans import paper_table

    table = paper_table(load_human_labels())
    published = {  # FinanceBench paper, Table 2, GPT-4-Turbo: correct, incorrect
        "closedBook": (14, 5),
        "sharedStore": (29, 20),
        "singleStore": (75, 17),
        "inContext_reverse": (118, 26),
        "oracle": (128, 22),
    }
    for mode, (correct, incorrect) in published.items():
        counts = table[("gpt-4-1106-preview", mode)]
        assert (counts["correct"], counts["incorrect"]) == (correct, incorrect)
        assert counts["refused"] == 150 - correct - incorrect
    assert len(table) == 16 and all(sum(c.values()) == 150 for c in table.values())


def test_every_labelled_answer_is_kept_as_text():
    # One answer is stored as the integer 0. Dropping it would turn a correct answer into an empty one.
    assert all(h.model_answer.strip() for h in load_human_labels())


def test_numeric_check_agrees_with_humans():
    from footnote.eval.humans import numeric_agreement

    result = numeric_agreement(load_human_labels(), {q.id: q for q in load_questions()})
    assert result["n"] == 832 and result["agreement"] > 0.95 and result["kappa"] > 0.9
