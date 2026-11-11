import json

import pytest

from conftest import make_question, scripted_llm
from footnote.data.financebench import HumanLabel
from footnote.eval.humans import judge_agreement, numeric_agreement, paper_table, sample_labels, summarize_agreement
from footnote.eval.judge import Judge


def labelled(questions):
    """Three answers per question from two pretend models: a right one, a wrong one and a refusal."""
    out = []
    for q in questions:
        for model in ("model-a", "model-b"):
            out.append(HumanLabel(q.id, model, "oracle", q.question, q.answer, f"It is {q.answer}.", "correct"))
            out.append(HumanLabel(q.id, model, "sharedStore", q.question, q.answer, "It is $7.00.", "incorrect"))
            out.append(HumanLabel(q.id, model, "closedBook", q.question, q.answer, "I cannot know that.", "refused"))
    return out


def test_paper_table_counts_labels_per_setting(questions):
    table = paper_table(labelled(questions))
    assert table[("model-a", "oracle")]["correct"] == 6 and table[("model-b", "closedBook")]["refused"] == 6


def test_numeric_agreement_on_clean_labels(questions):
    result = numeric_agreement(labelled(questions), {q.id: q for q in questions})
    assert result["n"] == 36 and result["agreement"] == 1.0 and result["kappa"] == 1.0
    assert result["table"] == [[24, 0], [0, 12]]
    assert numeric_agreement(labelled(questions), {q.id: q for q in questions}, split="dev")["n"] == 12


def test_sample_is_fixed_spread_over_settings_and_respects_the_split(questions):
    by_id = {q.id: q for q in questions}
    labels = labelled(questions)
    first = sample_labels(labels, by_id, n=12, split=None)
    assert first == sample_labels(labels, by_id, n=12, split=None) and len(first) == 12
    assert {(h.model_name, h.eval_mode) for h in first} == {(m, e) for m in ("model-a", "model-b") for e in ("oracle", "sharedStore", "closedBook")}
    assert all(by_id[h.question_id].split == "test" for h in sample_labels(labels, by_id, n=12, split="test"))


def test_judge_agreement_is_resumable_and_reports_every_disagreement(questions, tmp_path):
    by_id = {q.id: q for q in questions}
    labels = labelled(questions)
    # One human label is flipped, so the judge and the humans disagree on exactly one answer.
    noisy = [HumanLabel(h.question_id, h.model_name, h.eval_mode, h.question, h.gold_answer, h.model_answer, "incorrect" if i == 0 else h.label) for i, h in enumerate(labels)]
    llm = scripted_llm(questions)
    summary = judge_agreement(Judge(llm), noisy, by_id, tmp_path, log=lambda s: None)
    assert summary["n"] == 36 and summary["agreement"] == pytest.approx(35 / 36)
    assert summary["table"] == [[11, 0, 0], [1, 12, 0], [0, 0, 12]]  # rows are human labels, columns are judge labels
    assert len(summary["disagreements"]) == 1 and summary["disagreements"][0]["human"] == "incorrect"
    assert summary["accuracy_by_humans"] == pytest.approx(11 / 36) and summary["accuracy_by_judge"] == pytest.approx(12 / 36)
    assert summary["cost_usd"] == pytest.approx(llm.ledger.spent("grading")) and summary["cost_usd"] > 0
    assert json.loads((tmp_path / "summary.json").read_text())["n"] == 36
    calls = len(llm.calls)
    again = judge_agreement(Judge(llm), noisy, by_id, tmp_path, log=lambda s: None)
    assert len(llm.calls) == calls and again["n"] == 36  # nothing is graded or paid for twice


def test_per_label_precision_and_recall():
    rows = [{"human": h, "judge": j, "cost_usd": 0.0} for h, j in [("correct", "correct")] * 8 + [("correct", "incorrect")] * 2 + [("incorrect", "incorrect")] * 5 + [("refused", "refused")] * 5]
    s = summarize_agreement(rows)
    assert s["per_label"]["correct"] == {"recall": 0.8, "precision": 1.0, "human_n": 10}
    assert s["per_label"]["incorrect"]["precision"] == pytest.approx(5 / 7) and s["agreement"] == 0.9
