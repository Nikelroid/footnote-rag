import json

import pytest

from conftest import scripted_llm
from footnote.eval.judge import Judge, rubric_version
from footnote.eval.report import IncompleteRun, breakdown, compare, judge_overlap, load_run, render, render_breakdown, summarize
from footnote.eval.runner import RunAborted, RunLocked, _locked, answer_run, grade_run, run_dir_for
from footnote.ledger import BudgetExceeded, Ledger
from footnote.llm import FakeLLM
from footnote.pricing import worst_case_cost_usd
from footnote.systems.bounds import ClosedBook, Oracle


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run_and_grade(system, questions, llm, tmp_path, reps=2):
    run_dir = answer_run(system, questions, llm, reps=reps, runs_dir=tmp_path, concurrency=2, log=lambda s: None)
    grade_run(run_dir, {q.id: q for q in questions}, Judge(llm), concurrency=2, log=lambda s: None)
    return run_dir


def test_reference_answers_score_100_and_every_row_is_complete(questions, tmp_path):
    llm = scripted_llm(questions, "right")
    run_dir = run_and_grade(Oracle(), questions, llm, tmp_path)
    answers = rows(run_dir / "answers.jsonl")
    assert len(answers) == 12 and {(r["case_id"], r["rep"]) for r in answers} == {(q.id, rep) for q in questions for rep in (0, 1)}
    for row in answers:
        assert row["status"] == "ok" and row["model"] == "claude-haiku-5-5" and row["n_calls"] == 1
        assert row["cost_usd"] > 0 and row["usage"]["input_tokens"] > 0 and row["latency_s"] > 0
        assert row["retrieval"]["hit@1"] == 1.0
        trace = json.loads((run_dir / "traces" / f"{row['case_id']}_rep{row['rep']}.json").read_text())
        assert [t["role"] for t in trace] == ["system", "user", "assistant"]
    summary = summarize(load_run(run_dir))
    assert summary["correct"]["mean"] == 1.0 and summary["incorrect"]["mean"] == 0.0 and summary["flip_rate"] == 0.0
    assert summary["numeric_check"] == {"n": 12, "agree": 1.0, "disagreements": []}
    assert summary["judge_cost_total_usd"] > 0
    assert (summary["n_expected"], summary["n_missing_answers"], summary["n_ungraded"], summary["n_graded"]) == (12, 0, 0, 12)
    assert set(summary["by_split"]) == {"dev", "test"}


def test_cost_is_counted_from_the_tokens_so_a_corrected_price_reaches_the_table(questions, tmp_path):
    run = load_run(run_and_grade(Oracle(), questions, scripted_llm(questions, "right"), tmp_path))
    total = summarize(run)["answer_cost_total_usd"]
    assert total == pytest.approx(sum(row["cost_usd"] for row in run.answers))  # the price table has not changed, so both agree
    for row in run.answers:  # dollars written on a day when the price table was wrong
        row["cost_usd"] /= 2
    assert summarize(run)["answer_cost_total_usd"] == pytest.approx(total)


@pytest.mark.parametrize("style,label", [("wrong", "incorrect"), ("refuse", "refused"), ("empty", "incorrect")])
def test_null_answers_score_zero(questions, tmp_path, style, label):
    llm = scripted_llm(questions, style)
    summary = summarize(load_run(run_and_grade(ClosedBook(), questions, llm, tmp_path)))
    assert summary["correct"]["mean"] == 0.0 and summary[label]["mean"] == 1.0


def test_empty_answers_never_reach_the_judge(questions, tmp_path):
    llm = scripted_llm(questions, "empty")
    run_and_grade(ClosedBook(), questions, llm, tmp_path, reps=1)
    assert all(call.json_schema is None for call in llm.calls)


def test_resume_does_not_pay_twice(questions, tmp_path):
    llm = scripted_llm(questions, "right")
    run_and_grade(Oracle(), questions, llm, tmp_path)
    before = len(llm.calls)
    run_and_grade(Oracle(), questions, llm, tmp_path)
    assert len(llm.calls) == before


def test_a_transport_failure_is_an_error_row_not_a_wrong_answer(questions, tmp_path):
    llm = scripted_llm(questions, "right", fail_times=1)
    run_dir = answer_run(Oracle(), questions, llm, reps=1, runs_dir=tmp_path, concurrency=1, log=lambda s: None)
    assert len(rows(run_dir / "answers.jsonl")) == 5
    errors = rows(run_dir / "errors.jsonl")
    assert len(errors) == 1 and errors[0]["class"] == "harness_error" and errors[0]["stage"] == "answer"
    run_and_grade(Oracle(), questions, llm, tmp_path, reps=1)  # the retry fills the gap
    summary = summarize(load_run(run_dir))
    assert summary["n_answers"] == 6 and summary["n_missing_answers"] == 0 and summary["correct"]["mean"] == 1.0


def test_cut_off_answers_are_counted_but_not_graded(questions, tmp_path):
    llm = scripted_llm(questions, "right", stop_reason="max_tokens")
    run_dir = answer_run(Oracle(), questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    llm.stop_reason = "end_turn"
    grade_run(run_dir, {q.id: q for q in questions}, Judge(llm), log=lambda s: None)
    assert all(r["status"] == "truncated" for r in rows(run_dir / "answers.jsonl"))
    summary = summarize(load_run(run_dir))
    assert summary["n_not_gradable"] == 6 and summary["n_graded"] == 0 and summary["n_cases_scored"] == 0
    assert "6 answers were cut off or too long for the model" in render([summary], [])


def test_running_out_of_budget_stops_the_run_without_scoring_anything(questions, tmp_path):
    # Room for one request at its worst-case price, and a little more. A few real requests use up the rest.
    worst = max(worst_case_cost_usd("claude-haiku-5-5", Oracle().request(q).prompt_chars(), 4096) for q in questions)
    cap = worst * 1.02
    llm = scripted_llm(questions, "right", ledger=Ledger(":memory:", cap_usd=cap, lines={"development": cap}))
    with pytest.raises(BudgetExceeded):
        answer_run(Oracle(), questions, llm, reps=3, runs_dir=tmp_path, concurrency=1, log=lambda s: None)
    run_dir = run_dir_for(Oracle(), questions, tmp_path)
    assert len(rows(run_dir / "answers.jsonl")) < 18
    assert {e["class"] for e in rows(run_dir / "errors.jsonl")} == {"budget"}
    assert 0 < llm.ledger.spent() <= cap


def test_an_answer_from_the_wrong_model_is_rejected_but_still_billed(questions, tmp_path):
    llm = scripted_llm(questions, "right", served_model="claude-sonnet-5-5")
    run_dir = answer_run(Oracle(), questions[:1], llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    assert rows(run_dir / "answers.jsonl") == []
    assert rows(run_dir / "errors.jsonl")[0]["class"] == "served_model_mismatch"
    assert llm.ledger.spent() > 0


def test_a_different_system_gets_a_different_run_directory(questions, tmp_path):
    assert run_dir_for(Oracle(), questions, tmp_path) != run_dir_for(Oracle(model="claude-sonnet-5-5"), questions, tmp_path)
    assert run_dir_for(Oracle(), questions, tmp_path) != run_dir_for(Oracle(), questions[:3], tmp_path)
    assert run_dir_for(Oracle(), questions, tmp_path) == run_dir_for(Oracle(), list(reversed(questions)), tmp_path)


def test_paired_comparison_and_table(questions, tmp_path):
    floor = load_run(run_and_grade(ClosedBook(), questions, scripted_llm(questions, "refuse"), tmp_path))
    ceiling = load_run(run_and_grade(Oracle(), questions, scripted_llm(questions, "right"), tmp_path))
    diff = compare(floor, ceiling)
    assert diff["diff"] == 1.0 and diff["n_cases"] == 6 and diff["a"] == "floor" and diff["b"] == "ceiling"
    table = render([summarize(floor), summarize(ceiling)], [diff])
    assert "| floor | 0.0% (0.0 to 39.0) |" in table and "| 12 of 12 |" in table  # six questions cannot prove a rate of zero
    assert "| floor to ceiling | +100.0 points |" in table and "Judge: judge-" in table


def test_a_half_finished_run_cannot_pass_as_a_full_one(questions, tmp_path):
    llm = scripted_llm(questions, "right")
    run_dir = run_and_grade(Oracle(), questions, llm, tmp_path, reps=1)
    answer_run(Oracle(), questions, llm, reps=3, runs_dir=tmp_path, log=lambda s: None)  # two more runs, not graded yet
    with pytest.raises(IncompleteRun, match="12 answers are not graded"):
        summarize(load_run(run_dir))
    partial = summarize(load_run(run_dir), allow_partial=True)
    assert (partial["n_graded"], partial["n_expected"], partial["n_ungraded"]) == (6, 18, 12)
    assert "PARTIAL: 0 answers missing and 12 not graded" in render([partial], []) and "| 6 of 18 |" in render([partial], [])


def test_missing_answers_are_reported_against_what_was_asked(questions, tmp_path):
    llm = scripted_llm(questions, "right", fail_times=2)
    run_dir = answer_run(Oracle(), questions, llm, reps=1, runs_dir=tmp_path, concurrency=1, log=lambda s: None)
    grade_run(run_dir, {q.id: q for q in questions}, Judge(llm), log=lambda s: None)
    with pytest.raises(IncompleteRun, match="2 of 6 answers are missing"):
        summarize(load_run(run_dir))


def test_rows_written_twice_are_refused(questions, tmp_path):
    run_dir = run_and_grade(Oracle(), questions, scripted_llm(questions, "right"), tmp_path, reps=1)
    answers = run_dir / "answers.jsonl"
    answers.write_text(answers.read_text() + answers.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="more than once"):
        load_run(run_dir)


def test_only_one_process_may_write_a_run(questions, tmp_path):
    llm = scripted_llm(questions, "right")
    run_dir = run_dir_for(Oracle(), questions, tmp_path)
    run_dir.mkdir(parents=True)
    with _locked(run_dir):
        with pytest.raises(RunLocked):
            answer_run(Oracle(), questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    assert llm.calls == []
    answer_run(Oracle(), questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)  # free again afterwards


def test_a_broken_setup_stops_after_a_few_failures(questions, tmp_path):
    llm = scripted_llm(questions, "right", fail_times=100)
    with pytest.raises(RunAborted, match="5 failures in a row"):
        answer_run(Oracle(), questions, llm, reps=3, runs_dir=tmp_path, concurrency=1, log=lambda s: None)
    assert len(llm.calls) == 5  # not all eighteen


def test_runs_graded_by_different_judges_are_not_compared(questions, tmp_path):
    by_id = {q.id: q for q in questions}
    floor_dir = answer_run(ClosedBook(), questions, scripted_llm(questions, "refuse"), reps=1, runs_dir=tmp_path, log=lambda s: None)
    ceiling_dir = answer_run(Oracle(), questions, scripted_llm(questions, "right"), reps=1, runs_dir=tmp_path, log=lambda s: None)
    grade_run(floor_dir, by_id, Judge(scripted_llm(questions), model="claude-sonnet-5-5"), log=lambda s: None)
    grade_run(ceiling_dir, by_id, Judge(scripted_llm(questions), model="claude-haiku-5-5"), log=lambda s: None)
    with pytest.raises(ValueError, match="different judges"):
        compare(load_run(floor_dir), load_run(ceiling_dir))
    with pytest.raises(ValueError, match="different judges"):
        render([summarize(load_run(floor_dir)), summarize(load_run(ceiling_dir))], [])


def test_one_repeat_can_be_graded_as_a_sample_and_the_rest_later(questions, tmp_path):
    llm = scripted_llm(questions, "right")
    by_id = {q.id: q for q in questions}
    run_dir = answer_run(Oracle(), questions, llm, reps=2, runs_dir=tmp_path, log=lambda s: None)
    path = grade_run(run_dir, by_id, Judge(llm), only_rep=0, log=lambda s: None)
    assert {(g["case_id"], g["rep"]) for g in rows(path)} == {(q.id, 0) for q in questions}
    before = len(llm.calls)
    grade_run(run_dir, by_id, Judge(llm), log=lambda s: None)
    assert len(llm.calls) == before + 6 and len(rows(path)) == 12  # the sample is not paid for again


def test_two_judges_are_compared_on_the_answers_both_graded(questions, tmp_path):
    by_id = {q.id: q for q in questions}
    run_dir = answer_run(Oracle(), questions, scripted_llm(questions, "right"), reps=2, runs_dir=tmp_path, log=lambda s: None)

    def harsh(req):  # a second judge that marks every answer to the first question wrong
        wrong = questions[0].answer in req.messages[-1]["content"]
        return json.dumps({"reason": "scripted", "label": "incorrect" if wrong else "correct"})

    grade_run(run_dir, by_id, Judge(scripted_llm(questions), model="claude-sonnet-5-5"), only_rep=0, log=lambda s: None)
    grade_run(run_dir, by_id, Judge(FakeLLM(responder=harsh, ledger=Ledger(":memory:")), model="claude-haiku-5-5"), log=lambda s: None)
    s = judge_overlap([run_dir], f"{rubric_version()}:claude-sonnet-5-5", f"{rubric_version()}:claude-haiku-5-5")
    assert s["n"] == 6 and s["agreement"] == pytest.approx(5 / 6)  # only the six answers both judges graded
    assert s["table"] == [[5, 1, 0], [0, 0, 0], [0, 0, 0]]  # rows are the reference judge, columns the candidate
    assert s["correct_by_reference"] == 1.0 and s["correct_by_candidate"] == pytest.approx(5 / 6)
    assert [(d["case_id"], d["rep"]) for d in s["disagreements"]] == [("q00", 0)]
    with pytest.raises(ValueError, match="expected one grades file"):
        load_run(run_dir)  # two judges graded this run, so the report has to be told which one to use
    assert summarize(load_run(run_dir, "claude-haiku-5-5"))["correct"]["mean"] == pytest.approx(5 / 6)


def test_answers_settled_by_a_rule_are_left_out_of_the_judge_comparison(questions, tmp_path):
    by_id = {q.id: q for q in questions}
    run_dir = answer_run(ClosedBook(), questions, scripted_llm(questions, "empty"), reps=1, runs_dir=tmp_path, log=lambda s: None)
    for model in ("claude-sonnet-5-5", "claude-haiku-5-5"):
        grade_run(run_dir, by_id, Judge(scripted_llm(questions), model=model), log=lambda s: None)
    s = judge_overlap([run_dir], f"{rubric_version()}:claude-sonnet-5-5", f"{rubric_version()}:claude-haiku-5-5")
    assert s["n"] == 0 and s["agreement"] is None  # an empty answer never reached either model


def test_a_changed_prompt_starts_a_new_run(questions, tmp_path):
    class Rearranged(Oracle):
        def request(self, question):
            request = super().request(question)
            request.messages[0]["content"] = "Question first: " + request.messages[0]["content"]
            return request

    assert run_dir_for(Rearranged(), questions, tmp_path) != run_dir_for(Oracle(), questions, tmp_path)


def test_a_search_that_finds_nothing_scores_zero(questions, tmp_path):
    class FindsNothing(Oracle):
        name = "finds-nothing"

        def answer(self, question, ctx):
            out = super().answer(question, ctx)
            out.retrieved = []
            return out

    llm = scripted_llm(questions, "right")
    run_dir = answer_run(FindsNothing(), questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    assert all(r["retrieval"]["hit@5"] == 0.0 for r in rows(run_dir / "answers.jsonl"))
    floor_dir = answer_run(ClosedBook(), questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    assert all(r["retrieval"] is None for r in rows(floor_dir / "answers.jsonl"))  # a system that never searches has no score


def test_a_prompt_too_long_for_the_model_is_recorded_once_and_not_graded(questions, tmp_path):
    llm = scripted_llm(questions, "right", stop_reason="model_context_window_exceeded")
    run_dir = answer_run(Oracle(), questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    before = len(llm.calls)
    answer_run(Oracle(), questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    assert len(llm.calls) == before  # it is not paid for again on every resume
    assert {r["status"] for r in rows(run_dir / "answers.jsonl")} == {"overflow"}


def test_a_score_is_split_by_what_the_model_knew_and_by_what_search_found(questions, tmp_path):
    known = {questions[0].question, questions[1].question}  # the model knows two answers without any document

    def from_memory(req):
        body = req.messages[-1]["content"]
        if req.json_schema is not None:
            return scripted_llm(questions).responder(req)
        question = next(q for q in questions if q.question in body)
        return f"Final answer: {question.answer}" if question.question in known else "Final answer: $1.00"

    class MissesOnePage(Oracle):  # a system that reads the right page for every question but the last
        name = "misses-one"

        def answer(self, question, ctx):
            out = super().answer(question, ctx)
            if question.id == questions[-1].id:
                out.retrieved = [{"doc": "OTHER", "page": 1, "chunk_id": None, "score": None}]
            return out

    by_id = {q.id: q for q in questions}
    floor_dir = answer_run(ClosedBook(), questions, FakeLLM(responder=from_memory, ledger=Ledger(":memory:")), reps=2, runs_dir=tmp_path, log=lambda s: None)
    system_dir = answer_run(MissesOnePage(), questions, scripted_llm(questions, "right"), reps=2, runs_dir=tmp_path, log=lambda s: None)
    for run_dir in (floor_dir, system_dir):
        grade_run(run_dir, by_id, Judge(scripted_llm(questions)), log=lambda s: None)
    b = breakdown(load_run(floor_dir), load_run(system_dir))
    assert b["by_memory"]["never"] == {"questions": 4, "correct": 1.0} and b["by_memory"]["always"] == {"questions": 2, "correct": 1.0}
    assert b["evidence_read"] == {"answers": 10, "correct": 10} and b["evidence_not_read"] == {"answers": 2, "correct": 2}
    assert (b["misses"], b["misses_with_evidence_read"]) == (0, 0)
    assert "| in none of its runs | 4 | 100.0% |" in render_breakdown(b) and "| no | 2 | 100.0% |" in render_breakdown(b)
