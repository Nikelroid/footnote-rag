import dataclasses
import json

import pytest

from conftest import make_question
from footnote.eval.runner import answer_run, run_dir_for
from footnote.ledger import Ledger
from footnote.llm import FakeLLM
from footnote.rag.graph import answer_request
from footnote.rag.retrieve import Hit
from footnote.rag.stale import DEFENCE, POLICIES, RULE, TESTS, AfterAmendment, build_cases, classify, render, summarize, systems
from test_probes import PAGE


@pytest.fixture
def cases():
    """Two statement lines as probes, and one benchmark question whose answer is printed on the page."""
    q = dataclasses.replace(make_question(0), answer="$1577.00")
    q = dataclasses.replace(q, evidence=(dataclasses.replace(q.evidence[0], full_page_text=PAGE),))
    return build_cases([q])


def reads_first_version(cases, misread: str | None = None) -> FakeLLM:
    """A stand-in model that answers with whichever of the two amounts comes first in the prompt."""
    by_question = {c.question.question: c for c in cases}

    def respond(req):
        body = req.messages[-1]["content"]
        case = next(c for question, c in by_question.items() if question in body)
        first = min((a for a in (case.old, case.new) if a in body), key=body.index)
        answer = "7" if case.question.id == misread else first
        return json.dumps({"reasoning": "Read from the page.", "found": True, "final_answer": answer, "pages": [{"document": case.doc_name, "page": case.page}]})

    return FakeLLM(responder=respond, ledger=Ledger(":memory:"))


def test_cases_are_fixed_and_each_page_differs_in_one_amount(cases):
    assert [c.question.question_type == "probe" for c in cases] == [True, True, False]
    assert len({c.question.id for c in cases}) == 3
    for case in cases:
        assert case.old in case.before and case.new not in case.before
        assert case.new in case.after and case.old not in case.after
        assert case.after.count("\n") == case.before.count("\n")  # one cell changed, nothing else moved
        assert case.filed == ("2021-02-15", "2021-08-15")  # the filing is for fiscal 2020
    assert cases[2].question.answer != "$1577.00" and cases[2].old == "1,577"  # the reference answer follows the amendment


def test_with_one_page_the_request_is_the_assistants_own(cases):
    case = cases[0]
    for policy, text in (("before", case.before), ("synced", case.after)):
        system = AfterAmendment({}, policy)
        assert system.request(case) == answer_request(system.model, case.question.question, [Hit(case.doc_name, case.page, text, 0.0)])


def test_two_versions_are_shown_in_the_order_asked_for_and_dated_only_when_the_policy_says_so(cases):
    case = cases[0]
    appended = AfterAmendment({}, "appended", old_first=False).request(case)
    body = appended.messages[0]["content"]
    assert body.count("<page ") == 2 and body.index(case.new) < body.index(case.old)
    assert "filed=" not in body and RULE not in appended.system
    dated = AfterAmendment({}, "dated").request(case)
    old_date, new_date = case.filed
    assert dated.messages[0]["content"].index(old_date) < dated.messages[0]["content"].index(new_date) and RULE not in dated.system
    assert AfterAmendment({}, "dated+rule").request(case).system.endswith(RULE)
    with pytest.raises(ValueError):
        AfterAmendment({}, "newest wins")


def test_every_setting_gets_its_own_run(cases, tmp_path):
    todo = systems(cases)
    assert len(todo) == 8 and len({s.name for s in todo}) == 8  # two page orders wherever both versions are shown
    questions = [c.question for c in cases]
    assert len({run_dir_for(s, questions, tmp_path) for s in todo}) == 8
    assert run_dir_for(todo[1], questions, tmp_path) != run_dir_for(systems(cases, "claude-sonnet-5-5")[1], questions, tmp_path)


@pytest.mark.parametrize(
    "answer,outcome,leads,shows_both",
    [
        ("The line shows 1,613.\nFinal answer: 1,613", "new", "new", False),
        ("One page says 1,488 and the later one 1,613.\nFinal answer: $1,613 million", "new", "new", True),
        ("Final answer: 1488", "old", "old", False),
        ("Final answer: (1,488)", "old", "old", False),
        ("Final answer: 1,613 (an earlier version showed 1,488)", "both", "new", True),
        ("Final answer: 1,488 (the later copy shows 1,613)", "both", "old", True),
        ("The pages disagree, 1,488 against 1,613.\nFinal answer: NOT FOUND", "declined", None, True),
        ("Final answer: 11,613", "other", None, False),
        ("It is 1,613.", "other", None, False),  # no final line, so the answer commits to nothing
    ],
)
def test_an_answer_is_sorted_by_the_amount_it_commits_to(answer, outcome, leads, shows_both):
    assert classify(answer, "1,488", "1,613") == {"outcome": outcome, "leads": leads, "shows_both": shows_both}


def test_a_model_that_reads_the_first_page_is_stale_exactly_when_the_old_page_comes_first(cases, tmp_path):
    llm = reads_first_version(cases)
    questions = [c.question for c in cases]
    for system in systems(cases):
        answer_run(system, questions, llm, reps=2, runs_dir=tmp_path, log=lambda s: None)
    summary = summarize(cases, runs_dir=tmp_path)
    probes = summary["groups"]["free"]  # the test page shows no total, so nothing on it can be checked by adding up
    assert (probes["cases"], probes["read_correctly_before"]) == (2, 2) and summary["groups"]["benchmark"]["read_correctly_before"] == 1
    rows = probes["settings"]
    assert set(rows) == {s.setting for s in systems(cases)} - {"before"}
    assert rows["synced"]["new"]["mean"] == 1.0 and rows["synced"]["n_answers"] == 4
    assert rows["appended-old-first"]["old"]["mean"] == 1.0 and rows["appended-old-first"]["old_unmarked"]["mean"] == 1.0
    assert rows["appended-old-first"]["shows_both"]["mean"] == 0.0  # this stand-in never mentions the other version
    assert rows["appended-old-first"]["leads_old"]["mean"] == 1.0 and rows["appended-new-first"]["leads_old"]["mean"] == 0.0
    assert rows["appended-new-first"]["new"]["mean"] == 1.0 and rows["appended-new-first"]["old"]["mean"] == 0.0
    table = render(summary)
    assert "| appended-old-first | 0.0% (0.0 to 65.8) | 100.0% (34.2 to 100.0) |" in table and "2 of 2 were read correctly" in table


def test_a_case_that_was_misread_before_the_change_is_left_out(cases, tmp_path):
    llm = reads_first_version(cases, misread=cases[2].question.id)
    questions = [c.question for c in cases]
    for system in systems(cases, policies=("before", "synced")):
        answer_run(system, questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    summary = summarize(cases, runs_dir=tmp_path)
    assert summary["groups"]["benchmark"]["read_correctly_before"] == 0 and summary["groups"]["benchmark"]["settings"]["synced"]["n_answers"] == 0
    assert "n/a" in render(summary)
    with pytest.raises(ValueError, match="unchanged page"):
        summarize(cases, model="claude-sonnet-5-5", runs_dir=tmp_path)


def test_policies_are_the_ones_the_command_line_offers():
    assert POLICIES == ("before", "synced", "appended", "dated", "dated+rule")


def test_the_two_kinds_of_edit_never_share_a_run(cases, tmp_path):
    questions = [c.question for c in cases]
    single, moved = systems(cases)[1], systems(cases, edit="moved")[1]
    assert (single.name, moved.name) == ("stale-synced", "moved-synced") and single.setting == moved.setting == "synced"
    assert "edit" not in single.describe() and moved.describe()["edit"] == "moved"
    assert run_dir_for(single, questions, tmp_path) != run_dir_for(moved, questions, tmp_path)
    llm = reads_first_version(cases)
    for system in systems(cases, policies=("before", "synced"), edit="moved"):
        answer_run(system, questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    assert summarize(cases, edit="moved", runs_dir=tmp_path)["groups"]["free"]["settings"]["synced"]["new"]["mean"] == 1.0
    with pytest.raises(ValueError, match="unchanged page"):
        summarize(cases, edit="single", runs_dir=tmp_path)  # nothing was run for the other kind of edit
    with pytest.raises(ValueError):
        build_cases([], "rewritten")


def test_a_reply_the_api_stopped_is_recorded_and_not_lost(cases, tmp_path):
    llm = FakeLLM(responder=lambda req: '{"reasoning": "The page sho', ledger=Ledger(":memory:"), stop_reason="refusal")
    questions = [c.question for c in cases]
    system = systems(cases, policies=("synced",))[0]
    run_dir = answer_run(system, questions, llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    rows = [json.loads(line) for line in (run_dir / "answers.jsonl").open()]
    assert len(rows) == 3 and {r["status"] for r in rows} == {"blocked"} and not (run_dir / "errors.jsonl").exists()
    assert classify(rows[0]["answer"], cases[0].old, cases[0].new)["outcome"] == "other"  # half a reply commits to nothing


def two_companies():
    """The same statement page for two companies, with different amounts, as two benchmark questions."""
    def company(i, scale):
        q = make_question(i)
        page = PAGE.replace("1,488", f"{1488 * scale:,}").replace("1,577", f"{1577 * scale:,}").replace("Consolidated", f"Company {i} Consolidated")
        return dataclasses.replace(q, evidence=(dataclasses.replace(q.evidence[0], doc_name=q.doc_name, full_page_text=page),))

    return [company(0, 1), company(1, 3)]


def test_the_wrong_document_test_hands_over_another_companys_page_with_the_same_line():
    cases = build_cases(two_companies(), "wrong")
    assert len(cases) == 4 and {c.group for c in cases} == {"wrong"}  # two lines, each way round
    case = next(c for c in cases if c.doc_name == "COMPANY0_2020_10K" and "Depreciation" in c.question.question)
    assert (case.old, case.new, case.other_doc) == ("1,488", "4,464", "COMPANY1_2020_10K")
    assert "Company 0" in case.before and "Company 1" in case.after and case.new not in case.before and case.old not in case.after
    honest = AfterAmendment({}, "honest", edit="wrong").request(case).messages[0]["content"]
    assert 'document="COMPANY1_2020_10K"' in honest and "COMPANY0_2020_10K" in honest.split("Question:")[1]  # the page says what it is, the question asks for the other filing
    mislabelled = AfterAmendment({}, "mislabelled", edit="wrong").request(case).messages[0]["content"]
    assert 'document="COMPANY0_2020_10K"' in mislabelled and "Company 1" in mislabelled  # only the page's own text gives it away
    assert AfterAmendment({}, "before", edit="wrong").request(case) == AfterAmendment({}, "before").request(case)


def test_the_injection_test_adds_one_sentence_and_the_defence_one_more():
    from test_sums import SHEET

    q = dataclasses.replace(make_question(0), evidence=(dataclasses.replace(make_question(0).evidence[0], full_page_text=SHEET),))
    cases = build_cases([q], "injected")
    assert cases and {c.group for c in cases} == {"injected"}
    case = cases[0]
    assert case.after.startswith(case.before) and case.after.count("\n") == case.before.count("\n") + 1  # the page itself is untouched
    note = case.after.split("\n")[-1]
    assert case.new in note and case.new not in case.before and "Note to AI assistants" in note
    plain, defended = (AfterAmendment({}, policy, edit="injected").request(case) for policy in ("plain", "defended"))
    assert plain.messages == defended.messages and not plain.system.endswith(DEFENCE) and defended.system.endswith(DEFENCE)


def test_each_test_has_its_own_settings_and_its_own_report_words(cases, tmp_path):
    assert TESTS["wrong"] == ("before", "honest", "mislabelled") and TESTS["injected"] == ("before", "plain", "defended")
    assert [s.name for s in systems(cases, edit="injected")] == ["injected-before", "injected-plain", "injected-defended"]
    with pytest.raises(ValueError):
        AfterAmendment({}, "honest")  # that setting belongs to another test
    wrong = build_cases(two_companies(), "wrong")
    llm = FakeLLM(responder=lambda req: json.dumps({"reasoning": "No such filing here.", "found": False, "final_answer": "", "pages": []}), ledger=Ledger(":memory:"))
    reads = reads_first_version(wrong)
    questions = [c.question for c in wrong]
    for system in systems(wrong, edit="wrong"):
        answer_run(system, questions, reads if system.policy == "before" else llm, reps=1, runs_dir=tmp_path, log=lambda s: None)
    summary = summarize(wrong, edit="wrong", runs_dir=tmp_path)
    block = summary["groups"]["wrong"]
    assert block["read_correctly_before"] == 4 and block["settings"]["honest"]["declined"]["mean"] == 1.0 and block["settings"]["mislabelled"]["leads_new"]["mean"] == 0.0
    table = render(summary)
    assert "Other company's amount only" in table and "| honest | 0.0%" in table and "handed another company's page" in table
