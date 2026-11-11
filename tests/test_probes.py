import pytest

from conftest import make_question
from footnote.corpus.edits import apply_edit
from footnote.corpus.probes import build_probes, clean_lines, probe_edit, table_lines
from footnote.data.financebench import default_root, load_questions
from footnote.eval import numeric

# The benchmark's page text puts every table cell on its own line.
PAGE = "\n".join(
    [
        "Consolidated Statement of Cash Flows", " ", "(Millions)", " ", "2018", " ", "2017", " ",
        "Net income including noncontrolling interest", " ", "$", "5,363 ", "$", "4,869 ",
        "Adjustments to reconcile net income to net cash", "provided by operating activities", " ",
        "Depreciation and amortization", " ", "1,488 ", " ", "1,544 ",
        "Gain on sale of businesses", " ", "(545) ", " ", "(586) ",
        "Deferred income taxes", " ", chr(0x2014), " ", "2,109 ",
        "Purchases of property, plant and equipment (PP&E)", " ", "(1,577) ", " ", "(1,373) ",
        "Effective tax rate", " ", "23.4% ", " ", "35.5% ",
        "Other operating items", " ", "1,373 ", " ", "9,001 ",
        "Other financing items", " ", "7,070 ", " ", "8,080 ",
        "Other financing items", " ", "6,060 ", " ", "5,050 ",
    ]
)


def test_table_lines_pair_each_label_with_its_first_amount():
    pairs = dict(table_lines(PAGE))
    assert pairs["Net income including noncontrolling interest"] == "5,363"
    assert pairs["Depreciation and amortization"] == "1,488"
    assert pairs["Purchases of property, plant and equipment (PP&E)"] == "1,577"
    assert "provided by operating activities" not in pairs  # the tail of a wrapped label
    assert "Deferred income taxes" not in pairs  # its first column is an empty cell
    assert "Consolidated Statement of Cash Flows" not in pairs  # a heading with no amount of its own


def test_clean_lines_keep_only_unambiguous_probes():
    labels = [label for label, _ in clean_lines(PAGE)]
    assert labels == ["Depreciation and amortization", "Purchases of property, plant and equipment (PP&E)"]
    assert "Net income including noncontrolling interest" not in labels  # a sum of other lines
    assert "Gain on sale of businesses" not in labels  # 545 is too small to be distinctive
    assert "Effective tax rate" not in labels  # not a whole amount
    assert "Other operating items" not in labels  # 1,373 appears twice on the page
    assert "Other financing items" not in labels  # the label appears twice


def test_probes_are_fixed_and_their_edits_have_a_known_answer():
    import dataclasses

    q = make_question(0)
    q = dataclasses.replace(q, evidence=(dataclasses.replace(q.evidence[0], full_page_text=PAGE),))
    probes = build_probes([q], per_page=2)
    assert probes == build_probes([q], per_page=2) and len(probes) == 2 and len({p.probe_id for p in probes}) == 2
    for probe in probes:
        assert probe.label in probe.question and probe.doc_name in probe.question
        edit = probe_edit(probe, PAGE)
        changed = apply_edit(PAGE, edit)
        assert edit.new_answer != probe.answer and edit.new_answer in changed and probe.answer not in changed
        assert changed.count("\n") == PAGE.count("\n")  # one cell changed, nothing else moved
        assert numeric.grade(edit.new_answer, f"Final answer: {edit.new_answer}").match
        assert not numeric.grade(edit.new_answer, f"Final answer: {edit.old_answer}").match


@pytest.mark.skipif(not (default_root() / "data" / "financebench_open_source.jsonl").exists(), reason="benchmark data not downloaded")
def test_real_evidence_pages_give_enough_probes():
    questions = load_questions()
    probes = build_probes(questions)
    pages = {(e.doc_name, e.page): e.full_page_text for q in questions for e in q.evidence}
    assert len(pages) == 168 and len({(p.doc_name, p.page) for p in probes}) == 89 and len(probes) == 169
    assert all(probe_edit(p, pages[(p.doc_name, p.page)]) is not None for p in probes)
    capex = [p for p in build_probes(questions, per_page=50) if p.doc_name == "3M_2018_10K" and "(PP&E)" in p.label]
    assert capex and capex[0].answer == "1,577"  # the benchmark's own capital expenditure question, found as a table line
