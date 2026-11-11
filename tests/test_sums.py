import pytest

from footnote.corpus.edits import apply_edit
from footnote.corpus.probes import Probe
from footnote.corpus.sums import first_column, in_a_sum, move_edit, totals
from footnote.data.financebench import default_root, load_questions

# The benchmark's page text puts every table cell on its own line.
SHEET = "\n".join(
    [
        "Consolidated Balance Sheet", " ", "2020", " ", "2019", " ",
        "Cash and cash equivalents", " ", "$", "2,853 ", "$", "3,053 ",
        "Accounts receivable", " ", "5,020 ", " ", "4,911 ",
        "Finished goods", " ", "2,120 ", " ", "1,915 ",
        "Work in process", " ", "1,292 ", " ", "1,218 ",
        "Total inventories", " ", "3,412 ", " ", "3,133 ",
        "Total current assets", " ", "11,285 ", " ", "11,097 ",
        "Property, plant and equipment", " ", "24,873 ", " ", "24,914 ",
        "Less: Accumulated depreciation", " ", "(16,135) ", " ", "(16,048) ",
        "Property, plant and equipment net", " ", "8,738 ", " ", "8,866 ",
        "Total assets", " ", "$", "20,023 ", "$", "19,963 ",
        "Shares outstanding", " ", "4,321 ", " ", "4,300 ",
    ]
)


def probe(label, amount):
    return Probe(f"probe_{label[:4].lower()}", "SAMPLE_2020_10K", 3, label, f"What is the first amount on the line '{label}'?", amount)


def test_the_first_column_is_read_with_its_signs():
    rows = first_column(SHEET)
    assert [label for label, _, _ in rows][:3] == ["Cash and cash equivalents", "Accounts receivable", "Finished goods"]
    assert len(rows) == 11 and rows[0][1:] == ("2,853", 2853.0)
    assert rows[7] == ("Less: Accumulated depreciation", "16,135", -16135.0)  # parentheses mean a negative amount


def test_every_total_is_found_with_the_rows_it_adds_up():
    found = totals([value for _, _, value in first_column(SHEET)])
    assert found == {
        4: [2, 3],  # total inventories
        5: [0, 1, 4],  # total current assets, which takes the inventory total and not its parts
        8: [6, 7],  # property, plant and equipment net
        9: [5, 8],  # total assets
    }


def test_a_line_knows_whether_the_page_can_check_it():
    assert in_a_sum(probe("Accounts receivable", "5,020"), SHEET)
    assert in_a_sum(probe("Total current assets", "11,285"), SHEET)
    assert not in_a_sum(probe("Shares outstanding", "4,321"), SHEET)
    assert not in_a_sum(probe("Accounts receivable", "9,999"), SHEET)  # no such line


def test_a_moved_amount_leaves_every_total_as_it_was():
    target = probe("Accounts receivable", "5,020")
    edit = move_edit(target, SHEET)
    assert edit is not None and edit.old_text == "5,020" and edit.also[0][0] == "2,853"  # cash is the only other plain line under that total
    after = apply_edit(SHEET, edit)
    before_rows, after_rows = first_column(SHEET), first_column(after)
    moved = after_rows[1][2] - before_rows[1][2]
    assert moved > 0 and after_rows[0][2] == before_rows[0][2] - moved
    assert [r for i, r in enumerate(after_rows) if i > 1] == [r for i, r in enumerate(before_rows) if i > 1]  # nothing else changed
    assert totals([v for _, _, v in after_rows]) == totals([v for _, _, v in before_rows])  # and the page still adds up
    assert "4,911" in after and "3,053" in after and after.count("\n") == SHEET.count("\n")  # last year's column is untouched
    assert move_edit(target, SHEET) == edit  # the same edit every time


def test_a_total_never_gives_up_an_amount_even_when_its_sum_was_not_found():
    # The inventory lines are gone, so nothing on the page adds up to "Total inventories" any more.
    lines = SHEET.split("\n")
    start, stop = lines.index("Finished goods"), lines.index("Total inventories")
    page = "\n".join(lines[:start] + lines[stop:])
    assert 2 not in totals([value for _, _, value in first_column(page)])  # the code no longer sees that it is a total
    edit = move_edit(probe("Accounts receivable", "5,020"), page)
    assert edit is not None and edit.also[0][0] == "2,853"  # cash gives up the amount. The larger "Total inventories" is passed over
    assert move_edit(probe("Cash and cash equivalents", "2,853"), page).also[0][0] == "5,020"


def test_no_move_where_the_page_shows_no_total_or_the_line_is_one():
    assert move_edit(probe("Shares outstanding", "4,321"), SHEET) is None  # no total depends on it
    assert move_edit(probe("Total inventories", "3,412"), SHEET) is None  # a total is not moved
    assert move_edit(probe("Less: Accumulated depreciation", "16,135"), SHEET) is None  # a negative amount is left alone


@pytest.mark.skipif(not (default_root() / "data" / "financebench_open_source.jsonl").exists(), reason="benchmark data not downloaded")
def test_real_pages_give_moved_cases_whose_pages_still_add_up():
    from footnote.rag.stale import build_cases

    questions = load_questions()
    moved = build_cases(questions, "moved")
    assert len(moved) >= 30 and {c.group for c in moved} == {"moved"}
    for case in moved:
        before, after = first_column(case.before), first_column(case.after)
        assert totals([v for _, _, v in before]) == totals([v for _, _, v in after])
        changed = [(b, a) for b, a in zip(before, after) if a != b]
        assert len(changed) == 2  # the line asked about, and the neighbour that gave up the amount
        for (label, _, was), (_, _, now) in changed:
            assert not label.lower().startswith(("total", "net ", "subtotal", "gross ")) and now >= 0.7 * was
    single = build_cases(questions, "single")
    groups = {g: sum(c.group == g for c in single) for g in ("sum", "free", "benchmark")}
    assert sum(groups.values()) == 176 and groups["benchmark"] == 7 and groups["sum"] > 60 and groups["free"] > 60
