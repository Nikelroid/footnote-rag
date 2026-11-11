"""The amendment test through the real pipeline, with a stand-in for search. No PDF and no model is needed."""

import dataclasses

import pytest
from langchain_core.documents import Document

from conftest import make_question
from footnote.rag.amended_index import SUFFIX, VIEWS, ThroughIndex, _question, amend_filing, systems
from footnote.rag.graph import VERSION_RULE, answer_request
from footnote.rag.retrieve import Hit
from footnote.rag.stale import build_cases, changes
from test_sums import SHEET

# The same balance sheet as a PDF reader would give it: one line per row.
PDF_PAGE = "\n".join(
    [
        "Consolidated Balance Sheet 2020 2019",
        "Cash and cash equivalents $ 2,853 $ 3,053",
        "Accounts receivable 5,020 4,911",
        "Finished goods 2,120 1,915",
        "Work in process 1,292 1,218",
        "Total inventories 3,412 3,133",
        "Total current assets 11,285 11,097",
    ]
)


@pytest.fixture
def cases():
    q = make_question(0)
    q = dataclasses.replace(q, evidence=(dataclasses.replace(q.evidence[0], full_page_text=SHEET),))
    return build_cases([q], "moved")


def filing(text=PDF_PAGE):
    meta = {"source": "COMPANY0_2020_10K", "page": 40, "company": "Company 0", "doc_type": "10k", "period": "2020"}
    return [Document(page_content="Cover page", metadata={**meta, "page": 0}), Document(page_content=text, metadata=meta)]


def test_the_amended_filing_gets_the_name_of_an_amendment_and_differs_only_in_the_moved_amounts(cases):
    case = cases[0]
    amended, applied = amend_filing(filing(), [case])
    assert applied == [case] and {p.metadata["source"] for p in amended} == {"COMPANY0_2020_10K" + SUFFIX}
    assert amended[0].page_content == "Cover page"  # every other page is carried over as it is
    before, after = PDF_PAGE.split("\n"), amended[1].page_content.split("\n")
    assert sum(a != b for a, b in zip(before, after)) == 2 and len(before) == len(after)
    for _, old, new in changes(case):
        assert old not in amended[1].page_content and new in amended[1].page_content
    assert "3,053" in amended[1].page_content and "4,911" in amended[1].page_content  # last year's column is untouched


def test_a_case_is_left_out_when_its_amount_is_not_printed_exactly_once_in_the_pdf_text(cases):
    case = cases[0]
    twice = PDF_PAGE + f"\nSee also {case.old} above"
    amended, applied = amend_filing(filing(twice), [case])
    assert applied == [] and amended[1].page_content == twice  # nothing was changed


class Search:
    """A stand-in for the retriever: one page from every filing it is allowed to see."""

    def __init__(self):
        names = ("COMPANY0_2020_10K", "COMPANY0_2020_10K" + SUFFIX, "COMPANY0_2020_8K")
        self.filings = {name: {"company": "Company 0", "doc_period": "2020"} for name in names}
        self.chunks = []

    def route(self, question):
        return sorted(self.filings)

    def search(self, question, mode, sources):
        return [Hit(name, 40, f"page of {name}", 1.0) for name in sources]

    def pinned(self, sources):
        return []

    def pages(self, hits, n, first=None):
        return hits[:n]


def test_each_view_lets_the_search_see_the_filings_that_update_would_leave(cases):
    case = cases[0]
    case = dataclasses.replace(case, question=dataclasses.replace(case.question, question=_question(case)))
    assert "COMPANY0_2020_10K" not in case.question.question and "Company 0's annual report for fiscal 2020" in case.question.question
    seen = {s.view: [p.doc for p in s.find(case)] for s in systems(Search(), [case])}
    original, amended, other = "COMPANY0_2020_10K", "COMPANY0_2020_10K" + SUFFIX, "COMPANY0_2020_8K"
    assert seen["before"] == [original, other]  # the amended filing has not arrived yet
    assert seen["synced"] == [amended, other]  # it replaced the old one, and other filings are untouched
    assert seen["appended"] == seen["fixed"] == [original, amended, other]


def test_only_the_fixed_view_dates_the_pages_and_states_the_rule(cases):
    case = cases[0]
    by_view = {s.view: s for s in systems(Search(), [case])}
    appended = answer_request("claude-haiku-5-5", "q", by_view["appended"].find(case))
    assert "filed=" not in appended.messages[0]["content"] and not appended.system.endswith(VERSION_RULE)
    pages = by_view["fixed"].find(case)
    first_filed, amended = case.filed
    assert [p.filed for p in pages] == [first_filed, amended, first_filed] and first_filed < amended
    fixed = answer_request("claude-haiku-5-5", "q", pages)
    assert f'filed="{amended}"' in fixed.messages[0]["content"] and fixed.system.endswith(VERSION_RULE)
    assert len({s.name for s in by_view.values()}) == len(VIEWS) and by_view["fixed"].describe()["request"] != by_view["appended"].describe()["request"]
    with pytest.raises(ValueError):
        ThroughIndex(Search(), {}, "merged")
