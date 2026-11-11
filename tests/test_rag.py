"""The RAG pipeline with a stand-in search and a stand-in model. No index, no network."""

import json
import time

import pytest
from langchain_core.documents import Document

from conftest import make_question
from footnote.eval.judge import Judge
from footnote.eval.report import load_run, summarize
from footnote.eval.runner import answer_run, grade_run
from footnote.ledger import Ledger
from footnote.llm import FakeLLM
from footnote.rag import ingest
from footnote.rag.graph import MAX_SEARCHES, NOT_FOUND, build_graph, read_answer
from footnote.rag.retrieve import Hit, fuse, tokens
from footnote.rag.systems import AgenticRAG, SimpleRAG
from footnote.systems.base import RunContext

QUESTION = "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
BETTER = "3M 2018 purchases of property, plant and equipment"


class FakeRetriever:
    """Finds the cash flow page only when the query uses the filing's own words, like real search."""

    candidates = 30
    chunks = ["a", "b", "c"]

    def __init__(self):
        self.queries = []
        self.searched_in = []

    def route(self, question):
        return ["3M_2018_10K"] if "3m" in question.lower() else None

    def pinned(self, sources):
        return []

    def search(self, query, mode="hybrid+rerank", sources=None):
        self.queries.append(query)
        self.searched_in.append(sources)
        other = Hit("3M_2018_10K", 87, "Capital spending is discussed here.", 2.0)
        if "purchases of property" in query.lower():
            return [Hit("3M_2018_10K", 59, "Purchases of property, plant and equipment (1,577)", 9.0), other]
        return [other]

    def pages(self, hits, n, first=None):
        out, seen = list(first or []), set()
        for h in hits:
            if (h.doc, h.page) not in seen:
                seen.add((h.doc, h.page))
                out.append(Hit(h.doc, h.page, f"Whole page {h.page}. {h.text}", h.score))
        return out[:n]

    def find(self, question, mode="hybrid+rerank", n_pages=8, routed=True, pin=True):
        sources = self.route(question) if routed else None
        return self.pages(self.search(question, mode, sources), n_pages, first=self.pinned(sources) if pin else None)


def model(always_unsure=False):
    def respond(req):
        prompt = req.messages[-1]["content"]
        sees_it = "1,577" in prompt
        if "enough" in req.json_schema["properties"]:
            return json.dumps({"enough": sees_it and not always_unsure, "better_query": "" if sees_it else BETTER})
        if sees_it:
            pages = [{"document": "3M_2018_10K", "page": 59}, {"document": "3M_2018_10K", "page": 999}]
            return json.dumps({"reasoning": "The cash flow statement shows it.", "found": True, "final_answer": "$1,577 million", "pages": pages})
        return json.dumps({"reasoning": "The pages do not show the figure.", "found": False, "final_answer": "", "pages": []})

    return FakeLLM(responder=respond, ledger=Ledger(":memory:"))


def context(llm):
    return RunContext(llm=llm, line="development", run_id="test", deadline=time.monotonic() + 60)


def question(text=QUESTION):
    import dataclasses

    return dataclasses.replace(make_question(0), question=text, answer="$1577.00")


def test_simple_rag_says_not_found_when_search_misses_the_page():
    llm, retriever = model(), FakeRetriever()
    out = SimpleRAG(retriever).answer(question(), context(llm))
    assert out.answer.endswith(f"Final answer: {NOT_FOUND}") and out.citations == [] and len(llm.calls) == 1
    assert [(r["doc"], r["page"]) for r in out.retrieved] == [("3M_2018_10K", 87)]


def test_agent_searches_again_with_better_words_and_cites_the_page():
    llm, retriever = model(), FakeRetriever()
    out = AgenticRAG(retriever).answer(question(), context(llm))
    assert retriever.queries == [QUESTION, BETTER]
    assert retriever.searched_in == [["3M_2018_10K"], ["3M_2018_10K"]]  # the second search keeps the first one's routing
    assert out.answer.endswith("Final answer: $1,577 million")
    assert out.citations == [{"doc": "3M_2018_10K", "page": 59}]  # the page that was never handed over is dropped
    assert ("3M_2018_10K", 59) in [(r["doc"], r["page"]) for r in out.retrieved]
    assert len(llm.calls) == 2  # one check, one answer. No second check after the last search.


def test_agent_answers_at_once_when_the_first_search_is_enough():
    llm, retriever = model(), FakeRetriever()
    out = AgenticRAG(retriever).answer(question(BETTER), context(llm))
    assert len(retriever.queries) == 1 and len(llm.calls) == 2 and "1,577" in out.answer


def test_agent_never_searches_more_than_allowed():
    llm, retriever = model(always_unsure=True), FakeRetriever()
    AgenticRAG(retriever).answer(question(BETTER), context(llm))
    assert len(retriever.queries) == MAX_SEARCHES


def test_graph_state_is_plain_and_inspectable():
    llm, retriever = model(), FakeRetriever()
    ctx = context(llm)
    state = build_graph(retriever, ctx.complete, "claude-haiku-5-5").invoke({"question": QUESTION})
    assert state["searches"] == 2 and state["found"] is True and state["query"] == BETTER and state["sources"] == ["3M_2018_10K"]
    assert [p.page for p in state["pages"]] == [59, 87] and state["pages"][0].text.startswith("Whole page 59.")
    assert [c.page for c in state["chunks"]] == [59, 87]  # the second search's results first, each chunk once
    assert [c.to_record()["stop_reason"] for c in ctx.calls] == ["end_turn", "end_turn"] and ctx.calls[0].cost_usd > 0


def test_read_answer():
    pages = [Hit("D", 3, "text", 1.0)]
    reply = {"reasoning": " Because. ", "found": True, "final_answer": " 42 ", "pages": [{"document": "D", "page": 3}, {"document": "E", "page": 1}]}
    assert read_answer(json.dumps(reply), pages) == ("Because.\nFinal answer: 42", True, [{"doc": "D", "page": 3}])
    reply["found"] = False
    assert read_answer(json.dumps(reply), pages)[0] == f"Because.\nFinal answer: {NOT_FOUND}"


def test_reciprocal_rank_fusion_rewards_agreement():
    assert fuse([["a", "b", "c"], ["c", "a", "d"]]) == ["a", "c", "b", "d"]
    assert fuse([["x"], []]) == ["x"]
    assert tokens("Purchases of PP&E: (1,577)") == ["purchases", "of", "pp", "e", "1", "577"]


def test_a_changed_setting_starts_a_new_run():
    retriever = FakeRetriever()
    base = SimpleRAG(retriever).describe()
    assert base != SimpleRAG(retriever, mode="bm25").describe() and base != SimpleRAG(retriever, n_pages=3).describe()
    assert base != AgenticRAG(retriever).describe() and base["embed_model"] == ingest.EMBED_MODEL


def test_routing_reads_the_company_and_the_year():
    from footnote.rag.route import route, years

    table = {
        "3M_2018_10K": {"company": "3M", "doc_period": 2018},
        "3M_2022_10K": {"company": "3M", "doc_period": 2022},
        "JPMORGAN_2022_10K": {"company": "JPMorgan", "doc_period": 2022},
        "BESTBUY_2023_10K": {"company": "Best Buy", "doc_period": 2023},
    }
    assert route("What is the FY2018 capital expenditure for 3M?", table) == ["3M_2018_10K"]
    assert route("How did 3M's margins change?", table) == ["3M_2018_10K", "3M_2022_10K"]  # no year, so every 3M filing
    assert route("Are JPM's gross margins consistent in 2022?", table) == ["JPMORGAN_2022_10K"]  # a short name
    assert route("What is 3M's FY2019 revenue?", table) == ["3M_2018_10K", "3M_2022_10K"]  # no filing for that year
    assert route("Was there any drop in cash between FY 2023 and Q2 of FY2024?", table) is None  # no company named
    assert route("What is the best 23M estimate?", table) is None  # "3M" inside another token, "best" without "buy"
    assert years("Q2 of FY2024 against FY 2023 and FY22") == {"2022", "2023", "2024"}
    assert years("the biggest drop in Q22023 revenues") == {"2023"}  # a quarter written straight onto its year


def test_main_statements_are_the_three_pages_that_sit_together():
    from footnote.rag.statements import main_statements, statement_kind

    numbers = "\n".join(f"Line {i}\n{1000 + i:,}\n{2000 + i:,}" for i in range(20))
    page = lambda title: f"Table of Contents\n{title}\n(Millions)\n{numbers}"
    pages = {
        3: "Index\nConsolidated Statements of Income 55\nConsolidated Balance Sheets 57\nConsolidated Statements of Cash Flow s 59",
        45: page("Cash Flows discussion in MD&A"),
        46: page("Summary from the Consolidated Statement of Cash Flow s"),
        55: page("Consolidated Statement of Income"),
        57: page("Consolidated Balance Sheet"),
        59: page("Consolidated Statement of Cash Flow s"),
        103: page("Condensed Consolidated Statement of Income of the guarantor"),
    }
    assert statement_kind(pages[59]) == "cash flow" and statement_kind(pages[57]) == "balance sheet"
    assert statement_kind(pages[3]) is None  # a contents page names all three and holds no figures
    assert statement_kind(pages[45]) is None
    assert main_statements(pages) == [55, 57, 59]  # not 46 or 103, which carry a statement title but stand apart
    assert main_statements({1: "Risk factors and other prose."}) == []


def test_the_agent_runs_through_the_evaluation_harness(tmp_path):
    llm, retriever = model(), FakeRetriever()
    questions = [question()]
    run_dir = answer_run(AgenticRAG(retriever), questions, llm, reps=2, runs_dir=tmp_path, log=lambda s: None)
    judge_llm = FakeLLM(responder=lambda req: json.dumps({"reason": "Same value.", "label": "correct"}), ledger=Ledger(":memory:"))
    grade_run(run_dir, {q.id: q for q in questions}, Judge(judge_llm), log=lambda s: None)
    summary = summarize(load_run(run_dir))
    assert summary["correct"]["mean"] == 1.0 and summary["n_graded"] == 2
    assert summary["numeric_check"]["agree"] == 1.0  # "$1,577 million" matches the reference of $1577.00 in USD millions
    assert summary["retrieval"]["hit@5"] == 0.0  # the stand-in question's evidence page is not one the stand-in search returns


def test_chunks_stay_inside_their_page_and_say_where_they_are_from():
    meta = {"source": "CO_2020_10K", "company": "Co", "doc_type": "10k", "period": "2020"}
    pages = [Document(page_content=f"Page {n} text. " * 400, metadata={**meta, "page": n}) for n in (0, 7)]
    chunks = ingest.split(pages)
    assert {c.metadata["page"] for c in chunks} == {0, 7} and len(chunks) > 4
    for chunk in chunks:
        header, body = chunk.page_content.split("\n", 1)
        assert header == f"Co | 10k | 2020 | page {chunk.metadata['page'] + 1}"
        assert f"Page {chunk.metadata['page']} text." in body and len(body) <= ingest.CHUNK_SIZE


def test_page_store_replaces_a_filing_as_a_whole(tmp_path):
    store = ingest.PageStore(tmp_path / "pages.sqlite")
    doc = lambda n, text: Document(page_content=text, metadata={"page": n})
    store.replace("A", [doc(0, "first"), doc(1, "second")])
    store.replace("B", [doc(0, "other filing")])
    store.replace("A", [doc(0, "first, amended")])  # the amended filing has one page fewer
    assert store.get("A", 0) == "first, amended" and store.get("A", 1) is None and store.get("B", 0) == "other filing"
    assert store.count() == 2
