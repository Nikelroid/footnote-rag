"""Step 3. The agent, as a LangGraph graph.

Simple RAG is one straight line: route, search once, answer. The agent adds one
decision. After a search it checks whether the pages are enough. If they are
not, it writes a better query and searches again, at most once more.

    START -> route -> search -> check -> answer -> END
                         ^        |
                         +--------+   "not enough, try this query instead"

After the second search it answers directly. A check there could change nothing.

The answer step must cite the pages it used, and must say so when the pages do
not contain the answer. A wrong number that looks right is worse than
"not found".

Every model call goes through `complete`, the project's one door to the API,
so each call is counted against the budget and recorded with its cost.
"""

from __future__ import annotations

import json
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from ..llm import LLMRequest, LLMResponse
from ..systems.base import ANSWER_MAX_TOKENS
from .retrieve import Hit, Retriever

MAX_SEARCHES = 2
PAGES = 8  # whole pages handed to the model
MODE = "hybrid+rerank"
NOT_FOUND = "NOT FOUND"

ANSWER_SYSTEM = (
    "You answer questions about the financial filings of public companies. Use only the pages you are given. "
    "Each page is marked with its document and page number. Work out the answer and show the numbers behind any calculation, "
    "in a few sentences. List the pages you used. If the pages do not contain what is needed, set found to false and do not guess."
)
# Added to the instructions only when the pages carry filing dates. Without it the model reads a
# date as a detail and tends to trust the earlier filing (measured in `stale.py`).
VERSION_RULE = " Each page also shows the date it was filed. When two versions of a page disagree, use the one that was filed later."
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "found": {"type": "boolean"},
        "final_answer": {"type": "string"},
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"document": {"type": "string"}, "page": {"type": "integer"}},
                "required": ["document", "page"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reasoning", "found", "final_answer", "pages"],
    "additionalProperties": False,
}

CHECK_SYSTEM = (
    "You check search results for a question about a company's financial filings. Decide whether the pages contain "
    "everything needed to answer the question. If they do not, write one better search query. Name the company and the year, "
    "and use the words a filing would use for the statement or line item, such as 'purchases of property, plant and equipment' "
    "for capital expenditure."
)
CHECK_SCHEMA = {
    "type": "object",
    "properties": {"enough": {"type": "boolean"}, "better_query": {"type": "string"}},
    "required": ["enough", "better_query"],
    "additionalProperties": False,
}

Complete = Callable[[LLMRequest, str], LLMResponse]


class State(TypedDict, total=False):
    question: str
    sources: list[str] | None  # the filings the question was routed to. None means search everything.
    query: str  # what is searched for. It starts as the question itself.
    searches: int
    chunks: list[Hit]  # the best chunks found so far
    pages: list[Hit]  # the whole pages the model reads
    enough: bool
    answer: str
    found: bool
    cited: list[dict]


def show(pages: list[Hit]) -> str:
    def date(p: Hit) -> str:
        return f' filed="{p.filed}"' if p.filed else ""

    return "\n\n".join(f'<page document="{p.doc}" number="{p.page}"{date(p)}>\n{p.text}\n</page>' for p in pages)


def _prompt(question: str, pages: list[Hit]) -> list[dict]:
    # Long material goes first and the question last.
    return [{"role": "user", "content": f"<pages>\n{show(pages)}\n</pages>\n\nQuestion: {question}"}]


def answer_request(model: str, question: str, pages: list[Hit]) -> LLMRequest:
    system = ANSWER_SYSTEM + (VERSION_RULE if any(p.filed for p in pages) else "")
    return LLMRequest(model=model, system=system, messages=_prompt(question, pages), max_tokens=ANSWER_MAX_TOKENS, json_schema=ANSWER_SCHEMA)


def check_request(model: str, question: str, pages: list[Hit]) -> LLMRequest:
    return LLMRequest(
        model=model, system=CHECK_SYSTEM, messages=_prompt(question, pages), max_tokens=1024, effort="low", json_schema=CHECK_SCHEMA
    )


def read_answer(text: str, pages: list[Hit]) -> tuple[str, bool, list[dict]]:
    """Turn the model's JSON into the answer text, whether it found the answer, and the pages it cited."""
    data = json.loads(text)
    found = bool(data["found"])
    final = data["final_answer"].strip() if found else NOT_FOUND
    given = {(p.doc, p.page) for p in pages}
    # A citation counts only if that page was really handed to the model. A made-up page is dropped.
    cited = [{"doc": c["document"], "page": c["page"]} for c in data["pages"] if (c["document"], c["page"]) in given]
    return f"{data['reasoning'].strip()}\nFinal answer: {final}", found, cited


def read_reply(reply: LLMResponse, pages: list[Hit]) -> tuple[str, bool, list[dict]]:
    """`read_answer` for a reply that may have been stopped early.

    A reply that was cut off, or that the API stopped as a refusal, is not complete JSON. Its text
    is kept as it came, so the run records what happened and does not fail on it.
    """
    if reply.stop_reason != "end_turn":
        return reply.text, False, []
    return read_answer(reply.text, pages)


def build_graph(retriever: Retriever, complete: Complete, model: str, mode: str = MODE, n_pages: int = PAGES):
    def route(state: State) -> State:
        return {"sources": retriever.route(state["question"]), "query": state["question"], "searches": 0}

    def search(state: State) -> State:
        found = retriever.search(state["query"], mode, state["sources"])
        if state.get("chunks"):
            # Second search: the better query's results go first, and the earlier ones fill what is left.
            new = {c.chunk_id for c in found}
            found = found + [c for c in state["chunks"] if c.chunk_id not in new]
        pages = retriever.pages(found, n_pages, first=retriever.pinned(state["sources"]))
        return {"chunks": found, "pages": pages, "searches": state["searches"] + 1}

    def check(state: State) -> State:
        reply = json.loads(complete(check_request(model, state["question"], state["pages"]), "check").text)
        return {"enough": bool(reply["enough"]), "query": reply["better_query"].strip() or state["query"]}

    def answer(state: State) -> State:
        reply = complete(answer_request(model, state["question"], state["pages"]), "answer")
        composed, found, cited = read_reply(reply, state["pages"])
        return {"answer": composed, "found": found, "cited": cited}

    def after_search(state: State) -> str:
        return "answer" if state["searches"] >= MAX_SEARCHES else "check"

    def after_check(state: State) -> str:
        return "answer" if state["enough"] else "search"

    graph = StateGraph(State)
    graph.add_node("route", route)
    graph.add_node("search", search)
    graph.add_node("check", check)
    graph.add_node("answer", answer)
    graph.add_edge(START, "route")
    graph.add_edge("route", "search")
    graph.add_conditional_edges("search", after_search, {"check": "check", "answer": "answer"})
    graph.add_conditional_edges("check", after_check, {"search": "search", "answer": "answer"})
    graph.add_edge("answer", END)
    return graph.compile()
