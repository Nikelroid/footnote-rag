"""The whole system behind two calls: `search` finds pages, `ask` answers a question.

The API, the demo page and the MCP server all use this one class, so they
cannot drift apart from what the evaluation measures.
"""

from __future__ import annotations

import time

from .. import config
from ..eval.numeric import final_answer_text
from ..llm import LLM
from ..systems.base import RunContext
from .graph import MODE, PAGES, answer_request, build_graph, read_reply
from .ingest import Index
from .retrieve import Hit, Retriever


class Assistant:
    def __init__(self, retriever: Retriever | None = None, llm: LLM | None = None, model: str = config.ANSWER_MODEL, line: str = "development"):
        self.retriever = retriever or Retriever(Index())
        self._llm = llm
        self.model = model
        self.line = line

    @property
    def llm(self) -> LLM:
        if self._llm is None:  # created on first use, so searching works without an API key
            from ..ledger import Ledger
            from ..llm import AnthropicLLM

            self._llm = AnthropicLLM(Ledger())
        return self._llm

    @staticmethod
    def _page(hit: Hit) -> dict:
        return {"doc": hit.doc, "page": hit.page, "score": round(hit.score, 4), "text": hit.text}

    def search(self, query: str, pages: int = 5, mode: str = MODE) -> list[dict]:
        """The best pages for a query. No model is called, so this is free."""
        return [self._page(h) for h in self.retriever.find(query, mode, pages)]

    def ask(self, question: str, agent: bool = True, pages: int = PAGES) -> dict:
        """Answer a question. With `agent`, a weak first search is followed by a second one."""
        start = time.monotonic()
        ctx = RunContext(llm=self.llm, line=self.line, run_id="interactive", deadline=start + 300)
        if agent:
            state = build_graph(self.retriever, ctx.complete, self.model, n_pages=pages).invoke({"question": question})
            answer, found, cited, used, searches = state["answer"], state["found"], state["cited"], state["pages"], state["searches"]
        else:
            used = self.retriever.find(question, MODE, pages)
            answer, found, cited = read_reply(ctx.complete(answer_request(self.model, question, used), "answer"), used)
            searches = 1
        return {
            "question": question,
            "answer": answer,
            "final_answer": final_answer_text(answer),
            "found": found,
            "citations": cited,
            "pages": [self._page(p) for p in used],
            "searches": searches,
            "model_calls": len(ctx.calls),
            "cost_usd": round(sum(c.cost_usd for c in ctx.calls), 6),
            "seconds": round(time.monotonic() - start, 2),
        }
