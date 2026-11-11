"""The two RAG systems the evaluation compares: one straight line, and the agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import config
from ..data.financebench import Question
from ..systems.base import SAMPLE, RunContext, SystemOutput, fingerprint
from . import ingest, retrieve, route
from .graph import MODE, PAGES, answer_request, build_graph, check_request, read_reply
from .retrieve import Hit, Retriever

_SAMPLE_PAGES = [Hit(doc="SAMPLE_2020_10K", page=1, text="Capital expenditure 100", score=1.0)]


def _settings(system) -> dict:
    """Everything that shapes the answers. A run made with other settings must never be mixed with this one."""
    return {
        "name": system.name,
        "model": system.model,
        "mode": system.mode,
        "pages": system.n_pages,
        "embed_model": ingest.EMBED_MODEL,
        "rerank_model": retrieve.RERANK_MODEL,
        "chunk_size": ingest.CHUNK_SIZE,
        "chunk_overlap": ingest.CHUNK_OVERLAP,
        "chunks": len(system.retriever.chunks),
        "candidates": system.retriever.candidates,
        "routing_rules": route.RULES_VERSION,
        "answer_request": fingerprint(answer_request(system.model, SAMPLE.question, _SAMPLE_PAGES)),
    }


@dataclass
class SimpleRAG:
    """Route, search once, then answer from the pages that were found."""

    retriever: Retriever = field(repr=False)
    model: str = config.ANSWER_MODEL
    mode: str = MODE
    n_pages: int = PAGES
    name: str = "rag"
    retrieves: bool = True

    def describe(self) -> dict:
        return _settings(self)

    def plan(self, question: Question) -> list:
        """The requests this question will send, for a cost estimate. Search itself is free."""
        return [answer_request(self.model, question.question, self.retriever.find(question.question, self.mode, self.n_pages))]

    def answer(self, question: Question, ctx: RunContext) -> SystemOutput:
        pages = self.retriever.find(question.question, self.mode, self.n_pages)
        reply = ctx.complete(answer_request(self.model, question.question, pages), "answer")
        text, _, cited = read_reply(reply, pages)
        return SystemOutput(answer=text, retrieved=[p.to_record() for p in pages], citations=cited)


@dataclass
class AgenticRAG:
    """The LangGraph agent: route, search, check whether it is enough, search again if not, then answer."""

    retriever: Retriever = field(repr=False)
    model: str = config.ANSWER_MODEL
    mode: str = MODE
    n_pages: int = PAGES
    name: str = "agent"
    retrieves: bool = True

    def describe(self) -> dict:
        return {**_settings(self), "check_request": fingerprint(check_request(self.model, SAMPLE.question, _SAMPLE_PAGES))}

    def plan(self, question: Question) -> list:
        """At most one check and one answer. The second search adds no further call."""
        pages = self.retriever.find(question.question, self.mode, self.n_pages)
        return [check_request(self.model, question.question, pages), answer_request(self.model, question.question, pages)]

    def answer(self, question: Question, ctx: RunContext) -> SystemOutput:
        graph = build_graph(self.retriever, ctx.complete, self.model, mode=self.mode, n_pages=self.n_pages)
        state = graph.invoke({"question": question.question})
        return SystemOutput(answer=state["answer"], retrieved=[p.to_record() for p in state["pages"]], citations=state["cited"])
