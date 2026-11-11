"""The floor and the ceiling of the ladder.

The floor answers from the question alone, so it shows how much the model already
knows about these filings. The ceiling is handed the evidence pages, so it shows
what perfect retrieval would give. Every retrieval system lands between them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import config
from ..data.financebench import Question
from ..llm import LLMRequest
from .base import ANSWER_FORMAT, ANSWER_MAX_TOKENS, SAMPLE, RunContext, SystemOutput, fingerprint

ROLE = "You answer questions about the financial filings of public companies."


@dataclass
class ClosedBook:
    """Floor. No documents, only the question."""

    model: str = config.ANSWER_MODEL
    name: str = "floor"
    retrieves: bool = False

    @property
    def system_prompt(self) -> str:
        return f"{ROLE} {ANSWER_FORMAT}"

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model, "request": fingerprint(self.request(SAMPLE))}

    def request(self, question: Question) -> LLMRequest:
        return LLMRequest(
            model=self.model,
            system=self.system_prompt,
            messages=[{"role": "user", "content": question.question}],
            max_tokens=ANSWER_MAX_TOKENS,
        )

    def plan(self, question: Question) -> list:
        return [self.request(question)]

    def answer(self, question: Question, ctx: RunContext) -> SystemOutput:
        return SystemOutput(answer=ctx.complete(self.request(question)).text)


@dataclass
class Oracle:
    """Ceiling. The evidence pages are placed in the prompt, as in the oracle setting of the benchmark paper."""

    model: str = config.ANSWER_MODEL
    name: str = "ceiling"
    retrieves: bool = True

    @property
    def system_prompt(self) -> str:
        return f"{ROLE} Use only the filing pages you are given. {ANSWER_FORMAT}"

    def describe(self) -> dict:
        return {"name": self.name, "model": self.model, "request": fingerprint(self.request(SAMPLE))}

    def request(self, question: Question) -> LLMRequest:
        pages = "\n\n".join(
            f'<page document="{e.doc_name}" number="{e.page}">\n{e.full_page_text}\n</page>' for e in question.evidence
        )
        # Long material goes first and the question last.
        content = f"<filing_pages>\n{pages}\n</filing_pages>\n\nQuestion: {question.question}"
        return LLMRequest(
            model=self.model,
            system=self.system_prompt,
            messages=[{"role": "user", "content": content}],
            max_tokens=ANSWER_MAX_TOKENS,
        )

    def plan(self, question: Question) -> list:
        return [self.request(question)]

    def answer(self, question: Question, ctx: RunContext) -> SystemOutput:
        hits = [{"doc": e.doc_name, "page": e.page, "chunk_id": None, "score": None} for e in question.evidence]
        return SystemOutput(answer=ctx.complete(self.request(question)).text, retrieved=hits, citations=hits)


SYSTEMS = {"floor": ClosedBook, "ceiling": Oracle}
