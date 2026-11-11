"""What every rung of the ladder has in common."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..data.financebench import Evidence, Question
from ..llm import LLM, LLMRequest, LLMResponse

# One answer format for every level, so that a change in score between levels
# comes from the level and not from how the answer is written.
ANSWER_FORMAT = (
    "Work out the answer, then state it. Keep the explanation to a few sentences and show the numbers "
    "behind any calculation. End with one line that starts with 'Final answer:' followed by the answer alone."
)

ANSWER_MAX_TOKENS = 4096


SAMPLE = Question(
    id="sample",
    company="Sample Co",
    doc_name="SAMPLE_2020_10K",
    question_type="metrics-generated",
    question_reasoning=None,
    question="What is the FY2020 capital expenditure amount (in USD millions) for Sample Co?",
    answer="$100.00",
    justification=None,
    evidence=(Evidence(doc_name="SAMPLE_2020_10K", page=1, text="Capital expenditure 100", full_page_text="Capital expenditure 100"),),
    doc_type="10k",
    doc_period="2020",
    split="dev",
)


def fingerprint(request: LLMRequest) -> str:
    """A hash of everything a request sends. Put it in describe(), so a changed prompt starts a new run."""
    return hashlib.sha256(json.dumps(request.api_kwargs(), sort_keys=True).encode()).hexdigest()[:12]


class CaseTimeout(RuntimeError):
    pass


@dataclass
class RunContext:
    """Handed to a system for one question. It carries the client, the budget line and a deadline."""

    llm: LLM
    line: str
    run_id: str
    deadline: float
    calls: list[LLMResponse] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)

    def complete(self, request: LLMRequest, purpose: str = "answer") -> LLMResponse:
        if time.monotonic() > self.deadline:
            raise CaseTimeout("the question passed its time limit")
        # The trace keeps every call whole, prompt and reply, so a surprising
        # score can be traced to what the model actually saw.
        call = len(self.calls)
        if request.system:
            self.trace.append({"call": call, "role": "system", "content": request.system})
        for message in request.messages:
            self.trace.append({"call": call, "role": message["role"], "content": message["content"]})
        response = self.llm.complete(request, line=self.line, run_id=self.run_id, purpose=purpose)
        self.calls.append(response)
        self.trace.append({"call": call, "role": "assistant", "content": response.text, "stop_reason": response.stop_reason})
        return response


@dataclass
class SystemOutput:
    answer: str
    retrieved: list[dict] = field(default_factory=list)  # ranked hits: doc, page, chunk_id, score
    citations: list[dict] = field(default_factory=list)  # pages the answer says it used


class System(Protocol):
    name: str
    retrieves: bool  # True when the system searches. An empty result then scores zero instead of being left out.

    def describe(self) -> dict:
        """Everything that defines this system. The run directory is named after a hash of it."""
        ...

    def answer(self, question: Question, ctx: RunContext) -> SystemOutput: ...
