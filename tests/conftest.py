import json
import re

import pytest

from footnote.data.financebench import Evidence, Question
from footnote.ledger import Ledger
from footnote.llm import FakeLLM, LLMRequest


def make_question(i: int, split: str = "dev") -> Question:
    value = 1000 + 37 * i
    page = f"Consolidated Statement of Cash Flows\n\nPurchases of property, plant and equipment ({value:,}) (1,373)\n\nNet cash used"
    return Question(
        id=f"q{i:02d}",
        company=f"Company {i}",
        doc_name=f"COMPANY{i}_2020_10K",
        question_type="metrics-generated",
        question_reasoning="Information extraction",
        question=f"What is the FY2020 capital expenditure amount (in USD millions) for Company {i}?",
        answer=f"${value}.00",
        justification=None,
        evidence=(Evidence(doc_name=f"COMPANY{i}_2020_10K", page=40 + i, text=page, full_page_text=page),),
        doc_type="10k",
        doc_period="2020",
        split=split,
    )


@pytest.fixture
def questions():
    return [make_question(i, "dev" if i % 3 == 0 else "test") for i in range(6)]


def scripted_llm(questions, answer_style: str = "right", **kw) -> FakeLLM:
    """A stand-in model. It answers from a lookup table, and grades by comparing with the reference."""
    gold = {q.question: q.answer for q in questions}

    def respond(req: LLMRequest) -> str:
        body = req.messages[-1]["content"]
        if req.json_schema is not None:  # the judge
            reference = re.search(r"<reference_answer>\n(.*?)\n</reference_answer>", body, re.S).group(1)
            candidate = re.search(r"<candidate_answer>\n(.*?)\n</candidate_answer>", body, re.S).group(1)
            if "cannot" in candidate:
                label = "refused"
            else:
                label = "correct" if reference in candidate else "incorrect"
            return json.dumps({"reason": "scripted", "label": label})
        question = next(q for q in gold if q in body)
        if answer_style == "right":
            return f"The cash flow statement shows it.\nFinal answer: {gold[question]}"
        if answer_style == "refuse":
            return "I cannot answer without the filing."
        if answer_style == "empty":
            return ""
        return "Final answer: $1.00"

    kw.setdefault("ledger", Ledger(":memory:"))
    return FakeLLM(responder=respond, **kw)
