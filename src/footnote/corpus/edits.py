"""Controlled edits with a known answer, for testing what happens when a filing changes.

Take a question whose answer is a number printed on its evidence page. Write a
second version of that page in which the number is different. After the update,
the right answer is the new number, and it is known exactly because the edit
made it. The new number exists nowhere else, so a model cannot recall it from
training. An answer with the new number proves the system read the current page.
An answer with the old number is a stale answer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..data.financebench import Question
from ..eval.numeric import NUMBER, parse_gold


@dataclass(frozen=True)
class CounterfactualEdit:
    question_id: str
    doc_name: str
    page: int
    old_text: str  # the number as printed on the page, for example "1,577"
    new_text: str
    old_answer: str  # the reference answer before the edit
    new_answer: str  # the reference answer after it, in the same format
    occurrences: int  # how many times the number is replaced on the page
    also: tuple[tuple[str, str], ...] = ()  # other amounts changed with it, as (old, new), so the page's totals still add up


def _format(value: float, decimals: int, commas: bool) -> str:
    return f"{value:,.{decimals}f}" if commas else f"{value:.{decimals}f}"


def _on_page(form: str) -> re.Pattern:
    # The printed number must stand alone. "1,577" must not match inside "21,577" or "1,577.4".
    return re.compile(r"(?<![\d.,])" + re.escape(form) + r"(?![\d]|[.,]\d)")


def _surface_forms(raw: str) -> list[tuple[str, int, bool]]:
    """Ways the reference number may be printed in a filing: (text, decimals, uses commas)."""
    value = float(raw)
    written = len(raw.split(".")[1]) if "." in raw else 0
    meaningful = len(raw.split(".")[1].rstrip("0")) if "." in raw else 0
    forms = []
    for decimals in dict.fromkeys((meaningful, written)):
        for commas in (True, False):
            if commas and abs(value) < 1000:
                continue
            forms.append((_format(abs(value), decimals, commas), decimals, commas))
    return forms


def _factor(question_id: str) -> float:
    """A fixed change between 7% and 31%, different for each question."""
    digest = int(hashlib.sha256(question_id.encode()).hexdigest()[:8], 16)
    return 1.07 + (digest % 2400) / 10000


def make_edit(question: Question) -> CounterfactualEdit | None:
    """Return an edit for this question, or None when its answer is not a number printed on the evidence page."""
    if parse_gold(question.answer) is None:
        return None
    match = NUMBER.search(question.answer)
    raw = match.group("num").replace(",", "")
    value = abs(float(raw))
    if value == 0:
        return None
    for evidence in question.evidence:
        for form, decimals, commas in _surface_forms(raw):
            hits = _on_page(form).findall(evidence.full_page_text)
            if not hits:
                continue
            new_value = round(value * _factor(question.id), decimals)
            new_text = _format(new_value, decimals, commas)
            if new_text == form or _on_page(new_text).search(evidence.full_page_text):
                continue
            answer_decimals = len(raw.split(".")[1]) if "." in raw else 0
            new_number = _format(new_value, answer_decimals, "," in match.group("num"))
            new_answer = question.answer[: match.start("num")] + new_number + question.answer[match.end("num") :]
            return CounterfactualEdit(
                question_id=question.id,
                doc_name=evidence.doc_name,
                page=evidence.page,
                old_text=form,
                new_text=new_text,
                old_answer=question.answer,
                new_answer=new_answer,
                occurrences=len(hits),
            )
    return None


def apply_edit(page_text: str, edit: CounterfactualEdit) -> str:
    for old, new in ((edit.old_text, edit.new_text), *edit.also):
        page_text = _on_page(old).sub(new, page_text)
    return page_text
