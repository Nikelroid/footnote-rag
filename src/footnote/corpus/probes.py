"""Reading probes with an exact answer, built from table lines without any model.

The changing-filings test needs many questions whose answer is a number printed
on a known page. The benchmark has only a handful of those. The evidence pages,
though, are mostly financial statements, and a statement is a list of labelled
lines such as "Depreciation and amortization 1,488 1,544 1,474". Each such line
gives a probe. The question names the line, and the answer is its first amount.

A probe is a plain reading check. It is not a benchmark question, and its
results are always reported apart from the benchmark's.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass

from ..data.financebench import Question
from .edits import CounterfactualEdit, _factor, _format, _on_page

# In the benchmark's page text every table cell sits on its own line.
_AMOUNT = re.compile(r"^\(?\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\)?%?$")
_FILLER = {"$", "%", "(", ")"}
_EMPTY_CELL = {"-", chr(0x2014), chr(0x2013)}  # a dash printed where a line has no amount
# Lines that are sums of other lines. If one were edited the statement would no longer add
# up, and a model that adds the lines itself would give the old value from the new page.
_DERIVED = ("total", "net ", "subtotal", "gross ")


@dataclass(frozen=True)
class Probe:
    probe_id: str
    doc_name: str
    page: int
    label: str
    question: str
    answer: str  # the amount as printed, for example "1,488"


def table_lines(page_text: str) -> list[tuple[str, str]]:
    """(label, first amount) for each labelled line of a statement."""
    out: list[tuple[str, str]] = []
    label: str | None = None
    for raw in page_text.split("\n"):
        line = raw.strip()
        if not line or line in _FILLER:
            continue
        if line in _EMPTY_CELL:
            label = None  # the first column is empty, so the next amount belongs to another column
            continue
        amount = _AMOUNT.match(line)
        if amount:
            if label is not None:
                out.append((label, amount.group(1)))
                label = None
            continue
        # A label starts with a capital, has no digits and is at least two words.
        # A lower-case line is the second half of a wrapped label and is dropped.
        plain = len(line.split()) >= 2 and line[0].isupper() and not any(ch.isdigit() for ch in line)
        label = line if plain and 12 <= len(line) <= 80 else None
    return out


def clean_lines(page_text: str) -> list[tuple[str, str]]:
    """Lines that make an unambiguous probe: the label and the amount each appear once on the page."""
    pairs = table_lines(page_text)
    labels = Counter(label for label, _ in pairs)
    out = []
    for label, amount in pairs:
        whole_thousands = "." not in amount and len(amount.replace(",", "")) >= 4
        once = page_text.count(label) == 1 and len(_on_page(amount).findall(page_text)) == 1
        if labels[label] == 1 and whole_thousands and once and not label.lower().startswith(_DERIVED):
            out.append((label, amount))
    return out


def build_probes(questions: list[Question], per_page: int = 2) -> list[Probe]:
    """Up to `per_page` probes from every evidence page, chosen by a fixed hash so the set never changes."""
    pages: dict[tuple[str, int], tuple[str, str]] = {}
    for q in questions:
        for e in q.evidence:
            pages[(e.doc_name, e.page)] = (q.company, e.full_page_text)
    probes = []
    for (doc_name, page), (company, text) in sorted(pages.items()):
        # The same line is often printed again elsewhere in a filing. Only the pages at hand can be
        # checked here. The check is repeated on the whole filing once it is parsed.
        elsewhere = [t for (d, p), (_, t) in pages.items() if d == doc_name and p != page]
        usable = [
            (label, amount)
            for label, amount in clean_lines(text)
            if not any(label in other or _on_page(amount).search(other) for other in elsewhere)
        ]
        ranked = sorted(usable, key=lambda pair: hashlib.sha256(f"{doc_name}|{page}|{pair[0]}".encode()).hexdigest())
        for label, amount in ranked[:per_page]:
            digest = hashlib.sha256(f"{doc_name}|{page}|{label}".encode()).hexdigest()[:10]
            probes.append(
                Probe(
                    probe_id=f"probe_{digest}",
                    doc_name=doc_name,
                    page=page,
                    label=label,
                    question=(
                        f"In the filing {doc_name} ({company}), what is the first amount on the line "
                        f"'{label}'? Answer with the number as printed."
                    ),
                    answer=amount,
                )
            )
    return probes


def probe_edit(probe: Probe, page_text: str) -> CounterfactualEdit | None:
    """A second version of the probe's page in which its amount is different, with the new answer known."""
    value = float(probe.answer.replace(",", ""))
    for attempt in range(5):
        new_text = _format(round(value * _factor(f"{probe.probe_id}:{attempt}")), 0, "," in probe.answer)
        if new_text != probe.answer and not _on_page(new_text).search(page_text):
            return CounterfactualEdit(
                question_id=probe.probe_id,
                doc_name=probe.doc_name,
                page=probe.page,
                old_text=probe.answer,
                new_text=new_text,
                old_answer=probe.answer,
                new_answer=new_text,
                occurrences=1,
            )
    return None
