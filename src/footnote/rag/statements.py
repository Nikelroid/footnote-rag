"""Find a filing's three main financial statements: income statement, balance sheet, cash flow statement.

Most questions about a company's numbers are answered from these three pages.
Search often misses them, because a question says "capital expenditure" and the
page says "purchases of property, plant and equipment". So the pages are found
once, by their titles, and handed to the model whenever a question is routed to
that filing.

A statement title also appears in the table of contents and in the notes. The
real statements sit next to each other, so the three pages that lie closest
together are taken.
"""

from __future__ import annotations

import itertools
import re

TITLES = {
    "income": ("statementsofincome", "statementofincome", "statementsofoperations", "statementofoperations", "statementsofearnings", "statementofearnings", "incomestatement"),
    "balance sheet": ("balancesheet", "statementsoffinancialposition", "statementoffinancialposition", "statementsoffinancialcondition", "statementofcondition"),
    "cash flow": ("statementsofcashflows", "statementofcashflows", "statementofcashflow", "cashflowsstatement"),
}


def statement_kind(page_text: str) -> str | None:
    """Which statement a page is, judged by the title at its top. None for any other page."""
    head = re.sub(r"[^a-z]", "", page_text[:500].lower())  # the PDF text has stray spaces, as in "Cash Flow s"
    found = [kind for kind, titles in TITLES.items() if any(t in head for t in titles)]
    many_numbers = len(re.findall(r"\d[\d,]{2,}", page_text)) >= 25
    return found[0] if len(found) == 1 and many_numbers else None


def main_statements(pages: dict[int, str]) -> list[int]:
    """Page numbers of the main statements in one filing. `pages` maps page number to text."""
    by_kind: dict[str, list[int]] = {}
    for number, text in sorted(pages.items()):
        kind = statement_kind(text)
        if kind:
            by_kind.setdefault(kind, []).append(number)
    if not by_kind:
        return []
    closest = min(itertools.product(*by_kind.values()), key=lambda combo: (max(combo) - min(combo), min(combo)))
    return sorted(set(closest))
