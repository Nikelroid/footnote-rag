"""Work out which filings a question is about, before searching.

Most questions name a company and a year. Searching only that company's filings
for that year removes most of the wrong pages before any ranking happens. This
is metadata filtering, and it is often worth more than a better embedding model.

The lookup is a plain table of company names and the short names people use for
them. A bank would keep the same kind of table for its own document types,
business lines and dates. When a question names no company, nothing is filtered
and the whole index is searched.
"""

from __future__ import annotations

import re

# Short names that appear in questions. The full company name is always matched as well.
ALIASES = {
    "JPMorgan": ["jpm", "jp morgan"],
    "Johnson & Johnson": ["jnj", "j&j"],
    "American Express": ["amex"],
    "MGM Resorts": ["mgm"],
    "CVS Health": ["cvs"],
    "AES Corporation": ["aes"],
    "Coca-Cola": ["coca cola", "coke"],
    "Kraft Heinz": ["kraft"],
    "Lockheed Martin": ["lockheed"],
    "Ulta Beauty": ["ulta"],
    "Block": ["square"],
    "Amcor": ["amcor"],
}

_YEAR = re.compile(r"(?<!\d)(20[12]\d)(?!\d)")
_SHORT_YEAR = re.compile(r"\bFY\s?(\d{2})(?!\d)", re.IGNORECASE)
_QUARTER_YEAR = re.compile(r"\bQ[1-4](20[12]\d)(?!\d)", re.IGNORECASE)  # "Q22023", a quarter written straight onto its year

# Raised whenever the routing or pinning rules change. It is part of what names a run,
# so answers made under the old rules are never mixed with new ones.
RULES_VERSION = 2


def _names(company: str) -> list[str]:
    return [company.lower(), company.lower().replace("-", " ")] + ALIASES.get(company, [])


def years(question: str) -> set[str]:
    found = set(_YEAR.findall(question))
    found |= {f"20{yy}" for yy in _SHORT_YEAR.findall(question)}
    found |= set(_QUARTER_YEAR.findall(question))
    return found


def route(question: str, table: dict[str, dict]) -> list[str] | None:
    """The filings to search for this question. None means no company was recognised, so search everything.

    `table` maps a filing's name to its details: company and doc_period.
    """
    text = question.lower()
    companies = {
        info["company"]
        for info in table.values()
        if any(re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", text) for name in _names(info["company"]))
    }
    if not companies:
        return None
    filings = sorted(name for name, info in table.items() if info["company"] in companies)
    wanted = years(question)
    in_year = [name for name in filings if str(table[name].get("doc_period")) in wanted]
    return in_year or filings
