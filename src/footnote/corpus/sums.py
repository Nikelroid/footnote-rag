"""Find which lines of a financial statement add up to which totals.

A statement is a list of amounts with totals in between: three kinds of
inventory, then "Total inventories", then more lines, then "Total current
assets". Reading down the first column, a row is a total when it equals the sum
of the rows just before it that no earlier total has used up. That one rule
recovers the whole tree of a balance sheet.

It is needed for a fair test of amended filings. A model that sees two versions
of a page checks the arithmetic, and it sides with the version whose lines add
up. Changing one amount alone breaks a total, so that edit tests the arithmetic
and not the update. Raising one line and lowering another under the same total
leaves every total as it was. That is what a reclassification does in a real
amended filing, and it is the edit `move_edit` makes.
"""

from __future__ import annotations

import re

from .edits import CounterfactualEdit, _factor, _format, _on_page
from .probes import _DERIVED, _EMPTY_CELL, _FILLER, Probe

_CELL = re.compile(r"^(\()?\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\)?%?$")

# Raised whenever `move_edit` changes what it writes. It is part of what names a run of
# the amended-filing study, so answers about pages made by older rules are never counted.
MOVE_RULES = 2

# Lines that are results of other lines even when the page does not show the sum. Such a
# line is never the one that gives up an amount, because the lines behind it would no longer match it.
_RESULTS = ("income before", "earnings before", "operating income", "operating profit", "comprehensive income", "end of", "beginning of")


def first_column(page_text: str) -> list[tuple[str, str, float]]:
    """(label, amount as printed, value) for every row, top to bottom. An amount in parentheses is negative."""
    out: list[tuple[str, str, float]] = []
    label: str | None = None
    for raw in page_text.split("\n"):
        line = raw.strip()
        if not line or line in _FILLER:
            continue
        if line in _EMPTY_CELL:
            label = None  # the first column is empty, so the next amount belongs to another column
            continue
        cell = _CELL.match(line)
        if cell is None:
            label = line
        elif label is not None:
            value = float(cell.group(2).replace(",", ""))
            out.append((label, cell.group(2), -value if cell.group(1) else value))
            label = None
    return out


def totals(values: list[float]) -> dict[int, list[int]]:
    """The position of every total, with the positions of the rows it adds up."""
    open_rows: list[int] = []  # rows that no total has used yet
    found: dict[int, list[int]] = {}
    for i, value in enumerate(values):
        for k in range(2, len(open_rows) + 1):
            block = open_rows[-k:]
            if value != 0 and abs(sum(values[j] for j in block) - value) < 0.5:
                found[i] = block
                del open_rows[-k:]
                break
        open_rows.append(i)
    return found


def _locate(probe: Probe, rows: list[tuple[str, str, float]]) -> int | None:
    at = [i for i, (label, amount, _) in enumerate(rows) if label == probe.label and amount == probe.answer]
    return at[0] if len(at) == 1 else None


def in_a_sum(probe: Probe, page_text: str) -> bool:
    """True when the page itself lets a reader check the probe's amount, because it is a total or part of one."""
    rows = first_column(page_text)
    at = _locate(probe, rows)
    if at is None:
        return False
    found = totals([value for _, _, value in rows])
    return at in found or any(at in block for block in found.values())


def _plain(label: str) -> bool:
    """True for a line that is not itself a total or a result of other lines, judged by its name."""
    name = label.lower()
    return not name.startswith(_DERIVED) and not any(word in name for word in _RESULTS)


def move_edit(probe: Probe, page_text: str) -> CounterfactualEdit | None:
    """An amendment that moves part of a neighbouring line onto the probe's line, so every total still adds up.

    None when the page shows no total for the line, or no neighbour can give up the amount.
    """
    rows = first_column(page_text)
    at = _locate(probe, rows)
    if at is None or rows[at][2] <= 0:
        return None
    found = totals([value for _, _, value in rows])
    block = next((b for b in found.values() if at in b), None)
    if at in found or block is None:
        return None
    value = rows[at][2]
    # The largest neighbour is tried first, because the same amount changes it least. A total is never
    # a neighbour, whether the sum was found on the page or the line is only named like one.
    neighbours = sorted((j for j in block if j != at and j not in found and _plain(rows[j][0])), key=lambda j: -rows[j][2])
    for attempt in range(5):
        moved = round(value * (_factor(f"{probe.probe_id}:move:{attempt}") - 1))
        new_text = _format(value + moved, 0, "," in probe.answer)
        for j in neighbours:
            _, printed, other = rows[j]
            if "." in printed or other - moved < 0.7 * other or len(_on_page(printed).findall(page_text)) != 1:
                continue  # the neighbour keeps at least 70% of its amount, and that amount must be printed once
            other_text = _format(other - moved, 0, "," in printed)
            if new_text == other_text or any(_on_page(text).search(page_text) for text in (new_text, other_text)):
                continue
            return CounterfactualEdit(
                question_id=probe.probe_id,
                doc_name=probe.doc_name,
                page=probe.page,
                old_text=probe.answer,
                new_text=new_text,
                old_answer=probe.answer,
                new_answer=new_text,
                occurrences=1,
                also=((printed, other_text),),
            )
    return None
