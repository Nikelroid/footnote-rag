"""A free, deterministic check for questions whose reference answer is one number.

About a third of the FinanceBench questions ask for a single figure. For those,
this check needs no model. It is used as a second opinion next to the judge, and
its own agreement with the human labels is measured before it is trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SCALES = {
    "thousand": 1e3,
    "k": 1e3,
    "million": 1e6,
    "mn": 1e6,
    "mm": 1e6,
    "m": 1e6,
    "billion": 1e9,
    "bn": 1e9,
    "b": 1e9,
    "trillion": 1e12,
}

_MINUS = "-" + chr(0x2212) + chr(0x2013)  # hyphen, minus sign, en dash
_CURRENCY = re.escape("$" + chr(0x20AC) + chr(0x00A3))  # dollar, euro, pound

NUMBER = re.compile(
    rf"""
    (?<![\w.,])                                   # not glued to a word, so FY2018 and Q2 are skipped
    (?P<neg>[{_MINUS}]\s?|\(\s?)?                 # minus sign, or an accounting parenthesis
    (?P<cur>[{_CURRENCY}]\s?)?
    (?P<num>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)
    (?:\s?(?P<unit>%|percentage\s+points?|percent|trillion|billion|million|thousand|bn|mn|mm|[kmb])(?![a-z]))?
    """,
    re.VERBOSE | re.IGNORECASE,
)

FINAL_MARKER = re.compile(r"final\s+answer\s*[:\-]", re.IGNORECASE)

# Most questions name the unit of the answer, as in "(in USD millions)".
QUESTION_UNIT = re.compile(
    r"\bUSD\s+(thousand|million|billion)s?\b|\b(thousand|million|billion)s?\s+of\s+(?:USD|US\s+dollars|dollars)\b",
    re.IGNORECASE,
)

# Used only when the question names no unit and the reference is an amount of
# money. The candidate may then give the same amount in other units.
UNIT_FACTORS = tuple(10.0 ** (3 * k) for k in range(-4, 5))

# Two answers within 0.1% are the same answer. This is the rounding left by a
# calculation, such as 93.88 against 93.86.
REL_TOL = 0.001

# An answer one unit off in the last digit is accepted only when that is also a
# small relative error, such as 7.8% against 7.9%.
LAST_DIGIT_REL_LIMIT = 0.02


@dataclass(frozen=True)
class Num:
    value: float  # with its scale word applied, so "1.5 billion" is 1.5e9
    raw: float  # the number as printed, so "1.5 billion" is 1.5
    percent: bool
    currency: bool
    scaled: bool  # a scale word such as "million" follows the number
    ulp: float  # one unit in the last written digit, once trailing zeros are dropped


@dataclass(frozen=True)
class NumericVerdict:
    applicable: bool  # the reference answer is a single number
    match: bool | None = None
    scope: str | None = None  # "final" when a "Final answer:" line was used, else "any"
    gold: float | None = None
    matched: float | None = None

    def to_record(self) -> dict:
        return {"applicable": self.applicable, "match": self.match, "scope": self.scope, "gold": self.gold, "matched": self.matched}


def _to_num(m: re.Match) -> Num:
    text = m.group("num").replace(",", "")
    raw = float(text)
    if m.group("neg"):
        raw = -raw
    unit = (m.group("unit") or "").lower()
    scale = SCALES.get(unit, 1.0)
    # "$1577.00" is written with two decimals but is only known to the unit, so
    # precision is taken from the digits that carry information.
    decimals = len(text.split(".")[1].rstrip("0")) if "." in text else 0
    return Num(
        value=raw * scale,
        raw=raw,
        percent=unit == "%" or unit.startswith("percent"),
        currency=bool(m.group("cur")),
        scaled=unit in SCALES,
        ulp=10.0**-decimals * scale,
    )


def find_numbers(text: str) -> list[Num]:
    return [_to_num(m) for m in NUMBER.finditer(text)]


def parse_gold(answer: str) -> Num | None:
    """Return the number when the whole reference answer is one number, else None."""
    text = answer.strip().rstrip(".").strip()
    m = NUMBER.fullmatch(text)
    if m is None:
        closed = re.fullmatch(r"\((.+)\)", text)  # an accounting negative such as (0.02)
        if closed is None:
            return None
        inner = NUMBER.fullmatch(closed.group(1).strip())
        if inner is None:
            return None
        n = _to_num(inner)
        return Num(-abs(n.value), -abs(n.raw), n.percent, n.currency, n.scaled, n.ulp)
    return _to_num(m)


def question_unit(question: str | None) -> float | None:
    """The unit the question asks for, 1e6 for "in USD millions". None when it names none, or more than one."""
    if not question:
        return None
    found = {SCALES[(m.group(1) or m.group(2)).lower()] for m in QUESTION_UNIT.finditer(question)}
    return found.pop() if len(found) == 1 else None


def _close(x: float, g: float, ulp: float) -> bool:
    diff = abs(x - g)
    if diff <= 0.5 * ulp * (1 + 1e-9):  # equal once rounded to the reference's precision
        return True
    if diff <= REL_TOL * g:
        return True
    return diff <= ulp * (1 + 1e-9) and diff <= LAST_DIGIT_REL_LIMIT * g


def numbers_match(gold: Num, cand: Num, unit: float | None = None) -> bool:
    """Compare magnitudes. Signs are ignored, because answers state them in words as often as in symbols."""
    g = abs(gold.value)
    if g == 0:
        return cand.value == 0
    if gold.percent or cand.percent:
        if gold.currency or cand.currency or cand.scaled:
            return False  # a percentage is never an amount of money
        x = abs(cand.value)
        if gold.percent and cand.percent:
            return _close(x, g, gold.ulp)
        if gold.percent:
            # The answer drops the sign: "1.9", or the fraction "0.019", for a reference of 1.9%.
            return _close(x, g, gold.ulp) or _close(x * 100, g, gold.ulp)
        return _close(x / 100, g, gold.ulp)  # "66%" for a reference of 0.66
    printed, full = abs(cand.raw), abs(cand.value)
    if unit is not None:
        # The question names the unit, so the reference is a known amount.
        if cand.scaled:
            return _close(full, g * unit, gold.ulp * unit)
        return _close(printed, g, gold.ulp) or _close(printed, g * unit, gold.ulp * unit)
    if _close(printed, g, gold.ulp):
        return True
    if gold.currency:
        return any(_close(full * f, g, gold.ulp) for f in UNIT_FACTORS)
    if cand.currency or cand.scaled:
        # The reference is a bare number. Another unit is accepted only on an almost exact match.
        return any(abs(full * f - g) <= REL_TOL * g for f in UNIT_FACTORS)
    return False


def final_answer_text(answer: str) -> str | None:
    """The text after the last 'Final answer:' marker, if the answer has one."""
    last = None
    for m in FINAL_MARKER.finditer(answer):
        last = m
    return answer[last.end() :].strip() if last else None


def grade(gold_answer: str, candidate: str, question: str | None = None) -> NumericVerdict:
    gold = parse_gold(gold_answer)
    if gold is None:
        return NumericVerdict(applicable=False)
    unit = question_unit(question)
    final = final_answer_text(candidate)
    scope, text = ("final", final) if final is not None else ("any", candidate)
    for cand in find_numbers(text):
        if numbers_match(gold, cand, unit):
            return NumericVerdict(True, True, scope, gold.value, cand.value)
    return NumericVerdict(True, False, scope, gold.value, None)
