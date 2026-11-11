"""Paths, model names and budget limits, kept in one place so every run records the same facts."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("FOOTNOTE_HOME")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
MANIFESTS_DIR = ROOT / "manifests"
# The ledger lives outside runs/, so that clearing old runs can never reset the spend.
LEDGER_PATH = ROOT / "ledger.sqlite"
ENV_FILE = ROOT / ".env"

# The system under test answers with the cheap model. The default judge is a
# different, stronger model, because a model grading its own answers tends to
# favour them. docs/evaluation.md says when the cheap model may judge instead.
ANSWER_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-sonnet-5-5"

# The project may never spend more than this on the Claude API. The ledger refuses
# any call that would pass it. The limit was $50 at first. On 2026-11-10 the ceiling
# for the project was raised to $100, to pay for the tests of pages that mislead.
# The cap stays at $90, so that a bill the ledger undercounts by a few dollars is
# still under the ceiling. Raising it means editing this line on purpose.
HARD_CAP_USD = 90.00

# One line per kind of spending. The lines add up to the hard cap, and a call that
# would overrun its own line is refused even when the total still has room.
BUDGET_LINES: dict[str, float] = {
    "development": 4.00,  # runs on the dev questions, where choices are made
    "grading": 18.00,  # the judge
    "heldout": 4.00,  # the final runs on all 150 questions
    "comparison": 16.00,  # the same systems on a stronger model
    "changing_docs": 42.00,  # the tests of pages that mislead: amended, wrong and injected
    "reserve": 6.00,
}

# Anything run on a Gemini model is paid from Google Cloud credit and not from the
# Claude budget, so it has its own ledger file and its own cap. The credit is a fixed
# amount and nothing may be charged beyond it, so the cap stays under the credit.
# On 2026-11-10 the ledger was found to have counted Gemini at half the price the
# Cloud console showed. Recounted at the console's price, the runs behind results/
# cost $38.83, which is over this cap. Raising it means editing this line.
GOOGLE_MODEL = "gemini-3.8-flash"
GOOGLE_CAP_USD = 35.00
GOOGLE_BUDGET_LINES: dict[str, float] = {
    "adk": 3.00,  # single questions to the ADK agent
    "comparison": 20.00,  # the same systems on Gemini
    "changing_docs": 12.00,  # the amended-filing study on Gemini
}
GOOGLE_LEDGER_PATH = ROOT / "ledger-google.sqlite"

# Claude Haiku 5.5 does not accept sampling parameters, so temperature cannot be
# pinned to zero. Answers can differ between runs, and every reported number
# therefore comes from repeated runs with the spread shown next to it.
DEFAULT_REPS = 3

SPLIT_SEED = 20261110
DEV_FRACTION = 1 / 3
