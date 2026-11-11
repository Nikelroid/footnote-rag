import threading

import pytest

from footnote import config
from footnote.ledger import BudgetExceeded, Ledger
from footnote.pricing import Usage


def make(cap=1.0, lines=None):
    return Ledger(":memory:", cap_usd=cap, lines=lines or {"development": 0.6, "grading": 0.4})


def test_reserve_settle_and_release():
    ledger = make()
    a = ledger.reserve("development", "claude-haiku-5-5", 0.30)
    assert ledger.reserved() == pytest.approx(0.30)
    ledger.settle(a, "claude-haiku-5-5", Usage(input_tokens=100, output_tokens=10), 0.02, "req_1")
    assert ledger.spent() == pytest.approx(0.02) and ledger.reserved() == 0
    b = ledger.reserve("development", "claude-haiku-5-5", 0.30)
    ledger.release(b)
    assert ledger.remaining() == pytest.approx(0.98)
    with pytest.raises(RuntimeError):
        ledger.settle(b, "claude-haiku-5-5", Usage(), 0.0)


def test_total_cap_refuses_before_any_money_moves():
    ledger = make(cap=0.5, lines={"development": 10.0})
    ledger.reserve("development", "m", 0.4)
    with pytest.raises(BudgetExceeded, match="cap"):
        ledger.reserve("development", "m", 0.2)
    assert ledger.reserved() == pytest.approx(0.4)


def test_line_cap_refuses_even_when_the_total_has_room():
    ledger = make()
    ledger.reserve("grading", "m", 0.35)
    with pytest.raises(BudgetExceeded, match="grading"):
        ledger.reserve("grading", "m", 0.10)
    ledger.reserve("development", "m", 0.10)


def test_unknown_line_and_cap_above_hard_cap_are_rejected():
    with pytest.raises(KeyError):
        make().reserve("snacks", "m", 0.01)
    with pytest.raises(ValueError):
        Ledger(":memory:", cap_usd=config.HARD_CAP_USD + 1)


def test_racing_workers_cannot_overspend():
    ledger = make(cap=1.0, lines={"development": 1.0})
    granted, refused = [], []

    def worker():
        try:
            granted.append(ledger.reserve("development", "m", 0.1))
        except BudgetExceeded:
            refused.append(1)

    threads = [threading.Thread(target=worker) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(granted) == 10 and len(refused) == 30
    assert ledger.reserved() == pytest.approx(1.0)


def test_cap_survives_reopening_the_file(tmp_path):
    path = tmp_path / "ledger.sqlite"
    first = Ledger(path, cap_usd=1.0, lines={"development": 1.0})
    call = first.reserve("development", "m", 0.9)
    first.settle(call, "m", Usage(input_tokens=5), 0.9)
    first.close()
    with pytest.raises(BudgetExceeded):
        Ledger(path, cap_usd=1.0, lines={"development": 1.0}).reserve("development", "m", 0.2)


def test_summary_is_recomputed_from_rows():
    ledger = make()
    for cost in (0.01, 0.02):
        call = ledger.reserve("grading", "m", 0.05)
        ledger.settle(call, "m", Usage(input_tokens=10, output_tokens=2), cost)
    row = next(r for r in ledger.summary() if r["line"] == "grading")
    assert row["calls"] == 2 and row["spent_usd"] == pytest.approx(0.03) and row["prompt_tokens"] == 20


def test_overruns_and_unconfirmed_spend_are_visible():
    ledger = make()
    a = ledger.reserve("development", "m", 0.10)
    assert ledger.settle(a, "m", Usage(input_tokens=1), 0.04) == 0.0
    b = ledger.reserve("development", "m", 0.10)
    assert ledger.settle(b, "m", Usage(input_tokens=1), 0.13) == pytest.approx(0.03)  # cost more than was reserved
    c = ledger.reserve("development", "m", 0.10)
    ledger.keep_unconfirmed(c)
    ledger.add_unconfirmed("development", "m", 0.10)
    health = ledger.health()
    assert health["overrun_calls"] == 1 and health["largest_overrun_usd"] == pytest.approx(0.03)
    assert health["unconfirmed_calls"] == 2 and health["unconfirmed_usd"] == pytest.approx(0.20)
    assert health["highest_cost_to_reservation"] == pytest.approx(1.3)
    assert ledger.spent() == pytest.approx(0.04 + 0.13 + 0.20) and ledger.reserved() == 0


def test_a_reservation_left_by_a_dead_process_stays_counted(tmp_path):
    path = tmp_path / "ledger.sqlite"
    first = Ledger(path, cap_usd=1.0, lines={"development": 1.0})
    first.reserve("development", "m", 0.4)
    first._db.execute("UPDATE calls SET ts = '2020-01-01T00:00:00+00:00'")
    first.close()
    reopened = Ledger(path, cap_usd=1.0, lines={"development": 1.0})
    assert reopened.reserved() == 0 and reopened.spent() == pytest.approx(0.4)
    assert reopened.health()["unconfirmed_calls"] == 1
