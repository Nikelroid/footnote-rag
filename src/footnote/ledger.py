"""A spend ledger with a hard cap.

Every API call reserves its worst-case cost before it is sent and settles to the
real cost afterwards. The reservation is taken inside one database transaction,
so two workers can never both spend the last dollar. The ledger is a SQLite file,
so the cap also holds across processes and across days.

A request that fails after it was sent may still have been billed. Its
reservation is then kept as spend, at the worst-case amount, and marked as
unconfirmed. The ledger can therefore overstate the real bill. It should never
understate it.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config
from .pricing import Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    line TEXT NOT NULL,
    run_id TEXT,
    purpose TEXT,
    model TEXT NOT NULL,
    batch INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_5m_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_1h_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    reserved_usd REAL NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL,
    unconfirmed INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL CHECK (state IN ('reserved', 'settled', 'released')),
    request_id TEXT
);
CREATE INDEX IF NOT EXISTS calls_state ON calls (state);
"""

# A normal request never takes this long. A reservation older than this belongs
# to a process that died. Nobody knows whether its request was billed, so it is
# kept as unconfirmed spend. Batch jobs can run for a day and are left alone.
STALE_AFTER = timedelta(hours=2)


class BudgetExceeded(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(
        self,
        path: Path | str | None = None,
        cap_usd: float = config.HARD_CAP_USD,
        lines: dict[str, float] | None = None,
    ):
        if cap_usd > config.HARD_CAP_USD + 1e-9:
            raise ValueError(f"cap {cap_usd} is above the hard cap {config.HARD_CAP_USD}")
        self.cap_usd = cap_usd
        self.lines = dict(config.BUDGET_LINES if lines is None else lines)
        self.path = str(path if path is not None else config.LEDGER_PATH)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self._db.executescript(SCHEMA)
        self.close_stale()

    @contextmanager
    def _tx(self):
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    def _committed(self, db: sqlite3.Connection, line: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE state IN ('reserved', 'settled')"
        args: tuple = ()
        if line is not None:
            sql += " AND line = ?"
            args = (line,)
        return float(db.execute(sql, args).fetchone()[0])

    def reserve(
        self,
        line: str,
        model: str,
        worst_case_usd: float,
        run_id: str | None = None,
        purpose: str | None = None,
        batch: bool = False,
    ) -> int:
        if line not in self.lines:
            raise KeyError(f"unknown budget line {line!r}, expected one of {sorted(self.lines)}")
        if worst_case_usd < 0:
            raise ValueError("a reservation cannot be negative")
        with self._tx() as db:
            total = self._committed(db)
            if total + worst_case_usd > self.cap_usd + 1e-9:
                raise BudgetExceeded(
                    f"refused: ${total:.4f} is committed and this call could cost ${worst_case_usd:.4f}, "
                    f"which passes the ${self.cap_usd:.2f} cap"
                )
            on_line = self._committed(db, line)
            if on_line + worst_case_usd > self.lines[line] + 1e-9:
                raise BudgetExceeded(
                    f"refused: budget line {line!r} has ${on_line:.4f} committed of ${self.lines[line]:.2f} "
                    f"and this call could cost ${worst_case_usd:.4f}"
                )
            cur = db.execute(
                "INSERT INTO calls (ts, line, run_id, purpose, model, batch, reserved_usd, cost_usd, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'reserved')",
                (_now(), line, run_id, purpose, model, int(batch), worst_case_usd, worst_case_usd),
            )
            return int(cur.lastrowid)

    def settle(self, call_id: int, model: str, usage: Usage, actual_usd: float, request_id: str | None = None) -> float:
        """Replace a reservation with the real cost. Returns how far the cost went over the reservation, normally 0."""
        with self._tx() as db:
            row = db.execute("SELECT reserved_usd FROM calls WHERE id = ? AND state = 'reserved'", (call_id,)).fetchone()
            if row is None:
                raise RuntimeError(f"call {call_id} is not an open reservation")
            db.execute(
                "UPDATE calls SET state = 'settled', model = ?, cost_usd = ?, request_id = ?, input_tokens = ?, "
                "output_tokens = ?, cache_write_5m_tokens = ?, cache_write_1h_tokens = ?, cache_read_tokens = ? WHERE id = ?",
                (
                    model,
                    actual_usd,
                    request_id,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_write_5m_tokens,
                    usage.cache_write_1h_tokens,
                    usage.cache_read_tokens,
                    call_id,
                ),
            )
            return max(0.0, actual_usd - float(row[0]))

    def release(self, call_id: int) -> None:
        """Drop a reservation whose request was certainly not billed, such as one the API rejected."""
        with self._tx() as db:
            db.execute("UPDATE calls SET state = 'released', cost_usd = 0 WHERE id = ? AND state = 'reserved'", (call_id,))

    def keep_unconfirmed(self, call_id: int) -> None:
        """The request failed after it was sent. Keep its reservation as spend, since it may have been billed."""
        with self._tx() as db:
            db.execute("UPDATE calls SET state = 'settled', unconfirmed = 1 WHERE id = ? AND state = 'reserved'", (call_id,))

    def add_unconfirmed(self, line: str, model: str, amount_usd: float, run_id: str | None = None, purpose: str | None = None) -> None:
        """Record a further failed attempt of a call that may have been billed. It has already happened, so no cap applies."""
        with self._tx() as db:
            db.execute(
                "INSERT INTO calls (ts, line, run_id, purpose, model, reserved_usd, cost_usd, unconfirmed, state) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 1, 'settled')",
                (_now(), line, run_id, purpose, model, amount_usd, amount_usd),
            )

    def close_stale(self) -> int:
        cutoff = (datetime.now(timezone.utc) - STALE_AFTER).isoformat(timespec="seconds")
        with self._tx() as db:
            cur = db.execute(
                "UPDATE calls SET state = 'settled', unconfirmed = 1 WHERE state = 'reserved' AND batch = 0 AND ts < ?", (cutoff,)
            )
            return cur.rowcount

    def spent(self, line: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE state = 'settled'"
        args: tuple = ()
        if line is not None:
            sql += " AND line = ?"
            args = (line,)
        with self._lock:
            return float(self._db.execute(sql, args).fetchone()[0])

    def reserved(self) -> float:
        with self._lock:
            return float(
                self._db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM calls WHERE state = 'reserved'").fetchone()[0]
            )

    def remaining(self) -> float:
        return self.cap_usd - self.spent() - self.reserved()

    def summary(self) -> list[dict]:
        """Spend per budget line, recomputed from the raw rows."""
        with self._lock:
            rows = self._db.execute(
                "SELECT line, COUNT(*), COALESCE(SUM(cost_usd), 0), COALESCE(SUM(input_tokens + cache_write_5m_tokens "
                "+ cache_write_1h_tokens + cache_read_tokens), 0), COALESCE(SUM(output_tokens), 0), COALESCE(SUM(unconfirmed), 0) "
                "FROM calls WHERE state = 'settled' GROUP BY line"
            ).fetchall()
        by_line = {r[0]: r[1:] for r in rows}
        out = []
        for line, cap in self.lines.items():
            calls, spent, prompt, output, unconfirmed = by_line.get(line, (0, 0.0, 0, 0, 0))
            out.append(
                {
                    "line": line,
                    "cap_usd": cap,
                    "spent_usd": spent,
                    "calls": calls,
                    "prompt_tokens": prompt,
                    "output_tokens": output,
                    "unconfirmed_calls": unconfirmed,
                }
            )
        return out

    def health(self) -> dict:
        """Two warning signs: spend that could not be confirmed, and calls that cost more than was reserved for them."""
        with self._lock:
            unconfirmed = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd), 0) FROM calls WHERE state = 'settled' AND unconfirmed = 1"
            ).fetchone()
            overruns = self._db.execute(
                "SELECT COUNT(*), COALESCE(MAX(cost_usd - reserved_usd), 0), COALESCE(MAX(cost_usd / reserved_usd), 0) FROM calls "
                "WHERE state = 'settled' AND unconfirmed = 0 AND cost_usd > reserved_usd + 1e-12"
            ).fetchone()
            closest = self._db.execute(
                "SELECT COALESCE(MAX(cost_usd / reserved_usd), 0) FROM calls WHERE state = 'settled' AND unconfirmed = 0 AND reserved_usd > 0"
            ).fetchone()
        return {
            "unconfirmed_calls": unconfirmed[0],
            "unconfirmed_usd": float(unconfirmed[1]),
            "overrun_calls": overruns[0],
            "largest_overrun_usd": float(overruns[1]),
            # How close any real cost came to its reservation. Under 1 means the bound held.
            "highest_cost_to_reservation": float(closest[0]),
        }

    def close(self) -> None:
        self._db.close()


def google_ledger() -> Ledger:
    """The ledger for everything that runs on Gemini. That is paid from Google Cloud credit, so it is kept apart from the Claude budget."""
    return Ledger(config.GOOGLE_LEDGER_PATH, cap_usd=config.GOOGLE_CAP_USD, lines=config.GOOGLE_BUDGET_LINES)
