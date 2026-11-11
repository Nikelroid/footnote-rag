"""A versioned record of what is in the index.

Filings do not stand still. A new quarter arrives, a company files an amendment,
a number is restated, a document is withdrawn. The registry is the single source
of truth for which text is searchable at which time. It answers four questions.

1. Has this document changed at all? One hash of the source file says so before
   any parsing is done.
2. Which chunks are new? Chunks are identified by a hash of their text, so text
   that did not change is never embedded again, in this document or any other.
3. Which chunks are stale? When a new version arrives the old one is closed, not
   deleted, and search over current text can no longer return its chunks.
4. What did the corpus say on a given date? Because old versions are kept, a
   question can be answered as of any past date.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field

SCHEMA = """
CREATE TABLE IF NOT EXISTS versions (
    doc_id TEXT NOT NULL,
    version TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    source_hash TEXT,
    n_chunks INTEGER NOT NULL,
    PRIMARY KEY (doc_id, version)
);
CREATE TABLE IF NOT EXISTS placements (
    doc_id TEXT NOT NULL,
    version TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    page INTEGER NOT NULL,
    kind TEXT NOT NULL,
    chunk_hash TEXT NOT NULL,
    PRIMARY KEY (doc_id, version, ordinal)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_hash TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedded INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS placements_hash ON placements (chunk_hash);
"""


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    kind: str = "text"


@dataclass(frozen=True)
class ActiveChunk:
    doc_id: str
    version: str
    ordinal: int
    page: int
    kind: str
    chunk_hash: str
    text: str


@dataclass
class ChangeSet:
    """What one update did to the index."""

    doc_id: str
    version: str
    previous_version: str | None
    noop: bool = False  # the same version was applied again
    unchanged: list[str] = field(default_factory=list)  # chunk hashes kept as they are
    added: list[str] = field(default_factory=list)  # new to this document
    removed: list[str] = field(default_factory=list)  # no longer in this document
    to_embed: list[str] = field(default_factory=list)  # never seen anywhere, so they need an embedding

    @property
    def total(self) -> int:
        return len(self.unchanged) + len(self.added)

    @property
    def embed_ratio(self) -> float:
        """Share of the new version that has to be embedded. 1.0 means a full rebuild."""
        return len(self.to_embed) / self.total if self.total else 0.0


def chunk_hash(text: str) -> str:
    """Identity of a chunk. Whitespace is normalised so that re-extracting a page does not look like a change."""
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()[:32]


def source_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class CorpusRegistry:
    def __init__(self, path: str = ":memory:"):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.executescript(SCHEMA)

    @contextmanager
    def _tx(self):
        # An update is all or nothing. A reader never sees a document half replaced.
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            else:
                self._db.execute("COMMIT")

    def current_version(self, doc_id: str) -> tuple[str, str, str | None] | None:
        """(version, valid_from, source_hash) of the version that is current, if any."""
        with self._lock:  # never read a document in the middle of its own update
            row = self._db.execute(
                "SELECT version, valid_from, source_hash FROM versions WHERE doc_id = ? AND valid_to IS NULL", (doc_id,)
            ).fetchone()
        return tuple(row) if row else None

    def needs_update(self, doc_id: str, new_source_hash: str) -> bool:
        """False when the source file is byte for byte the one already indexed."""
        current = self.current_version(doc_id)
        return current is None or current[2] != new_source_hash

    def apply(self, doc_id: str, version: str, chunks: list[Chunk], valid_from: str, source: str | None = None) -> ChangeSet:
        """Make `version` the current version of `doc_id` from `valid_from` on."""
        hashes = [chunk_hash(c.text) for c in chunks]
        with self._tx() as db:
            used = db.execute("SELECT valid_to FROM versions WHERE doc_id = ? AND version = ?", (doc_id, version)).fetchone()
            if used is not None:
                stored = [
                    r[0]
                    for r in db.execute(
                        "SELECT chunk_hash FROM placements WHERE doc_id = ? AND version = ? ORDER BY ordinal", (doc_id, version)
                    )
                ]
                if used[0] is None and stored == hashes:
                    return ChangeSet(doc_id, version, None, noop=True)  # the same update, applied again
                raise ValueError(f"{doc_id}: the version name {version!r} was already used. New text needs a new version name.")
            current = db.execute(
                "SELECT version, valid_from FROM versions WHERE doc_id = ? AND valid_to IS NULL", (doc_id,)
            ).fetchone()
            previous_hashes: list[str] = []
            if current is None:
                # The document is new or was withdrawn. It cannot come back at a date before it left.
                left = db.execute("SELECT MAX(valid_to) FROM versions WHERE doc_id = ?", (doc_id,)).fetchone()[0]
                if left is not None and valid_from < left:
                    raise ValueError(f"{doc_id}: {valid_from} is earlier than {left}, when its last version ended")
            if current is not None:
                if valid_from < current[1]:
                    raise ValueError(f"{doc_id}: {valid_from} is earlier than the current version from {current[1]}")
                previous_hashes = [
                    r[0]
                    for r in db.execute(
                        "SELECT chunk_hash FROM placements WHERE doc_id = ? AND version = ? ORDER BY ordinal",
                        (doc_id, current[0]),
                    )
                ]
                db.execute("UPDATE versions SET valid_to = ? WHERE doc_id = ? AND version = ?", (valid_from, doc_id, current[0]))
            change = ChangeSet(doc_id, version, current[0] if current else None)
            # Counted as multisets, because the same text can appear twice in one filing.
            before, after = Counter(previous_hashes), Counter(hashes)
            for h, n in after.items():
                kept = min(n, before.get(h, 0))
                change.unchanged += [h] * kept
                change.added += [h] * (n - kept)
            for h, n in before.items():
                change.removed += [h] * max(0, n - after.get(h, 0))
            embedded: dict[str, int] = {}
            distinct = sorted(set(hashes))
            for i in range(0, len(distinct), 500):
                batch = distinct[i : i + 500]
                marks = ",".join("?" * len(batch))
                embedded.update(db.execute(f"SELECT chunk_hash, embedded FROM chunks WHERE chunk_hash IN ({marks})", batch))
            # Text that has no vector yet: new text, and text seen before that is still waiting for one.
            change.to_embed = [h for h in distinct if not embedded.get(h)]
            text_by_hash = {h: c.text for h, c in zip(hashes, chunks)}
            db.executemany(
                "INSERT INTO chunks (chunk_hash, text) VALUES (?, ?)", [(h, text_by_hash[h]) for h in distinct if h not in embedded]
            )
            db.execute(
                "INSERT INTO versions (doc_id, version, valid_from, source_hash, n_chunks) VALUES (?, ?, ?, ?, ?)",
                (doc_id, version, valid_from, source, len(chunks)),
            )
            db.executemany(
                "INSERT INTO placements (doc_id, version, ordinal, page, kind, chunk_hash) VALUES (?, ?, ?, ?, ?, ?)",
                [(doc_id, version, i, c.page, c.kind, h) for i, (c, h) in enumerate(zip(chunks, hashes))],
            )
            return change

    def retire(self, doc_id: str, valid_to: str) -> ChangeSet:
        """Withdraw a document. Its text stays in history and leaves current search."""
        with self._tx() as db:
            current = db.execute(
                "SELECT version, valid_from FROM versions WHERE doc_id = ? AND valid_to IS NULL", (doc_id,)
            ).fetchone()
            if current is None:
                raise KeyError(f"{doc_id} has no current version")
            if valid_to < current[1]:
                raise ValueError(f"{doc_id}: {valid_to} is earlier than the current version from {current[1]}")
            removed = [
                r[0] for r in db.execute("SELECT chunk_hash FROM placements WHERE doc_id = ? AND version = ?", (doc_id, current[0]))
            ]
            db.execute("UPDATE versions SET valid_to = ? WHERE doc_id = ? AND version = ?", (valid_to, doc_id, current[0]))
            return ChangeSet(doc_id, current[0], current[0], removed=removed)

    def pending_embeddings(self) -> list[tuple[str, str]]:
        """(chunk_hash, text) for every chunk that still needs a vector."""
        with self._lock:
            return [tuple(r) for r in self._db.execute("SELECT chunk_hash, text FROM chunks WHERE embedded = 0 ORDER BY chunk_hash")]

    def mark_embedded(self, hashes: list[str]) -> None:
        with self._tx() as db:
            db.executemany("UPDATE chunks SET embedded = 1 WHERE chunk_hash = ?", [(h,) for h in hashes])

    def active(self, as_of: str | None = None, doc_id: str | None = None) -> list[ActiveChunk]:
        """The chunks that search may return: the current ones, or those that were current at `as_of`."""
        if as_of is None:
            where, args = "v.valid_to IS NULL", []
        else:
            where, args = "v.valid_from <= ? AND (v.valid_to IS NULL OR v.valid_to > ?)", [as_of, as_of]
        if doc_id is not None:
            where += " AND v.doc_id = ?"
            args.append(doc_id)
        sql = (
            "SELECT p.doc_id, p.version, p.ordinal, p.page, p.kind, p.chunk_hash, c.text "
            "FROM versions v JOIN placements p ON p.doc_id = v.doc_id AND p.version = v.version "
            f"JOIN chunks c ON c.chunk_hash = p.chunk_hash WHERE {where} ORDER BY p.doc_id, p.ordinal"
        )
        with self._lock:
            return [ActiveChunk(*r) for r in self._db.execute(sql, args)]

    def history(self, doc_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT version, valid_from, valid_to, n_chunks FROM versions WHERE doc_id = ? ORDER BY valid_from, rowid", (doc_id,)
            ).fetchall()
        return [{"version": r[0], "valid_from": r[1], "valid_to": r[2], "n_chunks": r[3]} for r in rows]

    def close(self) -> None:
        self._db.close()
