"""Durable storage for task slices — the sidecar that makes a slice id mean something.

Lives in its own WAL SQLite file (``.repowise/slices/slices.db``), the same
shape :class:`repowise.core.sessions.staging.SessionStagingStore` and
:class:`repowise.core.distill.store.OmissionStore` use, and for the same
reason: writes happen while an agent is mid-task, and they must never contend
with an indexing run holding ``wiki.db``.

Not in ``wiki.db``, and not only for contention. A slice is *session* state,
not *index* state. Re-indexing rewrites every graph row; it must not silently
rewrite or delete the slices an agent is working through, and a slice must
survive being carried across a re-index so ``get_task_slice`` can say "these
members no longer exist" rather than losing the id. Keeping it out of the
indexed schema also means this whole subsystem adds no migration to the shared
ORM — the sidecar owns its own four tables and its own ``PRAGMA user_version``.

Synchronous on purpose. Every call here is a handful of rows against a local
file; wrapping that in an async engine would cost more than the query. The
async callers in :mod:`repowise.core.slices.service` are doing real I/O
against ``wiki.db``, and this is the cheap part of what they do.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from repowise.core.slices.errors import SliceNotFoundError, SliceStoreUnavailableError
from repowise.core.slices.models import (
    SliceEdge,
    SliceMember,
    SliceRecord,
    WalkPolicy,
)

SLICES_DIRNAME = "slices"
SLICES_DB_FILENAME = "slices.db"

#: Slice ids are opaque and random, not content-derived. Two slices built from
#: the same task at different times are different objects — one may already
#: have been extended — so a content hash would collide them.
SLICE_ID_PREFIX = "sl_"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS slices (
    slice_id TEXT PRIMARY KEY,
    repository_id TEXT NOT NULL,
    repo_path TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    policy_json TEXT NOT NULL DEFAULT '{}',
    seeds_json TEXT NOT NULL DEFAULT '[]',
    externals_json TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    member_cap_hit INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS slice_members (
    slice_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    layer TEXT NOT NULL,
    file_path TEXT NOT NULL DEFAULT '',
    name TEXT,
    kind TEXT,
    signature TEXT,
    docstring TEXT,
    start_line INTEGER,
    end_line INTEGER,
    language TEXT NOT NULL DEFAULT '',
    is_test INTEGER NOT NULL DEFAULT 0,
    pagerank REAL NOT NULL DEFAULT 0.0,
    distance INTEGER NOT NULL DEFAULT 0,
    is_seed INTEGER NOT NULL DEFAULT 0,
    frontier_down INTEGER NOT NULL DEFAULT 0,
    frontier_up INTEGER NOT NULL DEFAULT 0,
    reference_count INTEGER NOT NULL DEFAULT 0,
    max_confidence REAL NOT NULL DEFAULT 0.0,
    query_hits INTEGER NOT NULL DEFAULT 0,
    seed_score REAL NOT NULL DEFAULT 0.0,
    score REAL NOT NULL DEFAULT 0.0,
    rank INTEGER NOT NULL DEFAULT 0,
    reasons_json TEXT NOT NULL DEFAULT '[]',
    edge_types_json TEXT NOT NULL DEFAULT '[]',
    added_revision INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (slice_id, node_id)
);
CREATE TABLE IF NOT EXISTS slice_edges (
    slice_id TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    direction TEXT NOT NULL DEFAULT 'downstream',
    PRIMARY KEY (slice_id, source, target, edge_type)
);
CREATE TABLE IF NOT EXISTS slice_events (
    slice_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    detail_json TEXT NOT NULL DEFAULT '{}',
    at REAL NOT NULL,
    PRIMARY KEY (slice_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_slices_repo ON slices(repository_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_slice_members_rank ON slice_members(slice_id, rank);
"""


def new_slice_id() -> str:
    return f"{SLICE_ID_PREFIX}{uuid.uuid4().hex[:12]}"


def default_store_path(repo_path: Path | str) -> Path:
    return Path(repo_path) / ".repowise" / SLICES_DIRNAME / SLICES_DB_FILENAME


class SliceStore:
    """Synchronous SQLite store for built slices."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except (OSError, sqlite3.Error) as exc:
            raise SliceStoreUnavailableError(str(self.db_path), str(exc)) from exc

    @classmethod
    def open_default(cls, repo_path: Path | str) -> SliceStore:
        return cls(default_store_path(repo_path))

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SliceStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- write ---------------------------------------------------------------

    def save(self, record: SliceRecord, *, event: dict[str, Any] | None = None) -> None:
        """Upsert a whole slice. Members and edges are replaced wholesale.

        Wholesale rather than differential because a slice is small (hundreds
        of rows at the cap) and a differential write would have to reason
        about rank changes across every member — extension re-ranks the whole
        slice, so almost every row changes anyway.
        """
        record.updated_at = time.time()
        try:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO slices (slice_id, repository_id, repo_path, task, "
                    "policy_json, seeds_json, externals_json, created_at, updated_at, "
                    "revision, member_cap_hit) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(slice_id) DO UPDATE SET "
                    "policy_json=excluded.policy_json, seeds_json=excluded.seeds_json, "
                    "externals_json=excluded.externals_json, "
                    "updated_at=excluded.updated_at, revision=excluded.revision, "
                    "member_cap_hit=excluded.member_cap_hit, task=excluded.task",
                    (
                        record.slice_id,
                        record.repository_id,
                        record.repo_path,
                        record.task,
                        json.dumps(record.policy.to_dict()),
                        json.dumps(record.seeds),
                        json.dumps(record.external_dependencies),
                        record.created_at,
                        record.updated_at,
                        record.revision,
                        int(record.member_cap_hit),
                    ),
                )
                self._conn.execute(
                    "DELETE FROM slice_members WHERE slice_id = ?", (record.slice_id,)
                )
                self._conn.executemany(
                    "INSERT INTO slice_members (slice_id, node_id, node_type, layer, "
                    "file_path, name, kind, signature, docstring, start_line, end_line, "
                    "language, is_test, pagerank, distance, is_seed, frontier_down, "
                    "frontier_up, reference_count, max_confidence, query_hits, seed_score, "
                    "score, rank, reasons_json, edge_types_json, added_revision) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?)",
                    [self._member_row(record.slice_id, m) for m in record.members],
                )
                self._conn.execute("DELETE FROM slice_edges WHERE slice_id = ?", (record.slice_id,))
                self._conn.executemany(
                    "INSERT OR REPLACE INTO slice_edges "
                    "(slice_id, source, target, edge_type, confidence, direction) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            record.slice_id,
                            e.source,
                            e.target,
                            e.edge_type,
                            e.confidence,
                            e.direction,
                        )
                        for e in record.edges
                    ],
                )
                if event is not None:
                    row = self._conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) + 1 AS n FROM slice_events "
                        "WHERE slice_id = ?",
                        (record.slice_id,),
                    ).fetchone()
                    self._conn.execute(
                        "INSERT INTO slice_events (slice_id, seq, kind, detail_json, at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            record.slice_id,
                            int(row["n"]),
                            str(event.get("kind", "build")),
                            json.dumps(event),
                            time.time(),
                        ),
                    )
        except sqlite3.Error as exc:
            raise SliceStoreUnavailableError(str(self.db_path), str(exc)) from exc

    @staticmethod
    def _member_row(slice_id: str, m: SliceMember) -> tuple[Any, ...]:
        return (
            slice_id,
            m.node_id,
            m.node_type,
            m.layer,
            m.file_path,
            m.name,
            m.kind,
            m.signature,
            m.docstring,
            m.start_line,
            m.end_line,
            m.language,
            int(m.is_test),
            m.pagerank,
            m.distance,
            int(m.is_seed),
            int(m.frontier_down),
            int(m.frontier_up),
            m.reference_count,
            m.max_confidence,
            m.query_hits,
            m.seed_score,
            m.score,
            m.rank,
            json.dumps(m.reasons),
            json.dumps(sorted(m.edge_types)),
            m.added_revision,
        )

    # -- read ----------------------------------------------------------------

    def load(self, slice_id: str) -> SliceRecord:
        """Load one slice, or raise :class:`SliceNotFoundError`.

        Never returns an empty record for an unknown id. A slice that exists
        and has no members is a different fact from an id that names nothing,
        and collapsing the two is what turns a typo into a silent no-op.
        """
        head = self._conn.execute("SELECT * FROM slices WHERE slice_id = ?", (slice_id,)).fetchone()
        if head is None:
            raise SliceNotFoundError(slice_id, known_ids=self.recent_ids())

        members = [
            self._member_from_row(row)
            for row in self._conn.execute(
                "SELECT * FROM slice_members WHERE slice_id = ? ORDER BY rank, node_id",
                (slice_id,),
            )
        ]
        edges = [
            SliceEdge(
                source=row["source"],
                target=row["target"],
                edge_type=row["edge_type"],
                confidence=row["confidence"],
                direction=row["direction"],
            )
            for row in self._conn.execute(
                "SELECT * FROM slice_edges WHERE slice_id = ? ORDER BY source, target, edge_type",
                (slice_id,),
            )
        ]
        events = [
            {
                "seq": row["seq"],
                "kind": row["kind"],
                "at": row["at"],
                **json.loads(row["detail_json"] or "{}"),
            }
            for row in self._conn.execute(
                "SELECT * FROM slice_events WHERE slice_id = ? ORDER BY seq", (slice_id,)
            )
        ]
        return SliceRecord(
            slice_id=head["slice_id"],
            repository_id=head["repository_id"],
            repo_path=head["repo_path"],
            task=head["task"],
            policy=WalkPolicy.from_dict(json.loads(head["policy_json"] or "{}")),
            seeds=json.loads(head["seeds_json"] or "[]"),
            members=members,
            edges=edges,
            external_dependencies=json.loads(head["externals_json"] or "[]"),
            created_at=head["created_at"],
            updated_at=head["updated_at"],
            revision=head["revision"],
            events=events,
            member_cap_hit=bool(head["member_cap_hit"]),
        )

    @staticmethod
    def _member_from_row(row: sqlite3.Row) -> SliceMember:
        return SliceMember(
            node_id=row["node_id"],
            node_type=row["node_type"],
            layer=row["layer"],
            file_path=row["file_path"],
            distance=row["distance"],
            is_seed=bool(row["is_seed"]),
            name=row["name"],
            kind=row["kind"],
            signature=row["signature"],
            docstring=row["docstring"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            language=row["language"],
            is_test=bool(row["is_test"]),
            pagerank=row["pagerank"],
            reference_count=row["reference_count"],
            max_confidence=row["max_confidence"],
            query_hits=row["query_hits"],
            frontier_down=bool(row["frontier_down"]),
            frontier_up=bool(row["frontier_up"]),
            reasons=json.loads(row["reasons_json"] or "[]"),
            edge_types=set(json.loads(row["edge_types_json"] or "[]")),
            seed_score=row["seed_score"],
            score=row["score"],
            rank=row["rank"],
            added_revision=row["added_revision"],
        )

    def recent_ids(self, limit: int = 10) -> list[str]:
        """Most recently touched slice ids — recovery evidence for a bad id."""
        return [
            row["slice_id"]
            for row in self._conn.execute(
                "SELECT slice_id FROM slices ORDER BY updated_at DESC LIMIT ?", (limit,)
            )
        ]

    def list_slices(
        self, repository_id: str | None = None, limit: int = 25
    ) -> list[dict[str, Any]]:
        if repository_id:
            rows = self._conn.execute(
                "SELECT slice_id, task, revision, updated_at FROM slices "
                "WHERE repository_id = ? ORDER BY updated_at DESC LIMIT ?",
                (repository_id, limit),
            )
        else:
            rows = self._conn.execute(
                "SELECT slice_id, task, revision, updated_at FROM slices "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
        return [dict(row) for row in rows]

    def delete(self, slice_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute("DELETE FROM slices WHERE slice_id = ?", (slice_id,))
            self._conn.execute("DELETE FROM slice_members WHERE slice_id = ?", (slice_id,))
            self._conn.execute("DELETE FROM slice_edges WHERE slice_id = ?", (slice_id,))
            self._conn.execute("DELETE FROM slice_events WHERE slice_id = ?", (slice_id,))
        return cur.rowcount > 0
