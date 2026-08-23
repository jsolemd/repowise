"""Value types for a task slice.

A *slice* is the subgraph one task needs: a set of ranked members, the edges
that put them there, the policy that produced them, and enough bookkeeping to
extend it later without walking from scratch.

Three things here are load-bearing and are not incidental fields:

* ``SliceMember.distance`` and the two ``frontier_*`` flags are what make
  extension cheap. A member discovered at the walk's depth limit never had its
  own edges expanded, so it is a *frontier*; extension resumes from exactly
  those nodes at exactly their stored distance. Drop the flags and the only
  correct extension is a full re-walk.
* ``SliceMember.reasons`` is the answer to "why is this in my slice". Ranking
  without it produces an ordered list nobody can audit.
* ``SliceMember.layer`` records which of the graph's two layers the member
  came from. The graph joins files to files by imports and symbols to symbols
  by calls, and nothing points from a symbol back to a file
  (:mod:`repowise.core.ingestion.models` spells this out); a member list that
  forgets which layer a node belongs to cannot be walked further.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

#: Fidelity levels a member can be rendered at, cheapest first.
VIEWS: tuple[str, ...] = ("card", "skeleton", "full")

#: Direction labels used on :class:`SliceEdge` and in member reasons.
DOWNSTREAM = "downstream"
UPSTREAM = "upstream"


@dataclass(frozen=True)
class WalkPolicy:
    """How far and in which directions a slice walk runs.

    The asymmetric default is the point. A task needs *what the entry point
    reaches* in some depth — that is the code it will read and change — and
    only a shallow ring of *who reaches the entry point*, which is context for
    not breaking callers. Making both depths equal produces a slice dominated
    by inbound noise on any central node.
    """

    downstream_depth: int = 2
    upstream_depth: int = 1
    include_tests: bool = False
    #: How many of a file seed's own symbols also seed the symbol layer.
    #: Bounded because a file can define hundreds and the walk would then be
    #: seeded by its whole contents rather than by the task.
    seed_symbol_fanout: int = 6
    #: Hard ceiling on **discovered** members. Reaching it is disclosed on the
    #: result (``member_cap_hit``), never silent. Seeds are outside the cap on
    #: purpose: dropping one would return a slice that is not rooted where the
    #: caller asked, which is a worse answer than a slice one member over.
    max_members: int = 400
    #: Edges below this confidence are not traversed. 0.0 keeps everything the
    #: resolver emitted, including ``global_unique`` guesses at 0.50.
    min_edge_confidence: float = 0.0

    MAX_DEPTH = 6

    def clamped(self) -> WalkPolicy:
        """Return a copy with every knob inside its supported range."""
        return WalkPolicy(
            downstream_depth=max(0, min(self.downstream_depth, self.MAX_DEPTH)),
            upstream_depth=max(0, min(self.upstream_depth, self.MAX_DEPTH)),
            include_tests=self.include_tests,
            seed_symbol_fanout=max(0, min(self.seed_symbol_fanout, 50)),
            max_members=max(1, min(self.max_members, 5000)),
            min_edge_confidence=max(0.0, min(self.min_edge_confidence, 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "downstream_depth": self.downstream_depth,
            "upstream_depth": self.upstream_depth,
            "include_tests": self.include_tests,
            "seed_symbol_fanout": self.seed_symbol_fanout,
            "max_members": self.max_members,
            "min_edge_confidence": self.min_edge_confidence,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> WalkPolicy:
        raw = raw or {}
        base = cls()
        return cls(
            downstream_depth=int(raw.get("downstream_depth", base.downstream_depth)),
            upstream_depth=int(raw.get("upstream_depth", base.upstream_depth)),
            include_tests=bool(raw.get("include_tests", base.include_tests)),
            seed_symbol_fanout=int(raw.get("seed_symbol_fanout", base.seed_symbol_fanout)),
            max_members=int(raw.get("max_members", base.max_members)),
            min_edge_confidence=float(raw.get("min_edge_confidence", base.min_edge_confidence)),
        )


@dataclass(frozen=True)
class EntryCandidate:
    """One candidate entry point, with the evidence that nominated it."""

    node_id: str
    node_type: str
    file_path: str
    name: str | None
    kind: str | None
    language: str
    score: float
    matched_terms: tuple[str, ...]
    reasons: tuple[str, ...]
    is_entry_point: bool = False
    pagerank: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node_id,
            "node_type": self.node_type,
            "file": self.file_path,
            "name": self.name,
            "kind": self.kind,
            "score": round(self.score, 4),
            "matched_terms": list(self.matched_terms),
            "why": list(self.reasons),
            "declared_entry_point": self.is_entry_point,
        }


@dataclass
class SliceMember:
    """One node in the slice, with why it is here and how it ranked."""

    node_id: str
    node_type: str
    layer: str
    file_path: str
    distance: int
    is_seed: bool = False
    name: str | None = None
    kind: str | None = None
    signature: str | None = None
    docstring: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    language: str = ""
    is_test: bool = False
    pagerank: float = 0.0
    #: Distinct persisted graph relations from this node into the frontier that
    #: discovered it. GraphEdge is unique per (source, target, edge_type), so
    #: this counts relations, never call sites.
    reference_count: int = 0
    max_confidence: float = 0.0
    query_hits: int = 0
    frontier_down: bool = False
    frontier_up: bool = False
    reasons: list[str] = field(default_factory=list)
    edge_types: set[str] = field(default_factory=set)
    seed_score: float = 0.0
    score: float = 0.0
    rank: int = 0
    added_revision: int = 1

    def identity(self) -> dict[str, Any]:
        """The fields every view carries, whatever its fidelity."""
        return {
            "node": self.node_id,
            "node_type": self.node_type,
            "file": self.file_path,
            "name": self.name,
            "kind": self.kind,
            "lines": ([self.start_line, self.end_line] if self.start_line else None),
            "distance": self.distance,
            "is_seed": self.is_seed,
            "rank": self.rank,
            "score": round(self.score, 4),
            "added_in_revision": self.added_revision,
        }


@dataclass(frozen=True)
class SliceEdge:
    """One traversed relation, kept so the slice is a subgraph and not a bag."""

    source: str
    target: str
    edge_type: str
    confidence: float
    direction: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type,
            "confidence": round(self.confidence, 3),
            "direction": self.direction,
        }


@dataclass
class SliceRecord:
    """A persisted slice: identity, provenance, members, edges, history."""

    slice_id: str
    repository_id: str
    repo_path: str
    task: str
    policy: WalkPolicy
    seeds: list[str] = field(default_factory=list)
    members: list[SliceMember] = field(default_factory=list)
    edges: list[SliceEdge] = field(default_factory=list)
    external_dependencies: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    revision: int = 1
    events: list[dict[str, Any]] = field(default_factory=list)
    #: True when the walk stopped at ``policy.max_members`` rather than at its
    #: depth limits. Always reported; never inferred by the caller.
    member_cap_hit: bool = False

    def member(self, node_id: str) -> SliceMember | None:
        for m in self.members:
            if m.node_id == node_id:
                return m
        return None

    def frontier(self, direction: str) -> dict[str, int]:
        """``{node_id: distance}`` for members whose edges were never expanded."""
        attr = "frontier_down" if direction == DOWNSTREAM else "frontier_up"
        return {m.node_id: m.distance for m in self.members if getattr(m, attr)}

    def summary(self) -> dict[str, Any]:
        by_layer: dict[str, int] = {}
        for m in self.members:
            by_layer[m.layer] = by_layer.get(m.layer, 0) + 1
        return {
            "slice_id": self.slice_id,
            "task": self.task,
            "revision": self.revision,
            "seeds": list(self.seeds),
            "total_members": len(self.members),
            "members_by_layer": by_layer,
            "edges": len(self.edges),
            "external_dependencies": list(self.external_dependencies),
            "policy": self.policy.to_dict(),
            "member_cap_hit": self.member_cap_hit,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
