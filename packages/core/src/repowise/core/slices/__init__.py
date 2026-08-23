"""Task slices — the persistent, extendable subgraph one task needs.

A slice answers a question no single-shot tool does: *given this task, what
part of this codebase do I have to hold in my head?* It resolves entry points
from a task sentence or an explicit target, walks the SQL dependency graph
outward from them, ranks what it finds by why it is there, and stores the
result under an id that can be re-fetched and grown.

The pieces, in the order a build runs through them:

:mod:`~repowise.core.slices.entry_points`
    Query or explicit target → candidate entry points, scored. Reuses
    :mod:`repowise.core.entry_candidacy` for the one question that module
    owns: whether a reader can enter the system at a given path at all.
:mod:`~repowise.core.slices.walk`
    Breadth-first over ``graph_nodes`` / ``graph_edges``, one BFS per graph
    layer, downstream deep and upstream shallow. Records where it stopped.
:mod:`~repowise.core.slices.ranking`
    Five bounded signals, seeds first, deterministic ties. Rank is the order
    the budget spends itself in.
:mod:`~repowise.core.slices.store`
    A WAL SQLite sidecar at ``.repowise/slices/slices.db``. Slice ids are
    session state, not index state, so they outlive a re-index.
:mod:`~repowise.core.slices.views` / :mod:`~repowise.core.slices.budget`
    Three fidelities per member, and a budget that drops from the bottom of
    the ranking and says so every single time.

Every failure is typed (:mod:`~repowise.core.slices.errors`). An unknown slice
id, an unresolvable entry point and a budget too small for one member are
three different conditions with three different recoveries, and none of them
is ever spelled ``members: []``.
"""

from __future__ import annotations

from repowise.core.slices.budget import BudgetReport, DroppedMember, fit_members
from repowise.core.slices.entry_points import (
    nominate_entry_points,
    query_terms,
    resolve_entry_points,
)
from repowise.core.slices.errors import (
    BudgetTooSmallError,
    EntryPointsUnresolvedError,
    RepositoryNotIndexedError,
    SliceError,
    SliceNotFoundError,
    SliceStoreUnavailableError,
)
from repowise.core.slices.models import (
    VIEWS,
    EntryCandidate,
    SliceEdge,
    SliceMember,
    SliceRecord,
    WalkPolicy,
)
from repowise.core.slices.ranking import rank_members, ranking_contract
from repowise.core.slices.service import (
    build_slice,
    extend_slice,
    load_slice,
    render_slice,
)
from repowise.core.slices.store import SliceStore, default_store_path, new_slice_id

__all__ = [
    "VIEWS",
    "BudgetReport",
    "BudgetTooSmallError",
    "DroppedMember",
    "EntryCandidate",
    "EntryPointsUnresolvedError",
    "RepositoryNotIndexedError",
    "SliceEdge",
    "SliceError",
    "SliceMember",
    "SliceNotFoundError",
    "SliceRecord",
    "SliceStore",
    "SliceStoreUnavailableError",
    "WalkPolicy",
    "build_slice",
    "default_store_path",
    "extend_slice",
    "fit_members",
    "load_slice",
    "new_slice_id",
    "nominate_entry_points",
    "query_terms",
    "rank_members",
    "ranking_contract",
    "render_slice",
    "resolve_entry_points",
]
