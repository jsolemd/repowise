"""Typed failures for the task-slice service.

Every one of these exists so a broken dependency or a missing resource can
never reach a caller wearing the costume of an empty result. A slice id that
does not exist, a query that resolves to no entry point, a budget that cannot
hold even the top-ranked member — each is a distinct condition with a distinct
recovery, and each raises rather than returning ``members=[]``.

The MCP surface maps every one to a shaped response carrying ``status`` and
``error`` and *no* member list, so a caller can never mistake a failure for a
slice that happens to be empty.
"""

from __future__ import annotations

from typing import Any


class SliceError(Exception):
    """Base class for every task-slice failure."""

    #: Stable machine-readable discriminator, mirrored into the MCP response's
    #: ``status`` field. Subclasses override.
    status = "slice_error"

    def details(self) -> dict[str, Any]:
        """Structured recovery evidence carried alongside the message."""
        return {}


class SliceNotFoundError(SliceError):
    """No stored slice carries this id.

    Distinct from "a slice with no members": that is a legal (if unhelpful)
    build result and is returned as a success with ``total_members == 0``,
    while this means the id names nothing at all.
    """

    status = "slice_not_found"

    def __init__(self, slice_id: str, *, known_ids: list[str] | None = None) -> None:
        super().__init__(
            f"No stored slice with id {slice_id!r}. Slice ids are minted by "
            f"build_task_slice and look like 'sl_<12 hex>'."
        )
        self.slice_id = slice_id
        self.known_ids = known_ids or []

    def details(self) -> dict[str, Any]:
        return {"slice_id": self.slice_id, "recent_slice_ids": self.known_ids[:10]}


class EntryPointsUnresolvedError(SliceError):
    """Nothing in the index answered the query or the requested entry points."""

    status = "entry_points_unresolved"

    def __init__(
        self,
        *,
        task: str | None,
        requested: list[str] | None = None,
        unresolved: list[str] | None = None,
        near_misses: list[dict[str, Any]] | None = None,
    ) -> None:
        if requested:
            msg = (
                f"None of the requested entry points resolved to a graph node: "
                f"{', '.join(unresolved or requested)}."
            )
        else:
            msg = (
                f"No entry point in the index matched the task {task!r}. "
                f"Name a file path or symbol id explicitly, or re-index."
            )
        super().__init__(msg)
        self.task = task
        self.requested = requested or []
        self.unresolved = unresolved or []
        self.near_misses = near_misses or []

    def details(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "requested_entry_points": self.requested,
            "unresolved": self.unresolved,
            "near_misses": self.near_misses[:10],
        }


class RepositoryNotIndexedError(SliceError):
    """The repository has no dependency graph to walk."""

    status = "not_indexed"

    def __init__(self, repository_id: str) -> None:
        super().__init__(
            f"Repository {repository_id!r} has no graph nodes. Run `repowise update` "
            f"before building a slice."
        )
        self.repository_id = repository_id

    def details(self) -> dict[str, Any]:
        return {"repository_id": self.repository_id}


class BudgetTooSmallError(SliceError):
    """The requested budget cannot hold the response envelope plus one member.

    Returning an empty member list here would be the exact failure this
    service is meant to make impossible: the caller would read "no members"
    as "nothing to read" rather than "ask for more room".
    """

    status = "budget_too_small"

    def __init__(
        self,
        *,
        budget_tokens: int,
        envelope_tokens: int,
        smallest_member_tokens: int | None,
        view: str,
        total_members: int,
    ) -> None:
        need = envelope_tokens + (smallest_member_tokens or 0)
        super().__init__(
            f"budget_tokens={budget_tokens} cannot hold the response envelope "
            f"({envelope_tokens} tokens) plus the highest-ranked member at "
            f"view={view!r}; at least {need} tokens are needed. The slice has "
            f"{total_members} member(s) and was not truncated to nothing — nothing fits."
        )
        self.budget_tokens = budget_tokens
        self.envelope_tokens = envelope_tokens
        self.smallest_member_tokens = smallest_member_tokens
        self.view = view
        self.total_members = total_members

    def details(self) -> dict[str, Any]:
        return {
            "budget_tokens": self.budget_tokens,
            "envelope_tokens": self.envelope_tokens,
            "top_member_tokens": self.smallest_member_tokens,
            "view": self.view,
            "total_members": self.total_members,
            "minimum_budget_tokens": self.envelope_tokens + (self.smallest_member_tokens or 0),
        }


class SliceStoreUnavailableError(SliceError):
    """The slice sidecar could not be opened or written.

    Deliberately fatal rather than degrading to an in-memory slice: a slice id
    handed back that cannot be re-fetched is a promise the service did not keep.
    """

    status = "store_unavailable"

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"Slice store at {path} is unavailable: {reason}")
        self.path = path
        self.reason = reason

    def details(self) -> dict[str, Any]:
        return {"store_path": self.path, "reason": self.reason}
