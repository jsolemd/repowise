"""Regression tests for P6 (S12): library IDs are resolved case-insensitively.

Before the fix, ``search_docs(library_id="/greensock/gsap")`` returned
``Library not found`` because the registry stored
``/greensock/GSAP`` and SQL ``WHERE library_id = $1`` is case-sensitive.
Agents following the skill doc's ``resolve_library_id → search_docs``
flow then silently under-covered the result set.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

import pytest

from repowise.docs.library.models import LibraryStatus
from repowise.docs.tools.library_resolution import (
    AmbiguousLibraryError,
    _resolve_library,
)
from repowise.docs.tools.search import handle_query_docs_multi


class _Status(Enum):
    READY = "ready"
    PENDING = "pending"


@dataclass
class _FakeLibrary:
    library_id: str
    name: str = ""
    status: _Status | LibraryStatus = _Status.READY
    indexed_at: datetime | None = None
    error_message: str | None = None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe",
    ["/greensock/gsap", "/greensock/GSAP", "/greensock/Gsap"],
)
async def test_resolve_library_is_case_insensitive(probe: str) -> None:
    """Every reasonable casing of the probe must resolve to the one
    canonical ``/greensock/GSAP`` entry registered in the library set."""
    canonical = _FakeLibrary(library_id="/greensock/GSAP", status=_Status.READY)

    async def fake_get_library(library_id: str):
        return canonical if library_id == canonical.library_id else None

    async def fake_list_libraries():
        return [canonical]

    library, normalized = await _resolve_library(
        probe,
        get_library_fn=fake_get_library,
        list_libraries_fn=fake_list_libraries,
    )
    assert library is canonical
    assert normalized == canonical.library_id


@pytest.mark.asyncio
async def test_resolve_library_exact_match_skips_list_scan() -> None:
    """Exact-case matches must hit the fast path and not scan the whole
    library registry (the list call is expensive at production scale)."""
    canonical = _FakeLibrary(library_id="/foo/BAR", status=_Status.READY)
    list_calls = 0

    async def fake_get_library(library_id: str):
        return canonical if library_id == canonical.library_id else None

    async def fake_list_libraries():
        nonlocal list_calls
        list_calls += 1
        return [canonical]

    library, normalized = await _resolve_library(
        "/foo/BAR",
        get_library_fn=fake_get_library,
        list_libraries_fn=fake_list_libraries,
    )
    assert library is canonical
    assert normalized == "/foo/BAR"
    assert list_calls == 0, "Exact-case match must not trigger the list scan"


@pytest.mark.asyncio
async def test_resolve_library_returns_none_when_truly_missing() -> None:
    async def fake_get_library(library_id: str):
        return None

    async def fake_list_libraries():
        return []

    library, normalized = await _resolve_library(
        "/does-not-exist/anything",
        get_library_fn=fake_get_library,
        list_libraries_fn=fake_list_libraries,
    )
    assert library is None
    assert normalized == "/does-not-exist/anything"


@pytest.mark.asyncio
async def test_resolve_library_rejects_ambiguous_case() -> None:
    """Two library_ids that differ only by case must NOT be resolved
    silently — the caller needs to pick the exact-case ID themselves."""
    left = _FakeLibrary(library_id="/foo/Bar")
    right = _FakeLibrary(library_id="/foo/BAR")

    async def fake_get_library(library_id: str):
        if library_id == left.library_id:
            return left
        if library_id == right.library_id:
            return right
        return None

    async def fake_list_libraries():
        return [left, right]

    with pytest.raises(AmbiguousLibraryError):
        await _resolve_library(
            "/foo/bar",
            get_library_fn=fake_get_library,
            list_libraries_fn=fake_list_libraries,
        )


@pytest.mark.asyncio
async def test_query_docs_multi_reports_unmatched_and_skipped_structurally() -> None:
    """Round 8 Q6: ``search_docs_multi`` must split unmatched library IDs
    from skipped (resolved-but-unavailable) ones into two separate
    structured fields so agents can branch on them without parsing prose."""
    good = _FakeLibrary(
        library_id="/greensock/GSAP",
        name="GSAP",
        status=LibraryStatus.READY,
        indexed_at=datetime.now(UTC),
    )
    pending = _FakeLibrary(
        library_id="/pending/lib",
        name="Pending",
        status=LibraryStatus.PENDING,
    )

    async def fake_get_library(library_id: str):
        for lib in (good, pending):
            if lib.library_id == library_id:
                return lib
        return None

    async def fake_list_libraries():
        return [good, pending]

    async def fake_search_hybrid(**kwargs):
        return []

    response = await handle_query_docs_multi(
        {
            "library_ids": [
                "/foo/bar",  # unmatched (missing)
                "/bogus/missing",  # unmatched (missing)
                "/greensock/gsap",  # resolves via case-insensitive match
                "/pending/lib",  # resolved but pending
            ],
            "query": "tween",
            "output": "json",
        },
        get_library_fn=fake_get_library,
        list_libraries_fn=fake_list_libraries,
        search_hybrid_fn=fake_search_hybrid,
    )

    assert response.get("unmatched_library_ids") == ["/foo/bar", "/bogus/missing"]
    skipped = response.get("skipped_libraries") or []
    assert skipped == [{"library_id": "/pending/lib", "reason": "pending_indexing"}]
    assert response.get("normalized_library_ids") == {"/greensock/gsap": "/greensock/GSAP"}
