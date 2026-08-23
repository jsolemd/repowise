"""MCP Tools: get_reference_sites and preview_symbol_rename.

Both read the reference-site store, which records one row per *occurrence* —
file, range, kind, resolution confidence — rather than one row per relation.
That is the difference that matters here: ``get_dependents`` can tell you that
``widget.ts`` uses ``computeTotal``; only an occurrence store can tell you that
it does so four times, at which columns, and which of those four a rewriter
could touch unattended.

Every response carries a coverage block. An empty ``sites`` list is ambiguous
on its own — it could mean the symbol is unreferenced, or that the references
live in a language nothing here can read — so the two are separated by
``status`` and by the per-language rows, never left to the caller to guess.

Both tools register opt-in (``default=False``): they answer a narrow question
and do not belong on the curated default surface.

Registration could not be deferred, though it was tried.
``tests/unit/server/mcp/test_startup_budget.py::test_tool_modules_map_covers_
every_tool_module_on_disk`` asserts that every ``tool_*.py`` in this package
appears in ``_TOOL_MODULES``, precisely because a module that is not in the map
is silently never registered. So a tool module cannot land unregistered, and
registering it moves the published tool counts in the same change — see
``REGISTRATION.md`` for which counts move and which do not. ``REFSITE_TOOLS``
below is the registration order.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy.exc import OperationalError

from repowise.core.persistence.database import get_session
from repowise.core.refsites.coverage import CoverageStatus, declared_coverage
from repowise.core.refsites.rename import preview_rename, symbol_name
from repowise.core.refsites.store import SqlReferenceSiteStore
from repowise.core.refsites.taxonomy import (
    CONFIDENCE_CERTAIN,
    ReferenceKind,
)
from repowise.core.registry import mcp_tool_registry
from repowise.server.mcp_server._helpers import (
    _get_repo,
    _resolve_repo_context,
    _unsupported_repo_all,
)

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 500

_CONFIDENCE_SCALE = (
    "0.0-1.0: the probability that renaming the resolved target must edit this site. "
    "Not a relevance score; do not compare it against search ranking."
)

_NOT_INDEXED = (
    "No reference sites are stored for this repository. The store is populated by a "
    "reference-site extraction pass, which is separate from graph indexing — an empty "
    "result here does not mean the symbol is unreferenced."
)


def _serialize_site(site: Any) -> dict[str, Any]:
    return {
        "file": site.file_path,
        "language": site.language,
        "name": site.name,
        "kind": str(site.kind),
        "start_line": site.start_line,
        "end_line": site.end_line,
        "start_col": site.start_col,
        "end_col": site.end_col,
        "range_exact": site.range_exact,
        "confidence": site.confidence,
        "origin": str(site.origin),
        "tier": site.tier,
        "occurrence_index": site.occurrence_index,
        "enclosing_symbol": site.enclosing_symbol_id,
    }


def _serialize_coverage(records: list[Any]) -> dict[str, Any]:
    """The coverage block every response carries."""
    observed = [
        {
            "language": record.language,
            "status": str(record.status),
            "files_seen": record.files_seen,
            "files_with_sites": record.files_extracted,
            "sites": record.sites,
            "reason": record.reason,
            "limitations": list(record.limitations),
        }
        for record in records
    ]
    uncovered = [row["language"] for row in observed if row["status"] == str(CoverageStatus.NONE)]
    partial = [
        row["language"]
        for row in observed
        if row["status"] in (str(CoverageStatus.PARTIAL), str(CoverageStatus.UNGATED))
    ]
    return {
        "declared": [
            {
                "language": claim.language,
                "status": str(claim.status),
                "kinds": list(claim.kinds),
                "limitations": list(claim.limitations),
            }
            for claim in declared_coverage()
        ],
        "observed": observed,
        "uncovered_languages_present": uncovered,
        "partial_languages_present": partial,
        "complete": not uncovered and not partial,
    }


def _coverage_status(records: list[Any], sites: list[Any]) -> str:
    """``found``/``not_found`` only when nothing about the repo is unreadable."""
    limited = any(
        record.status is not CoverageStatus.FULL and record.files_seen > 0 for record in records
    )
    if limited:
        return "coverage_limited"
    return "found" if sites else "not_found"


async def _resolve_target(
    store: SqlReferenceSiteStore, repository_id: str, target: str
) -> tuple[str | None, str, list[str]]:
    """Resolve *target* to a symbol id. Returns ``(symbol_id, matched_by, candidates)``.

    A target containing ``::`` is taken as a symbol id and used as given, even
    if nothing is stored for it: a preview for an unknown id is a legitimate
    empty answer, whereas guessing would silently retarget the query.
    """
    if "::" in target:
        return target, "symbol_id", []

    definitions = [
        site
        for site in await store.sites_named(repository_id, target)
        if site.kind is ReferenceKind.DEFINITION and site.target_symbol_id
    ]
    unique = sorted({site.target_symbol_id for site in definitions if site.target_symbol_id})
    if len(unique) == 1:
        return unique[0], "name", []
    return None, "name", unique


@mcp_tool_registry.register(default=False)
async def get_reference_sites(
    target: str,
    repo: str | None = None,
    kinds: list[str] | None = None,
    min_confidence: float = 0.0,
    offset: int = 0,
    limit: int = _DEFAULT_LIMIT,
) -> dict[str, Any]:
    """List every recorded occurrence of a symbol, with position and confidence.

    Unlike a dependency query, this returns one entry per *occurrence*: four
    calls in one file are four entries with distinct columns, not one edge.
    Occurrences that could not be bound to a definition are included at low
    confidence rather than dropped, and the response always states which
    languages in the repository produce no sites at all.

    Args:
        target: Symbol id (``path::Name``) or an unambiguous bare symbol name.
        repo: Repository path, name, or ID.
        kinds: Restrict to these reference kinds (call, definition, import,
            import_binding, receiver, reexport, type_ref, heritage, reference,
            identifier, string_ref). Omit for all.
        min_confidence: Drop sites below this rubric confidence (0.0-1.0).
        offset: Zero-based result offset.
        limit: Page size (1-500, default 50).
    """
    if repo == "all":
        return _unsupported_repo_all("get_reference_sites")

    effective_offset = max(0, offset)
    effective_limit = max(1, min(limit, _MAX_LIMIT))
    wanted = {kind.strip().lower() for kind in kinds} if kinds else None

    ctx = await _resolve_repo_context(repo)
    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
        store = SqlReferenceSiteStore(session)
        try:
            # Asked before resolution, and for the same reason
            # ``preview_symbol_rename`` asks it first: on a repository no
            # extraction pass has touched, every bare name fails to resolve,
            # and answering "No definition named 'x' is recorded" would be
            # precisely the empty-success this tool exists to refuse. A symbol
            # id short-circuits resolution and would have reached the check
            # anyway, which is why the gap only ever showed on bare names.
            stored_any = await store.count(repository.id)
            if not stored_any:
                return {
                    "status": "not_indexed",
                    "target": {"query": target},
                    "sites": [],
                    "total": 0,
                    "explanation": _NOT_INDEXED,
                }
            symbol_id, matched_by, candidates = await _resolve_target(store, repository.id, target)
            if symbol_id is None:
                return {
                    "status": "ambiguous" if candidates else "not_found",
                    "target": {"query": target, "matched_by": matched_by},
                    "candidates": candidates,
                    "sites": [],
                    "total": 0,
                    "explanation": (
                        f"'{target}' matches {len(candidates)} definitions; pass one "
                        "symbol id and retry."
                        if candidates
                        else f"No definition named '{target}' is recorded."
                    ),
                }

            bound = await store.sites_for_target(
                repository.id, symbol_id, min_confidence=min_confidence
            )
            # The definition site's recorded name is the text actually in the
            # file. Slicing the symbol id instead would search for the
            # ``~<sha8>`` discriminator the parse layer appends to colliding
            # symbols, and quietly return no unbound candidates at all.
            definition = next(
                (site for site in bound if site.kind is ReferenceKind.DEFINITION), None
            )
            unbound_name = (
                definition.name
                if definition is not None
                else (symbol_name(symbol_id) if matched_by == "symbol_id" else target)
            )
            unbound = await store.unbound_sites_named(
                repository.id, unbound_name, min_confidence=min_confidence
            )
            coverage = await store.coverage(repository.id)
        except OperationalError:
            return {
                "status": "not_indexed",
                "target": {"query": target},
                "sites": [],
                "total": 0,
                "explanation": _NOT_INDEXED,
            }

    sites = [*bound, *unbound]
    if wanted is not None:
        sites = [site for site in sites if str(site.kind) in wanted]
    sites.sort(key=lambda site: (-site.confidence, site.file_path, site.start_line, site.start_col))

    page = sites[effective_offset : effective_offset + effective_limit]
    returned = len(page)
    has_more = effective_offset + returned < len(sites)

    return {
        "status": _coverage_status(coverage, sites),
        "target": {
            "query": target,
            "symbol_id": symbol_id,
            "matched_by": matched_by,
        },
        "sites": [_serialize_site(site) for site in page],
        "total": len(sites),
        "counts_by_kind": dict(Counter(str(site.kind) for site in sites)),
        "counts_by_language": dict(Counter(site.language for site in sites)),
        "unbound": sum(1 for site in sites if site.target_symbol_id is None),
        "pagination": {
            "offset": effective_offset,
            "limit": effective_limit,
            "returned": returned,
            "has_more": has_more,
            "next_offset": effective_offset + returned if has_more else None,
        },
        "coverage": _serialize_coverage(coverage),
        "confidence_scale": _CONFIDENCE_SCALE,
    }


@mcp_tool_registry.register(default=False)
async def preview_symbol_rename(
    symbol: str,
    new_name: str | None = None,
    repo: str | None = None,
    min_confidence: float = 0.0,
    limit: int = _MAX_LIMIT,
) -> dict[str, Any]:
    """Show every site a rename would touch. Reports only — changes nothing.

    Sites bound to the symbol come first at their resolution confidence, then
    sites that merely spell the same name and bind to nothing: an ambiguous
    global, an unresolved call, a name inside a string literal. Those are leads
    for a human, not edits, and ``mechanically_safe`` marks the difference.

    ``caveats`` is the part to read before acting. It names any language in the
    repository that produces no reference sites, any site whose column range
    could not be verified, and any collision with the proposed new name.

    Args:
        symbol: Symbol id (``path::Name``) or an unambiguous bare symbol name.
        new_name: Proposed name. Only used to report collisions; nothing is written.
        repo: Repository path, name, or ID.
        min_confidence: Drop sites below this rubric confidence (0.0-1.0).
        limit: Maximum sites to return (1-500).
    """
    if repo == "all":
        return _unsupported_repo_all("preview_symbol_rename")

    effective_limit = max(1, min(limit, _MAX_LIMIT))
    ctx = await _resolve_repo_context(repo)

    async with get_session(ctx.session_factory) as session:
        repository = await _get_repo(session)
        store = SqlReferenceSiteStore(session)
        try:
            stored_any = await store.count(repository.id)
            if not stored_any:
                return {
                    "status": "not_indexed",
                    "symbol": symbol,
                    "sites": [],
                    "applies_changes": False,
                    "explanation": _NOT_INDEXED,
                }
            symbol_id, matched_by, candidates = await _resolve_target(store, repository.id, symbol)
            if symbol_id is None:
                return {
                    "status": "ambiguous" if candidates else "not_found",
                    "symbol": symbol,
                    "candidates": candidates,
                    "sites": [],
                    "applies_changes": False,
                    "explanation": (
                        f"'{symbol}' matches {len(candidates)} definitions; pass one "
                        "symbol id and retry."
                        if candidates
                        else f"No definition named '{symbol}' is recorded."
                    ),
                }
            preview = await preview_rename(
                store,
                repository.id,
                symbol_id,
                new_name=new_name,
                min_confidence=min_confidence,
            )
        except OperationalError:
            return {
                "status": "not_indexed",
                "symbol": symbol,
                "sites": [],
                "applies_changes": False,
                "explanation": _NOT_INDEXED,
            }

    page = preview.sites[:effective_limit]
    safe = sum(1 for site in preview.sites if site.mechanically_safe)

    return {
        "status": _coverage_status(list(preview.coverage), list(preview.sites)),
        "symbol": {"query": symbol, "symbol_id": symbol_id, "matched_by": matched_by},
        "old_name": preview.old_name,
        "new_name": preview.new_name,
        "sites": [
            {**_serialize_site(site), "mechanically_safe": site.mechanically_safe} for site in page
        ],
        "files_touched": list(preview.files_touched),
        "summary": {
            "total": len(preview.sites),
            "returned": len(page),
            "mechanically_safe": safe,
            "needs_review": len(preview.sites) - safe,
            "safe_threshold": CONFIDENCE_CERTAIN,
        },
        "caveats": list(preview.caveats),
        "coverage": _serialize_coverage(list(preview.coverage)),
        "confidence_scale": _CONFIDENCE_SCALE,
        # Stated in the payload, not only in the docstring: this tool has no
        # apply path, and a caller must not infer one from a clean preview.
        "applies_changes": False,
    }


#: The tools this module provides, in the order REGISTRATION.md registers them.
#: Both are opt-in (``default=False``) — they answer a narrow question and do
#: not belong on the curated default surface.
REFSITE_TOOLS = (get_reference_sites, preview_symbol_rename)

__all__ = ["REFSITE_TOOLS", "get_reference_sites", "preview_symbol_rename"]
