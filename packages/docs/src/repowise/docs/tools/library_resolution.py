"""Library ID resolution and fuzzy matching for doc-search tools."""

from __future__ import annotations

from typing import Any

from repowise.docs.db import get_library
from repowise.docs.db import list_libraries as db_list_libraries
from repowise.docs.library.models import LibraryState, LibraryStatus

from .common import ToolError, get_arg


class AmbiguousLibraryError(ToolError):
    """Raised when case-insensitive library lookup finds multiple matches.

    The registry is not supposed to contain colliding-case ``library_id``
    values. If it ever does, callers need to disambiguate by passing the
    exact-case ID.
    """


async def _resolve_library(
    library_id: str,
    *,
    get_library_fn=get_library,
    list_libraries_fn=db_list_libraries,
):
    """Resolve ``library_id`` to a library, preserving registry casing."""
    library = await get_library_fn(library_id)
    if library is not None:
        return library, library.library_id

    target = library_id.strip().lower()
    if not target:
        return None, library_id

    candidates = await list_libraries_fn()
    matches = [lib for lib in candidates if lib.library_id.lower() == target]
    if not matches:
        return None, library_id
    if len(matches) > 1:
        raise AmbiguousLibraryError(
            "Ambiguous library_id "
            f"'{library_id}' - registry contains multiple entries whose "
            "IDs differ only in case: " + ", ".join(repr(m.library_id) for m in matches)
        )
    return matches[0], matches[0].library_id


def _normalize_lib_name(name: str) -> str:
    """Normalize library name for matching."""
    return name.lower().strip().replace("-", " ").replace("/", " ").replace("_", " ")


def _compute_match_score(query: str, lib: LibraryState) -> float:
    """Compute fuzzy match score between query and library."""
    query = query.lower().strip()
    query_norm = _normalize_lib_name(query)
    base_score = 0.0

    if query == lib.library_id.lower():
        base_score = 1.0
    elif query == lib.name.lower() or query_norm == _normalize_lib_name(lib.name):
        base_score = 0.98
    elif query == lib.repo.lower():
        base_score = 0.95
    elif lib.source_subpath and query == lib.source_subpath.lower():
        base_score = 0.88
    else:
        targets = [
            (lib.library_id.lower(), 0.9),
            (lib.name.lower(), 0.85),
            (lib.repo.lower(), 0.8),
            (lib.source_subpath.lower(), 0.72) if lib.source_subpath else ("", 0.0),
            (lib.description.lower() if lib.description else "", 0.5),
        ]
        for target, max_score in targets:
            if not target:
                continue
            if query in target:
                ratio = len(query) / len(target)
                score = max_score * (0.5 + 0.5 * ratio)
                base_score = max(base_score, score)

            query_words = set(query.replace("/", " ").replace("-", " ").split())
            target_words = set(target.replace("/", " ").replace("-", " ").split())
            if query_words & target_words:
                overlap = len(query_words & target_words) / len(query_words)
                score = max_score * 0.5 * overlap
                base_score = max(base_score, score)

    if lib.status == LibraryStatus.READY:
        base_score *= 1.15
    elif lib.status in (LibraryStatus.PENDING, LibraryStatus.ERROR):
        base_score *= 0.85

    return min(base_score, 1.0)


async def handle_resolve_library_id(
    arguments: dict[str, Any],
    *,
    list_libraries_fn=db_list_libraries,
) -> dict[str, Any]:
    """Resolve a library name to its full library ID."""
    library_name = arguments.get("libraryName", "").strip().lower()
    if not library_name:
        library_name = (
            str(get_arg(arguments, "library_name", aliases=("libraryName",), default=""))
            .strip()
            .lower()
        )
    if not library_name:
        raise ToolError("library_name is required")

    libraries = await list_libraries_fn()
    if not libraries:
        return {"library_id": None, "error": "No libraries indexed yet", "suggestions": []}

    matches: list[tuple[float, LibraryState]] = []
    for lib in libraries:
        score = _compute_match_score(library_name, lib)
        if score > 0:
            matches.append((score, lib))

    matches.sort(key=lambda x: x[0], reverse=True)
    if not matches:
        suggestions = [{"library_id": lib.library_id, "name": lib.name} for lib in libraries[:5]]
        return {
            "library_id": None,
            "error": f"No library matching '{library_name}' found",
            "suggestions": suggestions,
        }

    best_score, best_lib = matches[0]
    result = {
        "library_id": best_lib.library_id,
        "name": best_lib.name,
        "description": best_lib.description,
        "library_status": best_lib.status.value,
        "confidence": round(best_score, 2),
    }
    if best_score < 0.9 and len(matches) > 1:
        result["alternatives"] = [
            {"library_id": lib.library_id, "name": lib.name, "confidence": round(score, 2)}
            for score, lib in matches[1:4]
        ]
    return result
