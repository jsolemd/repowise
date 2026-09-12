"""Library administration and document-read doc-search tool handlers."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from repowise.docs.bundles import export_bundle, import_bundle
from repowise.docs.db import delete_library as db_delete_library
from repowise.docs.db import get_library, upsert_library
from repowise.docs.db import list_libraries as db_list_libraries
from repowise.docs.db_libraries import count_libraries as db_count_libraries
from repowise.docs.formats import DEFAULT_INCLUDE_PATTERNS
from repowise.docs.indexer import delete_by_library as qdrant_delete_by_library
from repowise.docs.jobs import JobType, clear_freshness_cache, enqueue_job
from repowise.docs.library.git_manager import (
    delete_repo_cache,
    get_remote_head_sha,
    resolve_default_branch,
)
from repowise.docs.library.models import (
    LibraryConfig,
    LibrarySourceType,
    LibraryStatus,
)
from repowise.docs.library.source_runtime import get_library_file_content

from .common import CHARS_PER_TOKEN, ToolError, clamp_int, get_arg, require_str

logger = logging.getLogger(__name__)


async def handle_list_libraries(
    arguments: dict[str, Any],
    *,
    list_libraries_fn=db_list_libraries,
    count_libraries_fn=db_count_libraries,
) -> dict[str, Any]:
    """List all indexed documentation libraries (paginated)."""
    include_stats = get_arg(arguments, "include_stats", aliases=("includeStats",), default=True)
    status_filter = arguments.get("status")

    status = None
    if status_filter:
        try:
            status = LibraryStatus(status_filter.lower())
        except ValueError as e:
            raise ToolError(
                f"Invalid status: {status_filter}. Use: pending, indexing, ready, error"
            ) from e

    raw_offset = get_arg(arguments, "offset", default=0)
    raw_limit = get_arg(arguments, "limit", default=None)
    offset = clamp_int(raw_offset, 0, 10_000, 0)
    limit: int | None
    limit = None if raw_limit is None or raw_limit == "" else clamp_int(raw_limit, 1, 200, 50)

    name_filter_raw = arguments.get("filter")
    name_filter = name_filter_raw.strip().lower() if isinstance(name_filter_raw, str) else ""

    if name_filter:
        # Substring filter requires post-DB filtering — count + paginate on
        # the filtered set so `next_offset` stays correct under the filter.
        all_libraries = await list_libraries_fn(status=status, offset=0, limit=None)

        def _matches(lib) -> bool:
            haystack_parts = (
                lib.library_id,
                lib.name,
                lib.description or "",
                lib.repo or "",
            )
            haystack = " ".join(part for part in haystack_parts if part).lower()
            return name_filter in haystack

        filtered = [lib for lib in all_libraries if _matches(lib)]
        total = len(filtered)
        libraries = filtered[offset:] if limit is None else filtered[offset : offset + limit]
    else:
        libraries = await list_libraries_fn(status=status, offset=offset, limit=limit)
        total = await count_libraries_fn(status=status)
    returned = len(libraries)
    next_offset: int | None
    if limit is None:
        next_offset = None
    else:
        candidate_next = offset + returned
        next_offset = candidate_next if candidate_next < total else None
    results = []
    for lib in libraries:
        item = {
            "library_id": lib.library_id,
            "name": lib.name,
            "description": lib.description,
            "status": lib.status.value,
            "source_type": lib.source_type.value,
        }
        if include_stats:
            item.update(
                {
                    "repo": lib.repo,
                    "source_subpath": lib.source_subpath or "",
                    "branch": lib.branch,
                    "chunk_count": lib.chunk_count,
                    "file_count": lib.file_count,
                    "indexed_at": lib.indexed_at.isoformat() if lib.indexed_at else None,
                    "freshness_checked_at": (
                        lib.freshness_checked_at.isoformat() if lib.freshness_checked_at else None
                    ),
                    "next_freshness_check_at": (
                        lib.next_freshness_check_at.isoformat()
                        if lib.next_freshness_check_at
                        else None
                    ),
                    "current_sha": lib.current_sha[:8] if lib.current_sha else None,
                }
            )
            if lib.status == LibraryStatus.ERROR:
                item["error_message"] = lib.error_message
            elif lib.error_message:
                item["warning_message"] = lib.error_message
        results.append(item)

    payload: dict[str, Any] = {
        "libraries": results,
        "total": total,
        "returned": returned,
        "offset": offset,
        "limit": limit,
    }
    if name_filter:
        payload["filter"] = name_filter
    if next_offset is not None:
        payload["next_offset"] = next_offset
    return payload


def _extract_section(content: str, section_anchor: str) -> str | None:
    """Extract content under a specific section header."""
    section_anchor = section_anchor.lower().strip().lstrip("#")

    def slugify(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_]+", "-", text)
        return text.strip("-")

    def extract_mdx_anchor(title: str) -> str | None:
        match = re.search(r"\[#([\w-]+)\]\s*$", title)
        if match:
            return match.group(1).lower()
        return None

    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headers = list(header_pattern.finditer(content))
    target_idx = None
    target_level = None
    matched_explicit_anchor = False

    for idx, match in enumerate(headers):
        level = len(match.group(1))
        title = match.group(2).strip()

        mdx_anchor = extract_mdx_anchor(title)
        if mdx_anchor and mdx_anchor == section_anchor:
            target_idx = idx
            target_level = level
            matched_explicit_anchor = True
            break

        clean_title = re.sub(r"\s*\[#[\w-]+\]\s*$", "", title)
        anchor = slugify(clean_title)
        if anchor == section_anchor:
            target_idx = idx
            target_level = level
            break

    if target_idx is None:
        return None

    start_pos = headers[target_idx].start()
    end_pos = len(content)
    for idx in range(target_idx + 1, len(headers)):
        level = len(headers[idx].group(1))
        should_stop = level <= target_level if matched_explicit_anchor else True
        if should_stop:
            end_pos = headers[idx].start()
            break

    return content[start_pos:end_pos].strip()


async def handle_read_doc(
    arguments: dict[str, Any],
    *,
    get_library_fn=get_library,
    get_library_file_content_fn=get_library_file_content,
) -> dict[str, Any]:
    """Read full document content from cached repository."""
    library_id = require_str(arguments, "library_id", aliases=("libraryId",))
    file_path = require_str(arguments, "path")
    section = arguments.get("section", "").strip() if arguments.get("section") else None
    max_tokens = clamp_int(
        get_arg(arguments, "max_tokens", aliases=("maxTokens",), default=10000), 100, 50000, 10000
    )

    library = await get_library_fn(library_id)
    if not library:
        raise ToolError(f"Library not found: {library_id}")

    content = await get_library_file_content_fn(library, file_path)
    if section:
        content = _extract_section(content, section)
        if content is None:
            raise ToolError(
                f"Section '{section}' not found in {file_path}. Try without section parameter to see available headers."
            )

    max_chars = max_tokens * CHARS_PER_TOKEN
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]
        last_para = content.rfind("\n\n")
        if last_para > max_chars * 0.8:
            content = content[:last_para]
        content += "\n\n... [truncated]"

    result = {
        "content": content,
        "path": file_path,
        "library_id": library_id,
        "truncated": truncated,
        "estimated_tokens": len(content) // CHARS_PER_TOKEN,
    }
    if section:
        result["section"] = section
    return result


async def handle_update_library(
    arguments: dict[str, Any],
    *,
    get_library_fn=get_library,
    enqueue_job_fn=enqueue_job,
    clear_freshness_cache_fn=clear_freshness_cache,
) -> dict[str, Any]:
    """Trigger a library reindex."""
    library_id = require_str(arguments, "library_id", aliases=("libraryId",))
    force = bool(arguments.get("force", False))

    library = await get_library_fn(library_id)
    if not library:
        raise ToolError(f"Library not found: {library_id}")

    job_type = JobType.FORCE if force else JobType.INCREMENTAL
    result = await enqueue_job_fn(library_id, job_type, priority=5)
    clear_freshness_cache_fn(library_id)

    payload = {
        "disposition": result.disposition.value,
        "library_id": library_id,
        "library_name": library.name,
        "job": {
            "id": result.job.id,
            "type": result.job.job_type.value,
            "priority": result.job.priority,
        },
    }
    return payload


async def handle_add_library(
    arguments: dict[str, Any],
    *,
    get_library_fn=get_library,
    get_remote_head_sha_fn=get_remote_head_sha,
    resolve_default_branch_fn=resolve_default_branch,
    upsert_library_fn=upsert_library,
    enqueue_job_fn=enqueue_job,
    default_include_patterns=DEFAULT_INCLUDE_PATTERNS,
) -> dict[str, Any]:
    """Register a new GitHub documentation library for indexing."""
    repo = require_str(arguments, "repo")
    library_id = (
        str(get_arg(arguments, "library_id", aliases=("libraryId",), default="")).strip() or None
    )
    name = require_str(arguments, "name")
    description = arguments.get("description", "").strip() if arguments.get("description") else None
    source_subpath = (
        str(get_arg(arguments, "source_subpath", aliases=("sourceSubpath",), default="")).strip()
        or None
    )
    docs_path = (
        str(get_arg(arguments, "docs_path", aliases=("docsPath",), default="")).strip() or None
    )
    branch = arguments.get("branch", "main").strip()
    include_patterns = get_arg(arguments, "include_patterns", aliases=("includePatterns",))
    exclude_patterns = get_arg(arguments, "exclude_patterns", aliases=("excludePatterns",))

    if "/" not in repo or repo.count("/") != 1:
        raise ToolError(
            f"Invalid repo format: '{repo}'. Expected 'owner/repo' format (e.g., 'vercel/next.js')"
        )

    owner, repo_name = repo.split("/")
    if not owner or not repo_name:
        raise ToolError(f"Invalid repo format: '{repo}'. Both owner and repo name are required.")

    sha = await get_remote_head_sha_fn(repo, branch)
    if sha is None:
        default_branch = await resolve_default_branch_fn(repo)
        if default_branch and default_branch != branch:
            raise ToolError(
                f"Branch '{branch}' not found for repo '{repo}'. The repository's default branch is '{default_branch}'. Try again with branch='{default_branch}'."
            )
        if default_branch is None:
            raise ToolError(
                f"Could not verify repo '{repo}' — it may not exist or may be private. Check the repo name and try again."
            )
        raise ToolError(
            f"Branch '{branch}' not found for repo '{repo}'. Verify the branch name and try again."
        )

    config = LibraryConfig(
        source_type=LibrarySourceType.GIT,
        library_id=library_id,
        repo=repo,
        name=name,
        description=description,
        source_subpath=source_subpath or "",
        docs_path=docs_path or "",
        branch=branch,
        include_patterns=include_patterns or list(default_include_patterns),
        exclude_patterns=exclude_patterns or [],
    )
    library_id = config.library_id
    existing = await get_library_fn(library_id)
    if existing:
        return {
            "library_id": library_id,
            "disposition": "exists",
            "library_name": existing.name,
            "current_status": existing.status.value,
        }

    await upsert_library_fn(config)
    logger.info("Registered new library: %s (%s)", library_id, name)

    job_result = await enqueue_job_fn(library_id, JobType.FULL, priority=3)
    result = {
        "library_id": library_id,
        "disposition": "created",
        "library_name": name,
        "repo": repo,
        "branch": branch,
    }
    if config.source_subpath:
        result["source_subpath"] = config.source_subpath
    if docs_path:
        result["docs_path"] = docs_path
    result["job"] = {
        "id": job_result.job.id,
        "type": job_result.job.job_type.value,
        "priority": job_result.job.priority,
        "disposition": job_result.disposition.value,
    }
    return result


async def handle_export_bundle(arguments: dict[str, Any]) -> dict[str, Any]:
    """Export a library's indexed chunks as a compressed bundle."""
    library_id = require_str(arguments, "library_id", aliases=("libraryId",))
    try:
        return await export_bundle(library_id)
    except ValueError as e:
        raise ToolError(str(e)) from e


async def handle_import_bundle(arguments: dict[str, Any]) -> dict[str, Any]:
    """Import a previously exported documentation bundle."""
    bundle_path = require_str(arguments, "bundle_path", aliases=("bundlePath",))
    try:
        return await import_bundle(Path(bundle_path))
    except (FileNotFoundError, ValueError) as e:
        raise ToolError(str(e)) from e


async def handle_delete_library(
    arguments: dict[str, Any],
    *,
    get_library_fn=get_library,
    list_libraries_fn=db_list_libraries,
    qdrant_delete_by_library_fn=qdrant_delete_by_library,
    delete_repo_cache_fn=delete_repo_cache,
    db_delete_library_fn=db_delete_library,
) -> dict[str, Any]:
    """Permanently delete a library and all its data."""
    library_id = require_str(arguments, "library_id", aliases=("libraryId",))
    confirm = arguments.get("confirm", False)
    if not confirm:
        raise ToolError(
            "confirm must be set to true to delete a library. This is a destructive, irreversible operation."
        )

    library = await get_library_fn(library_id)
    if library is None:
        raise ToolError(f"Library not found: {library_id}")

    try:
        chunks_deleted = await qdrant_delete_by_library_fn(library_id)
    except Exception as e:
        raise ToolError(
            f"Failed to delete Qdrant chunks for {library_id}: {e}. The library record was NOT deleted — you can retry."
        ) from e

    siblings = await list_libraries_fn()
    repo_is_shared = any(
        lib.library_id != library_id and lib.repo == library.repo for lib in siblings
    )
    cache_deleted = False
    try:
        if library.source_type == LibrarySourceType.GIT and not repo_is_shared:
            cache_deleted = delete_repo_cache_fn(library.repo)
    except Exception as e:  # pragma: no cover - best effort
        logger.warning("Failed to delete git cache for %s: %s", library.repo, e)

    await db_delete_library_fn(library_id)
    logger.info(
        "Deleted library %s: %s chunks, cache=%s",
        library_id,
        chunks_deleted,
        "yes" if cache_deleted else "no",
    )

    return {
        "disposition": "deleted",
        "library_id": library_id,
        "library_name": library.name,
        "chunks_deleted": chunks_deleted,
        "cache_deleted": cache_deleted,
        "cache_retained_for_shared_repo": repo_is_shared,
    }
