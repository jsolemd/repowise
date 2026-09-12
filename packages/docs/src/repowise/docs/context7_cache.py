"""Context7 comparison logging for doc-search optimization.

This module provides utilities to capture and compare outputs from:
- Our doc-search MCP server
- Context7 (external documentation search service)

The logged data helps identify improvements for LLM consumability.

Usage:
    from repowise.docs.context7_cache import log_comparison, get_comparison_log

    # Log a comparison (async)
    await log_comparison(
        query="@observe decorator Python",
        library_id="/langfuse/langfuse-docs",
        docsearch_output={"results": [...]},
        context7_output="### Title\nContent...",
    )

    # Read comparison log
    comparisons = get_comparison_log()
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repowise.docs.config import get_settings

logger = logging.getLogger(__name__)


def get_comparison_dir() -> Path:
    """Get the directory for comparison logs."""
    settings = get_settings()
    return settings.cache_root / "context7-comparison"


def get_comparison_log_path() -> Path:
    """Get the path to the comparison JSONL file."""
    return get_comparison_dir() / "comparison.jsonl"


async def log_comparison(
    query: str,
    library_id: str,
    docsearch_output: dict[str, Any],
    context7_output: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Log a comparison between doc-search and context7 outputs.

    Args:
        query: The search query used
        library_id: Library being searched
        docsearch_output: Our doc-search response (JSON or markdown content)
        context7_output: Context7's response (markdown string), if available
        notes: Optional notes about the comparison

    Returns:
        The logged entry dict
    """
    timestamp = datetime.now(UTC).isoformat()

    log_entry = {
        "timestamp": timestamp,
        "query": query,
        "library_id": library_id,
        "docsearch": docsearch_output,
        "context7": context7_output,
        "notes": notes,
    }

    # Ensure directory exists
    log_dir = get_comparison_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Append to JSONL file
    log_file = get_comparison_log_path()
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        logger.debug(f"Logged comparison for query: {query}")
    except Exception as e:
        logger.warning(f"Failed to log comparison: {e}")

    return log_entry


def get_comparison_log(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Read the comparison log.

    Args:
        limit: Maximum number of entries to return (most recent first).
               None returns all entries.

    Returns:
        List of comparison entries (most recent first)
    """
    log_file = get_comparison_log_path()

    if not log_file.exists():
        return []

    entries = []
    try:
        with open(log_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    except Exception as e:
        logger.warning(f"Failed to read comparison log: {e}")
        return []

    # Reverse for most recent first
    entries.reverse()

    if limit is not None:
        entries = entries[:limit]

    return entries


def clear_comparison_log() -> bool:
    """
    Clear the comparison log file.

    Returns:
        True if cleared successfully, False otherwise
    """
    log_file = get_comparison_log_path()

    if not log_file.exists():
        return True

    try:
        log_file.unlink()
        logger.info("Cleared comparison log")
        return True
    except Exception as e:
        logger.warning(f"Failed to clear comparison log: {e}")
        return False


def summarize_comparisons() -> dict[str, Any]:
    """
    Generate a summary of logged comparisons.

    Returns:
        Summary dict with statistics and patterns
    """
    entries = get_comparison_log()

    if not entries:
        return {
            "total_comparisons": 0,
            "message": "No comparisons logged yet",
        }

    # Count entries with context7 output
    with_context7 = sum(1 for e in entries if e.get("context7"))

    # Unique libraries
    libraries = set(e.get("library_id", "") for e in entries)

    # Unique queries
    queries = [e.get("query", "") for e in entries]

    return {
        "total_comparisons": len(entries),
        "with_context7": with_context7,
        "docsearch_only": len(entries) - with_context7,
        "unique_libraries": list(libraries),
        "unique_queries": len(set(queries)),
        "recent_queries": queries[:10],
    }
