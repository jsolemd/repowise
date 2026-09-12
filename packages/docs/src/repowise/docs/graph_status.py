"""Retired docs-to-Neo4j status contract.

The metadata relay had no production reader.  Keep its operator-facing shape so
existing dashboards do not break, but make retirement explicit rather than
reporting an empty or healthy graph.
"""

from __future__ import annotations


def get_docs_graph_sync_status() -> dict[str, object]:
    """Return the stable compatibility payload for the removed graph relay."""
    return {
        "enabled": False,
        "state": "disabled",
        "last_action": None,
        "last_library_id": None,
        "last_file_count": None,
        "last_attempt_at": None,
        "last_success_at": None,
        "last_error": None,
        "relay_running": False,
        "reason": "docs_neo4j_metadata_sync_retired",
    }


__all__ = ["get_docs_graph_sync_status"]
