"""Display titles and MCP behaviour annotations for the registered tool surface.

``tools/list`` carries two fields beyond name/description that a client acts on:

* ``title`` — the human-readable name an agent UI shows instead of the wire
  name. Without it clients render ``get_blast_radius`` verbatim.
* ``annotations`` — the behaviour hints from the MCP spec
  (:class:`mcp.types.ToolAnnotations`). ``readOnlyHint`` is the one that
  changes client behaviour today: a host that gates side-effecting tools behind
  a confirmation prompt has, until now, had to treat every repowise tool as
  potentially side-effecting because the server said nothing.

Both live here rather than on each ``@mcp.tool(...)`` decorator for two
reasons. The values are near-uniform — 26 of 29 registered tools are pure
readers with identical hints — so per-decorator arguments would be 29 copies of
one fact. And the decorators sit in upstream tool modules, which the fork
re-merges on every release absorption; keeping the fork's metadata in one file
it owns outright keeps that merge surface at zero.

The hints are hints, not enforcement — the spec is explicit that a client must
not trust them from an untrusted server. They are accurate for this surface:
every reader answers from the index and the working tree without mutating
either.
"""

from __future__ import annotations

from typing import Any

from repowise.core.registry import ToolEntry

#: Every tool that changes state the caller can observe afterwards.
#:
#: ``manage_decision`` appends to (and retires entries in) the repository's
#: git-tracked decisions journal. ``reindex_repository`` queues an ``index_only``
#: job that rewrites the index. Neither destroys anything: a decision is
#: superseded rather than deleted, and a reindex is a rebuild of derived data,
#: so both are additive (``destructiveHint=False``). Neither is idempotent —
#: a second ``manage_decision(action="record")`` appends a second entry, and a
#: second confirmed reindex queues a second job.
WRITER_TOOLS = frozenset({"manage_decision", "reindex_repository"})

#: Tools that reach a model provider outside this machine, so their domain of
#: interaction is not closed and their answers are not reproducible. They read
#: rather than write, so ``readOnlyHint`` still holds. Neither is served in the
#: SoleMD deployment (``REPOWISE_TOOLS_NO_GENERATIVE=1`` purges both), but the
#: registry carries them and a stock server serves them.
OPEN_WORLD_TOOLS = frozenset({"get_answer", "generate_refactoring_code"})

#: Titles where the mechanical derivation below reads badly. Everything else
#: gets ``snake_case`` -> ``Sentence case``, which is already the right answer
#: for 26 of 29 ("search_codebase" -> "Search codebase").
TITLE_OVERRIDES = {
    "get_why": "Explain why",
    "list_repos": "List repositories",
    "manage_decision": "Manage decisions",
}


def title_for(name: str) -> str:
    """Human-readable display name for a tool's wire name."""
    override = TITLE_OVERRIDES.get(name)
    if override:
        return override
    words = name.replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def annotations_for(name: str) -> dict[str, Any]:
    """MCP behaviour hints for a tool's wire name.

    Returns a plain dict; FastMCP validates it into a
    :class:`~mcp.types.ToolAnnotations` when the tool is registered. Every key
    is set explicitly, including the ones whose spec default matches, because
    an omitted hint and a false hint are different answers to a client that
    checks for the field.
    """
    if name in WRITER_TOOLS:
        return {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        }
    open_world = name in OPEN_WORLD_TOOLS
    return {
        "readOnlyHint": True,
        "destructiveHint": False,
        # A reader over a fixed index gives the same answer twice; a model call
        # does not, so the generative pair is not advertised as idempotent.
        "idempotentHint": not open_world,
        "openWorldHint": open_world,
    }


def resolve_tool_metadata(entry: ToolEntry) -> tuple[str, dict[str, Any]]:
    """``metadata`` hook for :meth:`MCPToolRegistry.apply`."""
    return title_for(entry.name), annotations_for(entry.name)


__all__ = [
    "OPEN_WORLD_TOOLS",
    "TITLE_OVERRIDES",
    "WRITER_TOOLS",
    "annotations_for",
    "resolve_tool_metadata",
    "title_for",
]
