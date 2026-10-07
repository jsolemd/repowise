"""Server side of MCP tool selection: gather inputs, apply the result.

The pure rules live in :mod:`repowise.core.registry.tool_selection`. This
module reads their inputs from disk (the ``mcp.tools`` config override,
workspace mode, :class:`AvailabilityFacts`) and applies the resolved set to
the FastMCP instance.

The registry attaches *every* tool to the FastMCP instance the first time a
caller asks for the full surface (``mcp_server.ensure_full_surface``).
:func:`apply_tool_selection` then trims that set once, at boot, by removing the
deselected tools from the FastMCP tool manager, so raw MCP clients see the
surface as it stood when the server started. Chat calls
:func:`selected_tool_entries` instead, which re-resolves on every turn. There
is no per-call cost and tool schemas are untouched.

The hard no-generative policy excludes tools regardless of profile or override.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from repowise.core.generative_policy import (
    NO_GENERATIVE_ENV,
    generative_calls_disabled,
)
from repowise.core.registry import ToolEntry, mcp_tool_registry
from repowise.core.registry.tool_selection import (
    ALL,
    AvailabilityFacts,
    normalize_override,
)
from repowise.core.registry.tool_selection import (
    resolve_enabled_tools as _resolve_enabled_tools,
)

_log = logging.getLogger("repowise.mcp")

# The only two tools that call an LLM: get_answer synthesizes a cited answer,
# generate_refactoring_code writes a patch. Every other tool reads the local
# index. Named here rather than derived from a per-tool flag because the point
# of the list is to be auditable — an operator reads two names, not seventeen
# tool modules.
GENERATIVE_TOOL_NAMES = frozenset({"get_answer", "generate_refactoring_code"})


# Hard exclusion switch for the tools above. Deployments exist that must be
# able to *demonstrate* the server never generates — no API key to leak, no
# model output in the audit trail, an agent host that supplies its own model —
# and for them "it is not in the config" is not an answer, because a config is
# editable by whoever the deployment is being defended against. With this set,
# the two tools are removed from the served surface at bind time and cannot be
# selected back in by any profile, delta or explicit allowlist.
def no_generative_tools_enabled(repo_path: Path | str | None = None) -> bool:
    """Whether the generative tools are hard-excluded from the surface.

    Off by default; on for ``1``/``true``/``yes``/``on`` (case-insensitive).
    A workspace server serves every member, so a truthy policy in any member is
    also authoritative. Read at the call site, not at import, so a test or an
    embedding process can set it after this module is loaded.
    """
    if generative_calls_disabled(repo_path):
        return True
    if repo_path is None:
        return False

    try:
        from repowise.core.workspace import WorkspaceConfig, find_workspace_root

        workspace_root = find_workspace_root(Path(repo_path))
        if workspace_root is None:
            return False
        config = WorkspaceConfig.load(workspace_root)
    except Exception:
        # Once a workspace config exists, failing to enumerate its members must
        # remove the generative surface rather than silently omit a member's
        # hard policy from the decision.
        _log.warning(
            "Could not resolve workspace members for the hard no-generative policy; failing closed",
            exc_info=True,
        )
        return True

    return any(
        generative_calls_disabled(path)
        for path in (workspace_root, *config.repo_paths(workspace_root))
    )


def _strip_generative(
    enabled: set[str],
    *,
    tokens: Sequence[str] | None,
    repo_path: Path | str | None = None,
) -> set[str]:
    """Remove the generative tools from a resolved selection when excluded.

    Runs after every profile/delta/allowlist path, which is what makes the
    exclusion unconditional: there is no ordering of config tokens that lands
    downstream of it. A selection that explicitly named one of the tools gets a
    warning per name, because that is a real conflict between two operator
    intents and silently winning it would look like the config was ignored.
    """
    if not no_generative_tools_enabled(repo_path):
        return enabled
    named = {t.lstrip("+-").strip() for t in (tokens or ())}
    for name in sorted(enabled & GENERATIVE_TOOL_NAMES & named):
        _log.warning(
            "Suppressing generative MCP tool %r requested by the tool selection: %s is set",
            name,
            NO_GENERATIVE_ENV,
        )
    return enabled - GENERATIVE_TOOL_NAMES


# Snapshot of every tool registered on the server, captured once after the
# registry applies them and before any selection trims the live set. Selection
# rebuilds the advertised set from this snapshot, so it is idempotent and can
# re-add a tool a previous call removed (the FastMCP manager only supports
# removal, not re-registration).
_full_surface: dict[str, Any] | None = None
_selected_surface: tuple[bool, frozenset[str]] | None = None


def _ensure_registered() -> None:
    """Make sure every tool module has been imported before reading the registry.

    Tool modules import lazily (see ``mcp_server/__init__``), so the registry is
    empty until something asks for the full surface. Both public entry points
    here read ``mcp_tool_registry.entries()``, and both are imported directly by
    callers that never touch the server instance — the dashboard's ``/mcp``
    routes among them — so neither can assume registration already happened.
    """
    from repowise.server.mcp_server import ensure_full_surface

    ensure_full_surface()


def snapshot_full_surface(mcp: Any) -> None:
    """Record the complete registered tool set so selection can rebuild from it.

    Called once by ``ensure_full_surface``, right after the registry attaches
    every tool. Safe to call again; the first non-empty snapshot wins so a later
    call (after the set has been trimmed) cannot shrink the source of truth.

    Also the enforcement point for :data:`NO_GENERATIVE_ENV`: the excluded tools
    are dropped from the live manager *and* from the snapshot before either is
    read. Selection only ever rebuilds from the snapshot, so a tool removed here
    has no path back onto the surface — and a server that never runs selection
    at all (any consumer that reads ``mcp_server.mcp`` directly) still never
    advertises it.
    """
    global _full_surface
    manager = getattr(mcp, "_tool_manager", None)
    registered = getattr(manager, "_tools", None)
    if registered:
        _purge_generative(mcp, registered)
    if _full_surface is not None:
        return
    if registered:
        _full_surface = dict(registered)


def _purge_generative(mcp: Any, registered: dict[str, Any]) -> None:
    """Drop the generative tools from a live FastMCP server, in place.

    Removal goes through ``FastMCP.remove_tool`` when the server offers it
    (mcp 1.28's public unregistration path, added after this code was written),
    so the one part of the trim that *is* a plain removal no longer depends on
    the private ``_tool_manager._tools`` mapping. The rebuild in
    :func:`apply_tool_selection` still does, because restoring a previously
    trimmed tool means re-adding a built ``Tool`` object and the SDK exposes no
    public call for that — ``ToolManager.add_tool`` takes a function and
    re-derives the schema, and ``FastMCP(tools=...)`` is constructor-only.
    """
    if not no_generative_tools_enabled():
        return
    removed = sorted(GENERATIVE_TOOL_NAMES & registered.keys())
    remove_tool = getattr(mcp, "remove_tool", None)
    for name in removed:
        if callable(remove_tool):
            remove_tool(name)
        else:
            del registered[name]
    if removed:
        _log.info(
            "Generative MCP tools excluded from the served surface (%s): %s",
            NO_GENERATIVE_ENV,
            ", ".join(removed),
        )


def resolve_enabled_tools(
    entries: Iterable[ToolEntry],
    *,
    is_workspace: bool,
    override: str | Sequence[str] | None = None,
    repo_path: Path | str | None = None,
    facts: AvailabilityFacts | None = None,
) -> set[str]:
    """Apply the shared pure selection rules, then the hard generation policy."""
    enabled = _resolve_enabled_tools(
        entries, is_workspace=is_workspace, override=override, facts=facts
    )
    return _strip_generative(enabled, tokens=normalize_override(override), repo_path=repo_path)


def _read_config_override(repo_path: str | None) -> str | Sequence[str] | None:
    """Read ``mcp.tools`` from ``.repowise/config.yaml`` if present."""
    if not repo_path:
        return None
    try:
        from repowise.core.repo_config import load_repo_config

        mcp_cfg = load_repo_config(repo_path).get("mcp") or {}
        if isinstance(mcp_cfg, dict):
            return mcp_cfg.get("tools")
    except Exception:
        _log.debug("Failed to read mcp.tools from config", exc_info=True)
    return None


def _is_workspace(repo_path: str | None) -> bool:
    if not repo_path:
        return False
    try:
        from repowise.core.workspace.config import find_workspace_root

        return find_workspace_root(Path(repo_path)) is not None
    except Exception:
        _log.debug("Workspace detection failed during tool selection", exc_info=True)
        return False


def _availability_facts(repo_path: str | None) -> AvailabilityFacts:
    """Facts that ``available_when`` predicates read for *repo_path*.

    Chat calls this every turn, so it must stay cheap: no database queries.
    No registered tool gates on a fact yet, so there is nothing to compute;
    when one does, add its fact here from a cheap source (config, a file on
    disk) and extend :class:`AvailabilityFacts` to carry it.
    """
    return AvailabilityFacts()


def apply_tool_selection(
    mcp: Any,
    *,
    repo_path: str | None,
    override: str | Sequence[str] | None = None,
) -> set[str]:
    """Trim *mcp*'s registered tools to the resolved surface.

    Resolves the enabled set from the registry metadata, the workspace mode of
    ``repo_path``, and ``override`` (which falls back to the ``mcp.tools`` config
    block when not given on the CLI), then removes every registered tool that is
    not enabled. Returns the enabled set. Safe to call once per server boot.
    """
    _ensure_registered()

    if override is None:
        override = _read_config_override(repo_path)

    is_workspace = _is_workspace(repo_path)
    enabled = resolve_enabled_tools(
        mcp_tool_registry.entries(),
        is_workspace=is_workspace,
        override=override,
        repo_path=repo_path,
        facts=_availability_facts(repo_path),
    )
    global _selected_surface
    _selected_surface = (is_workspace, frozenset(enabled))

    manager = getattr(mcp, "_tool_manager", None)
    registered = getattr(manager, "_tools", None)
    if registered is None:
        return enabled

    # Rebuild from the full snapshot when available so selection is idempotent
    # and can restore a tool a prior call trimmed; otherwise fall back to
    # in-place removal of the currently-registered set.
    #
    # Sorted, because registration order is the order the tool modules were
    # imported in, and that is no longer fixed: tool modules import lazily, so
    # whichever consumer forces the surface first decides it (an HTTP app has
    # already pulled tool_risk in through routers/git.py; a stdio server has
    # not). The advertised order is what an agent reads top-down, so it should
    # not depend on the entry point. Name order is arbitrary but stable, and it
    # still puts get_answer first.
    source = _full_surface if _full_surface is not None else dict(registered)
    registered.clear()
    for name in sorted(source):
        if name in enabled:
            registered[name] = source[name]

    return enabled


def selected_tool_names(*, is_workspace: bool) -> set[str]:
    """Return the surface selected for this running server, or its default."""
    _ensure_registered()
    if _selected_surface is not None and _selected_surface[0] == is_workspace:
        return set(_selected_surface[1])
    return resolve_enabled_tools(mcp_tool_registry.entries(), is_workspace=is_workspace)


def selected_tool_entries(repo_path: str | None) -> list[ToolEntry]:
    """Return the configured MCP entries available to one repository.

    This is the shared selection seam for external MCP clients and in-product
    chat. It deliberately resolves from the live registry on every request so
    neither surface can grow a second catalog or retain a stale config view,
    and so ``available_when`` predicates are re-evaluated every chat turn.
    """
    _ensure_registered()
    entries = mcp_tool_registry.entries()
    enabled = resolve_enabled_tools(
        entries,
        is_workspace=_is_workspace(repo_path),
        override=_read_config_override(repo_path),
        repo_path=repo_path,
        facts=_availability_facts(repo_path),
    )
    return [
        entry
        for entry in sorted(entries, key=lambda item: (item.surface_order, item.name))
        if entry.name in enabled
    ]


def get_registered_tool(name: str) -> Any | None:
    """Return FastMCP's generated tool contract for a registry entry."""
    _ensure_registered()
    return (_full_surface or {}).get(name)


def _tool_description(name: str, fn: Any | None = None) -> str:
    """One-line description for a tool, from its registered FastMCP schema."""
    tool = (_full_surface or {}).get(name)
    desc = getattr(tool, "description", "") or ""
    if not desc and fn is not None:
        desc = inspect.getdoc(fn) or ""
    return desc.strip().split("\n", 1)[0].strip()


def registry_tool_rows(entries: Iterable[ToolEntry] | None = None) -> list[dict[str, Any]]:
    """Mode-independent tool catalog derived only from registry metadata."""
    _ensure_registered()
    catalog = list(entries) if entries is not None else mcp_tool_registry.entries()
    single_default = resolve_enabled_tools(catalog, is_workspace=False)
    workspace_default = resolve_enabled_tools(catalog, is_workspace=True)
    return [
        {
            "name": entry.name,
            "description": _tool_description(entry.name, entry.fn),
            "tier": entry.tier,
            "default_single_repo": entry.name in single_default,
            "default_workspace": entry.name in workspace_default,
            "eligible_single_repo": not entry.requires_workspace,
            "eligible_workspace": True,
            "requires_workspace": entry.requires_workspace,
            "recipes": [
                {
                    "name": recipe.name,
                    "call": recipe.call,
                    "requires": list(recipe.requires),
                }
                for recipe in entry.recipes
            ],
            "artifact_type": entry.artifact_type,
            "presentation": entry.presentation,
            "safety": entry.safety,
            "evidence_basis": entry.evidence_basis,
        }
        for entry in sorted(catalog, key=lambda item: (item.surface_order, item.name))
    ]


def describe_tool_surface(repo_path: str | None) -> dict[str, Any]:
    """Describe the configurable tool surface for a repo (for the settings UI).

    Returns ``is_workspace``, the raw ``override`` currently in config, and one
    row per registered tool with its name, one-line description, and the flags a
    UI needs to render and edit the selection: ``default`` (in the curated
    default set for this mode), ``requires_workspace``, ``eligible`` (usable in
    this mode for this repo), and ``enabled`` (in the currently-resolved surface).

    Hard-excluded generative tools are omitted from the configuration toggles.
    """
    _ensure_registered()

    entries = mcp_tool_registry.entries()
    is_workspace = _is_workspace(repo_path)
    override = _read_config_override(repo_path)
    facts = _availability_facts(repo_path)

    def resolve(value: str | Sequence[str] | None) -> set[str]:
        return resolve_enabled_tools(
            entries, is_workspace=is_workspace, override=value, facts=facts, repo_path=repo_path
        )

    default_surface = resolve(None)
    enabled = resolve(override)
    # "all" is exactly the usable set: right mode and predicate satisfied.
    eligible = resolve(ALL)

    rows = registry_tool_rows(entries)
    tools = [
        {
            **row,
            "default": row["name"] in default_surface,
            "eligible": row["name"] in eligible,
            "enabled": row["name"] in enabled,
        }
        for row in rows
        if not (no_generative_tools_enabled(repo_path) and row["name"] in GENERATIVE_TOOL_NAMES)
    ]
    return {
        "is_workspace": is_workspace,
        "override": list(override) if isinstance(override, (list, tuple)) else override,
        "tools": tools,
    }


def set_tool_override(repo_path: str, tools: str | list[str] | None) -> None:
    """Persist the ``mcp.tools`` override into ``.repowise/config.yaml``.

    A falsy/empty ``tools`` clears the override (the repo falls back to the
    default surface); the ``mcp`` block is removed when it becomes empty so the
    file stays clean. Other config keys are preserved.
    """
    from repowise.core.repo_config import load_repo_config, save_repo_config

    config = load_repo_config(repo_path)
    mcp_cfg = config.get("mcp")
    if not isinstance(mcp_cfg, dict):
        mcp_cfg = {}

    if tools:
        mcp_cfg["tools"] = tools
    else:
        mcp_cfg.pop("tools", None)

    if mcp_cfg:
        config["mcp"] = mcp_cfg
    else:
        config.pop("mcp", None)

    save_repo_config(repo_path, config)
