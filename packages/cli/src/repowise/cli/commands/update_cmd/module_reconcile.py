"""Plan module-page refresh and retirement from one current store inventory.

Selection owns module identity; this CLI boundary owns rollout policy and
failure reporting. The returned plan shares its snapshot and grouping inputs
with rendering, while persistence independently checks the F45 prune floor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repowise.cli.helpers import console, run_async

from .deterministic import deterministic_generation_config

MODULE_RECONCILE_ENV = "REPOWISE_MODULE_RECONCILE"


@dataclass(frozen=True)
class ModuleReconcilePlan:
    authoritative_page_ids: frozenset[str]
    missing_page_ids: frozenset[str]
    drifted_page_ids: frozenset[str]
    kg_modules: list[dict] | None = None

    @property
    def render_page_ids(self) -> set[str]:
        return set(self.missing_page_ids | self.drifted_page_ids)


def module_reconcile_mode(*, degraded: list[str] | None = None) -> str:
    """Default to on; a misspelled rollout switch only reports changes."""
    raw = os.environ.get(MODULE_RECONCILE_ENV, "").strip().lower()
    if not raw:
        return "on"
    if raw in {"on", "report", "off"}:
        return raw
    message = f"Module page reconcile: invalid {MODULE_RECONCILE_ENV}={raw!r}; using report"
    if degraded is not None:
        degraded.append(message)
    console.print(f"[yellow]{message}[/yellow]")
    return "report"


async def _load_module_page_keys(repo_path: Path) -> dict[str, str]:
    """Read module stamps, propagating failure instead of inventing an empty wiki."""
    from sqlalchemy import select

    from repowise.cli.helpers import get_db_url_for_repo
    from repowise.core.persistence import create_engine, create_session_factory, get_session
    from repowise.core.persistence.models import Page

    engine = create_engine(get_db_url_for_repo(repo_path))
    try:
        async with get_session(create_session_factory(engine)) as session:
            rows = await session.execute(
                select(Page.id, Page.structural_key).where(Page.page_type == "module_page")
            )
            return {pid: str(key or "") for pid, key in rows}
    finally:
        await engine.dispose()


def plan_module_reconcile(
    *,
    repo_path: Path,
    parsed_files: list,
    graph_builder: Any,
    git_meta_map: dict,
    kg_modules: list[dict] | None,
    cfg: dict,
    concurrency: int,
    degraded: list[str],
) -> ModuleReconcilePlan | None:
    """Return an actionable plan only after selection and the store read succeed.

    The parse must cover the full repository. Off skips all work; report logs
    the delta but returns no plan. An empty derivation also preserves pages:
    this boundary cannot distinguish a valid empty repo from under-selection.
    If stored modules remain, that refusal is reported as degraded.
    """
    mode = module_reconcile_mode(degraded=degraded)
    if mode == "off":
        return None

    try:
        from repowise.core.generation.selection.module_reconcile import (
            authoritative_module_pages,
        )

        module_ids = authoritative_module_pages(
            parsed_files=parsed_files,
            graph_builder=graph_builder,
            config=deterministic_generation_config(cfg, concurrency=concurrency),
            repo_path=repo_path,
            kg_modules=kg_modules or None,
            git_meta_map=git_meta_map,
        )
        stored = run_async(_load_module_page_keys(repo_path))
    except Exception as exc:
        degraded.append(f"Module page reconcile: {exc}")
        console.print(f"[yellow]module_reconcile skipped: {exc}[/yellow]")
        return None

    if not module_ids:
        if stored:
            message = "Module page reconcile: derived no module pages; preserving stored pages"
            degraded.append(message)
            console.print(f"[yellow]{message}[/yellow]")
        return None

    missing = frozenset(module_ids.keys() - stored.keys())
    # Preserve pre-stamp rows under the existing rollout policy. Their lack
    # of a key is not measured structural drift.
    drifted = frozenset(
        pid for pid, key in module_ids.items() if stored.get(pid) and key and stored[pid] != key
    )
    would_retire = sorted(stored.keys() - module_ids.keys())
    console.print(
        f"  module_reconcile mode={mode} authoritative={len(module_ids)} "
        f"stored={len(stored)} would_retire={len(would_retire)} "
        f"missing={len(missing)} drifted={len(drifted)}"
    )
    if mode == "report":
        for page_id in would_retire:
            console.print(f"    module_reconcile would retire {page_id}")
        return None
    return ModuleReconcilePlan(frozenset(module_ids), missing, drifted, kg_modules)
