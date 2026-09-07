"""Derive the ``module_page`` set the current file list would produce.

``update --index-only`` on a deterministic wiki re-renders file pages and
leaves every repo-wide page as the last full run wrote it. That is right for
content and wrong for existence: a module page whose directory was deleted
stays ``fresh`` forever, and FTS, the vector store and the dashboard keep
serving it. Retiring one needs an answer to "which module pages *should* exist
at this commit", and this module is that answer.

The answer is computed by calling the same selection chain ``init`` calls, not
by re-deriving the grouping. Re-implementing it would drop the code-file
filter, the clone-dedupe pre-step and the knowledge-graph layer steering, each
of which moves a group's ``key`` — and a key that moved is a live page this
would delete. So the only honest derivation is the one the generator itself
uses, driven from the same inputs.

The caller pairs the returned ids with
:func:`repowise.core.pipeline.persist.reconcile_module_pages`, which applies
the mass-deletion floor before anything is retired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


def authoritative_module_pages(
    *,
    parsed_files: list[Any],
    graph_builder: Any,
    config: Any,
    repo_path: Path,
    kg_modules: list[dict] | None = None,
    git_meta_map: dict[str, dict] | None = None,
) -> dict[str, str]:
    """``module_page id -> structural_key`` the current file set would produce.

    *parsed_files* must be the **whole** current repo parse, not an update's
    changed slice: the concept partition is a total cover of the production
    files, so a truncated input would call almost every live page stale.

    *kg_modules* wins over the on-disk knowledge graph when the caller has a
    fresh in-memory result, matching the precedence in
    ``_GenerationRun._compute_selection``. The artifact is one run stale on
    update and absent on a fresh init; its absence costs the grouping's taste
    and never its coverage, so a repo with no ``knowledge-graph.json`` still
    derives a complete set.

    Raises whatever the selection chain raises. The caller decides what a
    failed derivation means, and the only safe meaning is "retire nothing".
    """
    # Deferred: ``page_generator`` reaches back into ``generation.selection``
    # at call time, so importing it at module scope from inside this package
    # risks a partially-initialised cycle. Same reason ``orchestrate`` defers
    # its import of this package.
    from repowise.core.generation.kg_context import KnowledgeGraphContext
    from repowise.core.generation.models import compute_page_id
    from repowise.core.generation.page_generator.helpers import (
        _CODE_LANGUAGES,
        _is_infra_file,
        _select_clone_representatives,
    )
    from repowise.core.generation.page_generator.orchestrate import _compute_kg_file_scores

    from .selector import SelectionInputs, select_pages

    pagerank = graph_builder.pagerank()

    code_files = [
        p
        for p in parsed_files
        if not p.file_info.is_api_contract
        and not _is_infra_file(p)
        and p.file_info.language in _CODE_LANGUAGES
    ]

    # Clone dedupe runs before selection for the same reason it does in
    # generation: a dropped clone is absent from the partition, so a set
    # derived without this step contains groups init never minted.
    parsed_files_for_selection = parsed_files
    if getattr(config, "dedupe_near_clones", True):
        drop_paths = _select_clone_representatives(code_files, pagerank)
        if drop_paths:
            parsed_files_for_selection = [
                p for p in parsed_files if p.file_info.path not in drop_paths
            ]

    try:
        community_info_map = graph_builder.community_info() or {}
    except Exception:
        community_info_map = {}

    # Same single argument ``_GenerationRun`` passes: the context treats an
    # absent file as "no knowledge graph" on its own, and adding a repo root
    # here would index paths differently from the run this must reproduce.
    kg_ctx = KnowledgeGraphContext(repo_path / ".repowise" / "knowledge-graph.json")
    kg_scores = _compute_kg_file_scores(kg_ctx)

    selection = select_pages(
        SelectionInputs(
            parsed_files=parsed_files_for_selection,
            pagerank=pagerank,
            betweenness=graph_builder.betweenness_centrality(),
            community=graph_builder.community_detection(),
            community_info=community_info_map,
            sccs=list(graph_builder.strongly_connected_components()),
            git_meta_map=git_meta_map,
            config=config,
            kg_file_scores=kg_scores or None,
            kg_modules=kg_modules or kg_ctx.get_modules() or None,
        )
    )

    # ``module_groups`` is unrationed — every group that exists becomes a page
    # — so the selection's list *is* the authoritative set, with no budget to
    # reproduce. ``key`` is the persisted ``target_path`` and is not always a
    # filesystem path: the grouper falls back to ``<prefix>#<hash>`` when every
    # candidate directory is already taken, so nothing here may treat it as one.
    pages = {
        compute_page_id("module_page", group.key): group.structural_key
        for group in selection.module_groups
    }
    log.info(
        "module_reconcile.derived",
        modules=len(pages),
        parsed_files=len(parsed_files),
        kg_available=kg_ctx.available,
    )
    return pages
