"""Response-budget contracts for the tools this fork adds to the surface.

Upstream #2051 made one budget contract per tool the rule, and pins it with
``test_every_registered_tool_declares_a_contract``. That rule is right and this
module is how the fork keeps it: upstream owns seventeen tools and declares
seventeen contracts, and the eleven served tools the fork adds declare theirs
here rather than by editing upstream's table, so the next absorption's conflict
surface stays one import line.

Without a contract a tool falls to ``_DEFAULT_CONTRACT``, whose shed order is
empty. That is not "unbounded" — the final guard still fits the response — but
an empty shed order means the guard is the *only* thing that acts, and it works
by removing whole unprotected top-level keys before it trims any list. On these
tools that is backwards: the lists are the bulk and the scalars are the answer.
``get_dependents`` would lose ``total`` and ``has_more`` while keeping every row
it could not honestly count; ``get_index_status`` would lose ``trust`` and
``recipe``, the two keys the whole tool exists to report.

So each contract below reads the same way: **shed the row lists, tail-first,
and protect everything a reader has to have in order to act on what survives**
— every scalar claim, ``status``/``error``, and the keys
``infra/repowise/consumer-check.sh`` and the SoleMD doctrine quote by name.
"""

from __future__ import annotations

from repowise.server.mcp_server._budget.contracts import (
    _CONTRACTS,
    ResponseBudgetContract,
)

__all__ = ["FORK_CONTRACTS"]

#: Keys every fork tool protects. A response that cannot say what it is or that
#: it failed is not a smaller answer, it is a different one.
_ALWAYS = ("status", "error")

#: Pagination is a claim about rows the caller did not get. Shedding it while
#: keeping the rows turns "12 of 380" into an unqualified 12.
_PAGING = ("total", "returned", "offset", "next_offset", "has_more", "limit", "pagination")

FORK_CONTRACTS: dict[str, ResponseBudgetContract] = {
    # --- task slices -------------------------------------------------------
    # One envelope, three tools. ``history`` is the last five events and goes
    # first; ``edges`` and ``externals`` are already capped and ranked, so the
    # budget only has to finish what the tool started; ``members`` is the slice
    # itself and goes last. ``seeds`` is protected: it names what was asked for,
    # and a slice that cannot say what it was cut from is not re-cuttable.
    **{
        tool: ResponseBudgetContract(
            "blocks",
            ("history[]", "edges[]", "externals[]", "members[]"),
            expansion_argument=None,
            protected=(
                *_ALWAYS,
                "slice_id",
                "task",
                "view",
                "summary",
                "seeds",
                "ranking",
                "budget",
                "next",
                "extension",
                "index_drift",
                "walk_truncated",
                "externals_truncated",
                "edges_truncated",
                "edges_ranked_by",
            ),
        )
        for tool in ("build_task_slice", "get_task_slice", "extend_task_slice")
    },
    # --- duplication and patterns -----------------------------------------
    "find_clones": ResponseBudgetContract(
        "blocks",
        ("degradations[]", "findings[]"),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            "summary",
            "scope",
            "definition",
            "near_clones",
            "no_results_reason",
            "limit_note",
            "ignored_arguments",
        ),
    ),
    "find_patterns": ResponseBudgetContract(
        "blocks",
        ("available_patterns[]", "matches[]"),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            "pattern",
            "definition",
            "summary",
            "limit_note",
            "ignored_arguments",
            "no_results_reason",
            "percentile",
            "min_lines",
            "min_files",
            "min_callers",
            "min_siblings",
            "min_directories",
            "include_tests",
        ),
    ),
    # --- graph -------------------------------------------------------------
    # ``explanation`` and ``matched_by`` say how the target was resolved. A
    # dependent list whose target resolution is unstated cannot be checked.
    "get_dependents": ResponseBudgetContract(
        "blocks",
        ("dependents[]",),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            *_PAGING,
            "target",
            "node",
            "node_type",
            "file",
            "query",
            "depth",
            "edge_types",
            "matched_by",
            "explanation",
            "ranking",
            "reference_count_unit",
            "include_tests",
            # A dict keyed by depth, not a row list: it is the shape of the
            # answer, and it stays true no matter how many rows are shed.
            "counts_by_depth",
        ),
    ),
    # --- reference sites ---------------------------------------------------
    # ``coverage`` and ``unbound`` are what make a site list safe to act on:
    # they say what the walk could not bind. ``working_tree`` is a freshness
    # claim the doctrine tells agents to read.
    "get_reference_sites": ResponseBudgetContract(
        "blocks",
        ("candidates[]", "counts_by_language[]", "counts_by_kind[]", "sites[]"),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            *_PAGING,
            "symbol_id",
            "target",
            "query",
            "matched_by",
            "explanation",
            "coverage",
            "unbound",
            "confidence_scale",
            "working_tree",
        ),
    ),
    # A rename preview is a safety verdict. Every key that carries the verdict
    # outranks the site list it was computed from.
    "preview_symbol_rename": ResponseBudgetContract(
        "blocks",
        ("candidates[]", "sites[]"),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            "total",
            "returned",
            "symbol",
            "symbol_id",
            "old_name",
            "new_name",
            "query",
            "matched_by",
            "explanation",
            "summary",
            "mechanically_safe",
            "needs_review",
            "caveats",
            "safe_threshold",
            "applies_changes",
            "files_touched",
            "coverage",
            "confidence_scale",
            "working_tree",
        ),
    ),
    # --- retrieval quality -------------------------------------------------
    "get_query_quality": ResponseBudgetContract(
        "blocks",
        ("cases[]",),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            "mode",
            "repo",
            "report",
            "verdicts",
            "suite",
            "write",
            "next_action",
            "case_count",
            "cases_listed",
            "pending_cases",
            "valid_buckets",
        ),
    ),
    # --- index status ------------------------------------------------------
    # Nothing here is a row list: the whole payload is the trust claim, and the
    # doctrine tells agents to read `trust`, `generation_id`,
    # `published_generation_id` and `working_tree` off it before believing a
    # search result. The two file lists inside `generation` are the only
    # sheddable parts, and each already carries its own `_count`.
    "get_index_status": ResponseBudgetContract(
        "blocks",
        (
            "generation.uncommitted_indexed_paths[]",
            "generation.stale_files[]",
            "degradation_findings[]",
        ),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            "mode",
            "repo",
            "trust",
            "generation",
            "recipe",
            "stores",
            "queue",
            "working_tree",
            "degraded",
            "degraded_reason",
            "last_update",
            "repos",
        ),
    ),
    # --- decisions ---------------------------------------------------------
    # ``journal_exists`` / ``journal_created`` are the fork's journal-mode
    # disclosure: consumer-check.sh reads them, and an agent that cannot tell a
    # missing journal from an empty one will write into the derived store.
    "manage_decision": ResponseBudgetContract(
        "blocks",
        ("chain[]", "decisions[]"),
        expansion_argument=None,
        protected=(
            *_ALWAYS,
            "total",
            "returned",
            "offset",
            "action",
            "repo",
            "per_repo",
            "decision",
            "decisions",
            "journal",
            "journal_exists",
            "journal_created",
            "journal_available",
            "confirmed",
            "note",
            "paging",
        ),
    ),
}

# Additive by construction: a fork tool never overwrites an upstream contract,
# so a future release that adopts one of these names wins and this module goes
# quiet rather than silently shadowing it.
for _name, _contract in FORK_CONTRACTS.items():
    _CONTRACTS.setdefault(_name, _contract)
