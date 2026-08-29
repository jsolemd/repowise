"""Titles and behaviour annotations for the whole registered tool surface.

``test_solemd_wire_contract`` pins the 25 tools this deployment serves. This
covers the rest of what the registry carries — the opt-in writer and the two
generative tools — plus the derivation rule itself, so a tool added upstream
gets a sane title and honest hints without anyone remembering to add a row.
"""

from __future__ import annotations

import pytest

from repowise.server.mcp_server._tool_metadata import (
    OPEN_WORLD_TOOLS,
    WRITER_TOOLS,
    annotations_for,
    resolve_tool_metadata,
    title_for,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("search_codebase", "Search codebase"),
        ("get_symbol", "Get symbol"),
        ("preview_symbol_rename", "Preview symbol rename"),
        # Overridden, because the derivation reads badly for these three.
        ("get_why", "Explain why"),
        ("list_repos", "List repositories"),
        ("manage_decision", "Manage decisions"),
    ],
)
def test_titles_are_sentence_case_with_named_overrides(name: str, expected: str):
    assert title_for(name) == expected


def test_every_registered_tool_gets_a_title_and_annotations():
    from repowise.core.registry import mcp_tool_registry
    from repowise.server.mcp_server import ensure_full_surface

    ensure_full_surface()
    entries = mcp_tool_registry.entries()
    assert entries, "the registry is empty; ensure_full_surface did not run"

    for entry in entries:
        title, annotations = resolve_tool_metadata(entry)
        assert title and title[0].isupper(), entry.name
        assert set(annotations) == {
            "readOnlyHint",
            "destructiveHint",
            "idempotentHint",
            "openWorldHint",
        }, entry.name


def test_writers_are_not_read_only_and_are_not_destructive():
    # Both writers are additive: a decision is superseded rather than deleted,
    # and a reindex rebuilds derived data.
    assert sorted(WRITER_TOOLS) == ["manage_decision", "reindex_repository"]
    for name in WRITER_TOOLS:
        hints = annotations_for(name)
        assert hints["readOnlyHint"] is False
        assert hints["destructiveHint"] is False
        assert hints["idempotentHint"] is False
        assert hints["openWorldHint"] is False


def test_generative_tools_read_the_repo_but_reach_outside_it():
    assert sorted(OPEN_WORLD_TOOLS) == ["generate_refactoring_code", "get_answer"]
    for name in OPEN_WORLD_TOOLS:
        hints = annotations_for(name)
        assert hints["readOnlyHint"] is True
        assert hints["openWorldHint"] is True
        # A model call does not give the same answer twice.
        assert hints["idempotentHint"] is False


def test_plain_readers_are_closed_world_and_idempotent():
    hints = annotations_for("search_codebase")
    assert hints == {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
