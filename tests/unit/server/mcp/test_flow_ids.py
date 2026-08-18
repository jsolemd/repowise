"""Flow handles: what they are a function of, and what they survive."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from repowise.server.mcp_server._flow_ids import (
    GENERATION_DIGEST_CHARS,
    index_generation,
    mint_flow_id,
    parse_flow_id,
)


def _repo(head="abc123", parser="parser-v1"):
    return SimpleNamespace(head_commit=head, graph_edges_parser_fingerprint=parser)


# ---------------------------------------------------------------------------
# The generation
# ---------------------------------------------------------------------------


def test_the_same_index_yields_the_same_generation():
    assert index_generation(_repo()) == index_generation(_repo())


def test_the_digest_is_the_documented_width():
    assert len(index_generation(_repo())) == GENERATION_DIGEST_CHARS


def test_a_new_commit_is_a_new_generation():
    assert index_generation(_repo(head="aaa")) != index_generation(_repo(head="bbb"))


def test_a_new_parser_build_is_a_new_generation():
    """Re-indexing one commit with a changed extractor changes the call graph."""
    assert index_generation(_repo(parser="v1")) != index_generation(_repo(parser="v2"))


def test_an_index_missing_a_field_still_gets_a_stable_generation():
    """An older store recorded no parser fingerprint; it must not raise."""
    older = _repo(parser=None)
    assert index_generation(older) == index_generation(_repo(parser=None))
    assert index_generation(older) != index_generation(_repo(parser="v1"))


def test_a_repository_object_missing_the_attributes_entirely_is_tolerated():
    assert len(index_generation(SimpleNamespace())) == GENERATION_DIGEST_CHARS


def test_the_generation_does_not_move_with_wall_clock_time():
    """Two indexes of one commit find the same flows; the handle must agree."""
    first = SimpleNamespace(
        head_commit="abc", graph_edges_parser_fingerprint="p", indexed_at="2026-01-01"
    )
    second = SimpleNamespace(
        head_commit="abc", graph_edges_parser_fingerprint="p", indexed_at="2026-08-17"
    )
    assert index_generation(first) == index_generation(second)


# ---------------------------------------------------------------------------
# Minting and parsing
# ---------------------------------------------------------------------------


def test_a_handle_is_a_function_of_its_two_inputs():
    assert mint_flow_id("gen1", "a/b.py::Foo") == mint_flow_id("gen1", "a/b.py::Foo")
    assert mint_flow_id("gen1", "a/b.py::Foo") != mint_flow_id("gen2", "a/b.py::Foo")
    assert mint_flow_id("gen1", "a/b.py::Foo") != mint_flow_id("gen1", "a/b.py::Bar")


@pytest.mark.parametrize(
    "entry_point",
    [
        "a/b.py::Foo",
        # The node id carries colons of its own — the whole reason parsing
        # splits once and keeps the remainder.
        "a/b.py::Class::method",
        "bin/solemd::_err",
        "plain_file_node.py",
    ],
)
def test_a_handle_round_trips_whatever_node_id_it_carries(entry_point):
    parsed = parse_flow_id(mint_flow_id("abcdef123456", entry_point))
    assert parsed is not None
    assert parsed.generation == "abcdef123456"
    assert parsed.entry_point == entry_point


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "not-a-flow-id",
        "flow:",
        "flow:gen-only",
        "flow::missing-generation",
        "flow:gen:",
        "notflow:gen:a/b.py::Foo",
    ],
)
def test_anything_that_is_not_a_handle_is_refused(text):
    assert parse_flow_id(text) is None


def test_surrounding_whitespace_is_forgiven():
    assert parse_flow_id("  flow:gen:a/b.py::Foo  ") is not None
