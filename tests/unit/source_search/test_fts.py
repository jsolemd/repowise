"""The lexical leg: the shared tokenizer and BM25 ranking over it."""

from __future__ import annotations

import sqlite3

import pytest

from repowise.core.source_search.chunks import (
    SymbolRecord,
    build_symbol_chunk,
    iter_file_windows,
)
from repowise.core.source_search.fts import SourceFTSIndex, default_fts_path, tokenize

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("parseConfig", ["parse", "config"]),
        ("parse_config", ["parse", "config"]),
        ("ParseConfig", ["parse", "config"]),
        ("PARSE_CONFIG", ["parse", "config"]),
        ("parse-config", ["parse", "config"]),
        ("parse.config", ["parse", "config"]),
        ("parseHTTPConfig", ["parse", "httpconfig"]),
        ("value2Name", ["value2", "name"]),
    ],
)
def test_identifier_spellings_collapse_to_one_stream(text, expected):
    assert tokenize(text) == expected


def test_camel_and_snake_of_the_same_name_tokenize_identically():
    """The whole point: a query in one casing must reach a definition in another."""
    assert tokenize("getUserById") == tokenize("get_user_by_id")


def test_single_characters_are_dropped():
    assert tokenize("a bb c ddd") == ["bb", "ddd"]
    assert tokenize("x = y + zz") == ["zz"]


def test_punctuation_and_paths_become_separators():
    assert tokenize("src/app/main.py:42") == ["src", "app", "main", "py", "42"]


def test_empty_and_symbol_only_text_yields_nothing():
    assert tokenize("") == []
    assert tokenize("!!! ??? ---") == []


# ---------------------------------------------------------------------------
# Index round-trip
# ---------------------------------------------------------------------------


def _chunk(name: str, body: str, path: str | None = None):
    path = path or f"src/{name}.py"
    return build_symbol_chunk(
        SymbolRecord(
            symbol_id=f"{path}::{name}",
            file_path=path,
            name=name,
            qualified_name=name,
            kind="function",
            signature="",
            docstring=None,
            start_line=1,
            end_line=len(body.splitlines()),
            language="python",
        ),
        body.splitlines(),
    )


@pytest.fixture
def index(tmp_path):
    with SourceFTSIndex(tmp_path / "source_fts.db") as fts:
        yield fts


def test_default_path_is_a_sidecar_under_repowise(tmp_path):
    assert default_fts_path(tmp_path) == tmp_path / ".repowise" / "source_search" / "source_fts.db"


def test_round_trip_counts_what_was_written(index):
    chunks = [_chunk("alpha", "def alpha():\n    pass"), _chunk("beta", "def beta():\n    pass")]
    assert index.index_chunks(chunks) == 2
    assert index.count() == 2


def test_an_exact_identifier_query_ranks_its_definition_first(index):
    index.index_chunks(
        [
            _chunk("parse_config", "def parse_config(path):\n    return read(path)"),
            _chunk("write_output", "def write_output(data):\n    return dump(data)"),
            _chunk("render_page", "def render_page(page):\n    return html(page)"),
        ]
    )
    hits = index.query("parse_config")
    assert hits
    assert hits[0].chunk_id == "src/parse_config.py::parse_config"


def test_a_camel_query_finds_a_snake_definition(index):
    index.index_chunks(
        [
            _chunk("get_user_by_id", "def get_user_by_id(uid):\n    return db.get(uid)"),
            _chunk("unrelated", "def unrelated():\n    return 0"),
        ]
    )
    hits = index.query("getUserById")
    assert hits[0].chunk_id == "src/get_user_by_id.py::get_user_by_id"


def test_scores_are_higher_is_better(index):
    index.index_chunks([_chunk("alpha", "def alpha():\n    return alpha_thing")])
    hits = index.query("alpha")
    assert hits[0].score > 0


def test_hits_are_ordered_by_descending_score(index):
    index.index_chunks(
        [
            _chunk("alpha", "def alpha():\n    return alpha + alpha + alpha"),
            _chunk("beta", "def beta():\n    return alpha"),
        ]
    )
    scores = [hit.score for hit in index.query("alpha")]
    assert scores == sorted(scores, reverse=True)


def test_a_query_with_no_usable_tokens_returns_nothing(index):
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass")])
    assert index.query("!!!") == []
    assert index.query("a") == []


def test_fts_syntax_in_a_query_is_not_executed(index):
    """A query that would be an FTS5 expression must be read as words."""
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass")])
    assert index.query('alpha" OR "') == index.query("alpha or")


def test_limit_is_honoured(index):
    index.index_chunks([_chunk(f"alpha{i}", "def f():\n    return alpha") for i in range(10)])
    assert len(index.query("alpha", limit=3)) == 3


def test_delete_by_file_removes_only_that_file(index):
    index.index_chunks(
        [
            _chunk("alpha", "def alpha():\n    pass", path="src/a.py"),
            _chunk("beta", "def beta():\n    pass", path="src/b.py"),
        ]
    )
    assert index.delete_by_file(["src/a.py"]) == 1
    assert index.count() == 1
    assert index.query("beta")[0].file_path == "src/b.py"


def test_delete_by_file_with_no_paths_is_a_no_op(index):
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass")])
    assert index.delete_by_file([]) == 0
    assert index.count() == 1


def test_term_file_evidence_keeps_concepts_bound_to_their_own_files(index):
    index.index_chunks(
        [
            _chunk("neo4j_writer", "def neo4j_writer():\n    return graph", path="src/graph.py"),
            _chunk(
                "cashflow_projection",
                "def cashflow_projection():\n    return forecast",
                path="src/finance.py",
            ),
            _chunk(
                "combined",
                "def combined():\n    return neo4j_cashflow",
                path="src/combined.py",
            ),
        ]
    )

    evidence = index.term_file_evidence(["Neo4j", "cashflow", "neo4j"])

    assert evidence == {
        "neo4j": frozenset({"src/graph.py", "src/combined.py"}),
        "cashflow": frozenset({"src/finance.py", "src/combined.py"}),
    }


def test_term_file_evidence_cache_is_invalidated_by_a_write(index):
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass", path="src/a.py")])
    assert index.term_file_evidence(["alpha"])["alpha"] == frozenset({"src/a.py"})

    index.delete_by_file(["src/a.py"])

    assert index.term_file_evidence(["alpha"])["alpha"] == frozenset()
    assert index.active_file_paths() == []


def test_legacy_delete_invalidates_term_and_path_caches(index):
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass", path="src/a.py")])
    index._versioned = False
    assert index.term_file_evidence(["alpha"])["alpha"] == frozenset({"src/a.py"})
    assert index.active_file_paths() == ["src/a.py"]

    index.delete_by_file(["src/a.py"])

    assert index.term_file_evidence(["alpha"])["alpha"] == frozenset()
    assert index.active_file_paths() == []


def test_file_inventory_counts_active_lanes_exactly(index):
    path = "src/alpha.py"
    chunks = [
        _chunk("alpha", "def alpha():\n    pass", path=path),
        *iter_file_windows(path, "alpha = 1\nbeta = 2\n"),
    ]
    index.index_chunks(chunks)

    inventory = index.inventory_for_file(path)

    assert inventory.total == 2
    assert inventory.symbol == 1
    assert inventory.file_window == 1


def test_file_inventory_reads_the_indexed_versions_table_without_scanning_fts(index):
    path = "src/alpha.py"
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass", path=path)])
    statements: list[str] = []
    index._conn.set_trace_callback(statements.append)
    try:
        inventory = index.inventory_for_file(path)
    finally:
        index._conn.set_trace_callback(None)

    inventory_selects = [
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ]
    assert inventory.total == 1
    assert len(inventory_selects) == 1
    assert "FROM source_fts_versions AS v" in inventory_selects[0]
    assert "source_fts AS f" not in inventory_selects[0]


def test_file_inventory_reports_exact_zero_for_absent_path(index):
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass")])

    inventory = index.inventory_for_file("src/missing.py")

    assert inventory.total == 0
    assert inventory.symbol == 0
    assert inventory.file_window == 0


def test_recreate_empties_the_table(index):
    index.index_chunks([_chunk("alpha", "def alpha():\n    pass")])
    index.recreate()
    assert index.count() == 0
    assert index.query("alpha") == []


def test_reopening_sees_the_previous_run(tmp_path):
    path = tmp_path / "source_fts.db"
    with SourceFTSIndex(path) as first:
        first.index_chunks([_chunk("alpha", "def alpha():\n    pass")])
    with SourceFTSIndex(path) as second:
        assert second.count() == 1
        assert second.query("alpha")


# ---------------------------------------------------------------------------
# Read-only opens
# ---------------------------------------------------------------------------


def test_a_read_only_open_reads_the_store_without_writing_a_byte_of_it(tmp_path):
    """Observers must be able to answer questions about a store they cannot alter."""

    path = tmp_path / "store" / "source_fts.db"
    with SourceFTSIndex(path) as writer:
        writer.index_chunks([_chunk("alpha", "def alpha():\n    pass")])
    before = (path.read_bytes(), path.stat().st_mtime_ns)

    with SourceFTSIndex(path, read_only=True) as reader:
        assert reader.count() == 1
        assert reader.query("alpha")
        assert reader.inventory_for_file("src/alpha.py").total == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.index_chunks([_chunk("beta", "def beta():\n    pass")])

    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_a_read_only_open_refuses_a_missing_store_instead_of_creating_one(tmp_path):
    """The default constructor mkdirs and applies the schema; this one must not.

    A status read on a repository that never built a source index has to be
    able to say "there is nothing here" without leaving an empty store behind
    that the next read would then report as a real, if empty, publication.
    """

    path = tmp_path / "never-built" / "source_fts.db"

    with pytest.raises(FileNotFoundError, match="source FTS store not found"):
        SourceFTSIndex(path, read_only=True)

    assert not path.exists()
    assert not path.parent.exists()
