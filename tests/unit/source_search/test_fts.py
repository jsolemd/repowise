"""The lexical leg: the shared tokenizer and BM25 ranking over it."""

from __future__ import annotations

import pytest

from repowise.core.source_search.chunks import SymbolRecord, build_symbol_chunk
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
