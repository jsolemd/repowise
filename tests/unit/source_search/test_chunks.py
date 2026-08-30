"""The chunk recipe: header shape, the smart cap, and window coverage."""

from __future__ import annotations

import pytest

from repowise.core.source_search.chunks import (
    SMART_CAP_CHARS,
    SOURCE_FILE_WINDOW,
    SOURCE_SYMBOL,
    TRUNCATION_MARKER,
    WINDOW_LINES,
    WINDOW_STRIDE,
    SymbolRecord,
    apply_smart_cap,
    build_symbol_chunk,
    iter_file_windows,
    looks_binary,
    recipe_parameters,
    smart_cap,
    window_eligible,
)

_FILE = """import os


def parse_config(path):
    \"\"\"Read it.\"\"\"
    return os.path.exists(path)
"""


def _symbol(**overrides) -> SymbolRecord:
    base = {
        "symbol_id": "src/app.py::parse_config",
        "file_path": "src/app.py",
        "name": "parse_config",
        "qualified_name": "app.parse_config",
        "kind": "function",
        "signature": "def parse_config(path)",
        "docstring": "Read it.",
        "start_line": 4,
        "end_line": 6,
        "language": "python",
    }
    base.update(overrides)
    return SymbolRecord(**base)


# ---------------------------------------------------------------------------
# Symbol chunk text
# ---------------------------------------------------------------------------


def test_symbol_chunk_text_is_header_signature_docstring_body():
    chunk = build_symbol_chunk(_symbol(), _FILE.splitlines())
    assert chunk.text == (
        "# File: src/app.py\n"
        "# function: app.parse_config\n"
        "# Lines: 4-6\n"
        "def parse_config(path)\n"
        '"""Read it."""\n'
        "def parse_config(path):\n"
        '    """Read it."""\n'
        "    return os.path.exists(path)"
    )
    assert chunk.chunk_id == "src/app.py::parse_config"
    assert chunk.source == SOURCE_SYMBOL


def test_symbol_header_falls_back_to_the_bare_name():
    chunk = build_symbol_chunk(_symbol(qualified_name=""), _FILE.splitlines())
    assert chunk.text.splitlines()[1] == "# function: parse_config"


def test_local_symbol_chunk_keeps_its_full_lexical_identity():
    chunk = build_symbol_chunk(
        _symbol(
            symbol_id="src/app.py::outer::LocalAdapter::run",
            name="run",
            qualified_name="app.outer.LocalAdapter.run",
            kind="method",
        ),
        _FILE.splitlines(),
    )
    assert chunk.chunk_id == "src/app.py::outer::LocalAdapter::run"
    assert chunk.text.splitlines()[1] == "# method: app.outer.LocalAdapter.run"


def test_optional_parts_are_omitted_not_blanked():
    """A missing signature or docstring must not leave an empty line behind.

    An empty line in the middle of the header would be indexed as content and
    would change the chunk's hash for a reason that is not a change in source.
    """
    chunk = build_symbol_chunk(_symbol(signature="", docstring=None), _FILE.splitlines())
    lines = chunk.text.splitlines()
    assert lines[:3] == ["# File: src/app.py", "# function: app.parse_config", "# Lines: 4-6"]
    assert lines[3] == "def parse_config(path):"


def test_body_is_the_inclusive_one_indexed_slice():
    lines = [f"line{i}" for i in range(1, 11)]
    chunk = build_symbol_chunk(
        _symbol(signature="", docstring=None, start_line=3, end_line=5), lines
    )
    assert chunk.text.splitlines()[3:] == ["line3", "line4", "line5"]


def test_unpopulated_bounds_yield_a_header_only_chunk():
    """A row whose bounds default to 0 must not slice from the top of the file."""
    chunk = build_symbol_chunk(
        _symbol(signature="", docstring=None, start_line=0, end_line=0),
        [f"line{i}" for i in range(1, 11)],
    )
    assert chunk.text == "# File: src/app.py\n# function: app.parse_config\n# Lines: 0-0"


def test_content_hash_tracks_the_text():
    a = build_symbol_chunk(_symbol(), _FILE.splitlines())
    b = build_symbol_chunk(_symbol(), _FILE.replace("os.path", "posixpath").splitlines())
    assert a.content_hash != b.content_hash
    assert a.content_hash == build_symbol_chunk(_symbol(), _FILE.splitlines()).content_hash


# ---------------------------------------------------------------------------
# Smart cap
# ---------------------------------------------------------------------------


def test_text_at_the_cap_is_untouched():
    text = "x" * SMART_CAP_CHARS
    assert apply_smart_cap(text) == text


def test_one_character_over_the_cap_truncates():
    text = "x" * (SMART_CAP_CHARS + 1)
    capped = apply_smart_cap(text)
    assert capped != text
    assert TRUNCATION_MARKER in capped


def test_truncation_keeps_the_head_and_the_tail():
    head = "H" * 5000
    tail = "T" * 5000
    capped = apply_smart_cap(head + tail)
    before, after = capped.split(TRUNCATION_MARKER)
    assert before == "H" * int(SMART_CAP_CHARS * 0.68)
    assert after == "T" * int(SMART_CAP_CHARS * 0.28)


def test_truncated_length_is_a_constant_of_the_cap():
    """Two very different oversized inputs must produce the same length."""
    short_over = apply_smart_cap("a" * (SMART_CAP_CHARS + 10))
    long_over = apply_smart_cap("b" * (SMART_CAP_CHARS * 20))
    assert len(short_over) == len(long_over)


def test_the_cap_applies_to_symbol_chunks():
    lines = ["y" * 200 for _ in range(100)]
    chunk = build_symbol_chunk(
        _symbol(signature="", docstring=None, start_line=1, end_line=100), lines
    )
    assert TRUNCATION_MARKER in chunk.text
    assert len(chunk.text) < SMART_CAP_CHARS + len(TRUNCATION_MARKER)


# ---------------------------------------------------------------------------
# Token-aware cap (F35)
# ---------------------------------------------------------------------------
#
# ``SMART_CAP_CHARS`` measures characters, and characters are a proxy for what
# an embedding endpoint actually refuses. The proxy is off by a factor of four
# in both directions — 30,000 characters of CJK is 28,500 tokens, 30,000 of
# ASCII source is 6,800 — so the cap either wastes most of the budget or fails
# to bind at all, depending on the text. A caller that can count tokens passes
# a counter and gets the real limit; one that cannot keeps the old behaviour.


def _one_token_per_char(text: str) -> int:
    """The densest a tokenizer can be: every character its own token."""
    return len(text)


def _four_tokens_per_char(text: str) -> int:
    """Denser than any real tokenizer, so a short text is over any budget."""
    return len(text) * 4


def test_the_default_path_reports_characters_and_returns_the_old_text():
    fits = smart_cap("x" * SMART_CAP_CHARS)
    assert (fits.text, fits.truncated, fits.cap_basis) == ("x" * SMART_CAP_CHARS, False, "chars")

    over = smart_cap("x" * (SMART_CAP_CHARS + 1))
    assert over.truncated is True
    assert over.cap_basis == "chars"
    assert over.measured == SMART_CAP_CHARS + 1
    assert over.text == apply_smart_cap("x" * (SMART_CAP_CHARS + 1))


def test_a_counter_switches_the_basis_to_tokens():
    result = smart_cap("abcdefghij", cap=100, token_counter=_one_token_per_char)
    assert (result.truncated, result.cap_basis, result.measured) == (False, "tokens", 10)


def test_the_counter_is_called_once_for_a_text_that_fits():
    """The common case must not pay for a search it does not need."""
    calls = []

    def counting(text: str) -> int:
        calls.append(len(text))
        return len(text)

    smart_cap("short enough", cap=1000, token_counter=counting)
    assert calls == [len("short enough")]


def test_a_text_the_character_cap_would_pass_is_cut_by_tokens():
    """The case the character cap cannot see.

    2,000 characters is far inside ``SMART_CAP_CHARS`` and untouched by the
    default path, while a tokenizer that measures four tokens to the character
    puts it at 8,000 — over the 1,000-token budget asked for here.
    """
    text = "y" * 2000
    assert apply_smart_cap(text) == text

    result = smart_cap(text, cap=1000, token_counter=_four_tokens_per_char)
    assert result.truncated is True
    assert result.cap_basis == "tokens"
    assert result.measured == 8000
    assert TRUNCATION_MARKER in result.text
    head, tail = result.text.split(TRUNCATION_MARKER)
    # Head and tail shares are taken of the *budget in tokens*, then converted
    # back to characters by the counter: 680 tokens is 170 characters at this
    # density, 280 tokens is 70.
    assert _four_tokens_per_char(head) <= int(1000 * 0.68)
    assert _four_tokens_per_char(tail) <= int(1000 * 0.28)
    assert head == "y" * 170
    assert tail == "y" * 70


def test_the_token_cut_never_overshoots_its_budget():
    """A cut that fits is the invariant; the exact length is not.

    BPE counts are only almost monotone in prefix length, so the search may
    land a character or two short. It must never land long.
    """
    text = "".join(chr(ord("a") + i % 26) for i in range(4000))
    result = smart_cap(text, cap=200, token_counter=_one_token_per_char)
    head, tail = result.text.split(TRUNCATION_MARKER)
    assert len(head) <= int(200 * 0.68)
    assert len(tail) <= int(200 * 0.28)
    assert result.truncated is True


def test_a_budget_that_rounds_to_nothing_yields_an_empty_side():
    """A cap so small the tail share is zero must not slice the whole string.

    ``text[-0:]`` is the whole text, which would return more than the cap
    rather than less — the one way this could fail open.
    """
    result = smart_cap("abcdef", cap=1, token_counter=_one_token_per_char)
    assert result.text == TRUNCATION_MARKER
    assert result.truncated is True


def test_the_recipe_fingerprint_inputs_are_untouched_by_the_token_option():
    """Existing indexes must not re-embed.

    ``recipe_parameters`` is hashed verbatim into the manifest's recipe
    fingerprint, so this pins the exact pairs. The token option is deliberately
    absent from them: it belongs to a caller, not to the chunk recipe, and the
    recipe calls the cap with no counter.
    """
    assert recipe_parameters() == (
        ("recipe_version", "source-chunk/1"),
        ("smart_cap_chars", "6000"),
        ("cap_head_share", "0.6800"),
        ("cap_tail_share", "0.2800"),
        ("window_lines", "160"),
        ("window_stride", "120"),
    )


# ---------------------------------------------------------------------------
# File windows
# ---------------------------------------------------------------------------


def test_window_header_names_the_file_and_the_range():
    chunks = list(iter_file_windows("infra/up.sh", "set -e\necho hi\n"))
    assert len(chunks) == 1
    assert chunks[0].text == "# File: infra/up.sh\n# file_window: 1-2\nset -e\necho hi"
    assert chunks[0].chunk_id == "file:infra/up.sh:1-2"
    assert chunks[0].source == SOURCE_FILE_WINDOW
    assert (chunks[0].start_line, chunks[0].end_line) == (1, 2)


def test_windows_overlap_by_the_stride_difference():
    text = "\n".join(f"line{i}" for i in range(1, 401))
    windows = list(iter_file_windows("infra/big.yaml", text))
    bounds = [(w.start_line, w.end_line) for w in windows]
    assert bounds[0] == (1, WINDOW_LINES)
    assert bounds[1] == (1 + WINDOW_STRIDE, WINDOW_STRIDE + WINDOW_LINES)
    assert bounds[-1][1] == 400
    overlap = bounds[0][1] - bounds[1][0] + 1
    assert overlap == WINDOW_LINES - WINDOW_STRIDE


def test_the_last_window_is_short_rather_than_back_extended():
    total = WINDOW_LINES + 10
    text = "\n".join(f"line{i}" for i in range(1, total + 1))
    windows = list(iter_file_windows("a.yml", text))
    assert [(w.start_line, w.end_line) for w in windows] == [
        (1, WINDOW_LINES),
        (WINDOW_STRIDE + 1, total),
    ]
    assert windows[-1].text.splitlines()[-1] == f"line{total}"


def test_an_empty_file_yields_no_windows():
    assert list(iter_file_windows("empty.yaml", "")) == []


def test_a_file_shorter_than_one_window_yields_exactly_one():
    text = "\n".join(f"line{i}" for i in range(1, WINDOW_LINES + 1))
    windows = list(iter_file_windows("a.yml", text))
    assert [(w.start_line, w.end_line) for w in windows] == [(1, WINDOW_LINES)]


# ---------------------------------------------------------------------------
# Window eligibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "infra/up.sh",
        "scripts/run.bash",
        "docker-compose.yaml",
        "config.yml",
        "package.json",
        "pyproject.toml",
        "pom.xml",
        "nginx.conf",
        "units/atlas.service",
        "units/atlas.timer",
        "scripts/setup.ps1",
        "lib/task.rb",
        "views/index.erb",
        "migrations/001.sql",
        "Dockerfile",
        "Dockerfile.gpu",
        "Makefile",
    ],
)
def test_text_formats_are_windowed_even_when_symbols_exist(path):
    assert window_eligible(path, indexed_symbols=0)
    assert window_eligible(path, indexed_symbols=5)


def test_a_code_file_with_symbols_is_not_windowed():
    assert not window_eligible("src/app.py", indexed_symbols=3)


def test_a_code_file_with_no_persisted_symbols_is_windowed():
    """A parse failure or a file of bare statements must still reach the index."""
    assert window_eligible("src/app.py", indexed_symbols=0)
    assert window_eligible("web/main.ts", indexed_symbols=0)


def test_a_file_with_no_grammar_and_no_listed_suffix_is_skipped():
    assert not window_eligible("docs/README.md", indexed_symbols=0)
    assert not window_eligible("assets/logo.png", indexed_symbols=0)


@pytest.mark.parametrize(
    "path",
    ["package-lock.json", "uv.lock", "yarn.lock", "pnpm-lock.yaml", "vendor/poetry.lock"],
)
def test_lockfiles_are_skipped_despite_an_eligible_suffix(path):
    assert not window_eligible(path, indexed_symbols=0)


def test_repowise_own_index_is_never_windowed():
    assert not window_eligible(".repowise/config.yaml", indexed_symbols=0)
    assert not window_eligible("sub/.repowise/state.json", indexed_symbols=0)


def test_test_paths_are_flagged():
    chunk = build_symbol_chunk(_symbol(file_path="tests/test_app.py"), _FILE.splitlines())
    assert chunk.is_test
    assert not build_symbol_chunk(_symbol(), _FILE.splitlines()).is_test


def test_binary_detection_reads_the_first_8kb():
    assert looks_binary(b"ELF\x00\x01")
    assert not looks_binary(b"#!/bin/sh\necho hi\n")
    assert not looks_binary(b"a" * 9000 + b"\x00")
