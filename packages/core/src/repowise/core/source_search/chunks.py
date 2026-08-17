"""The chunk recipe: what a unit of retrievable source text looks like.

Two lanes, one shape. A **symbol chunk** is one persisted ``wiki_symbols``
row rendered with a provenance header and its body sliced from the live file,
so a hit carries the lines a reader would open. A **file window** covers what
the symbol lane cannot: shell scripts, compose files, systemd units and the
odd code file whose grammar produced nothing. Windows overlap by a third of
their length, so a fact never falls in the seam between two of them.

Everything here is pure — no database, no disk, no network. The caller reads
the bytes; this module decides what text goes in the index.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import PurePosixPath

__all__ = [
    "MAX_WINDOW_FILE_BYTES",
    "RECIPE_VERSION",
    "SMART_CAP_CHARS",
    "SOURCE_FILE_WINDOW",
    "SOURCE_SYMBOL",
    "TRUNCATION_MARKER",
    "WINDOW_LINES",
    "WINDOW_STRIDE",
    "SourceChunk",
    "SymbolRecord",
    "apply_smart_cap",
    "build_symbol_chunk",
    "chunk_content_hash",
    "iter_file_windows",
    "language_for_path",
    "looks_binary",
    "recipe_parameters",
    "window_eligible",
]

#: Bumped whenever the *text* a chunk carries changes shape. It is hashed into
#: the manifest's recipe fingerprint, so a bump invalidates every stored vector
#: rather than leaving an index that is half one recipe and half another.
RECIPE_VERSION = "source-chunk/1"

#: Longest chunk text kept whole. Past it the middle is dropped rather than the
#: tail: the head of a symbol carries its signature and the top of its body,
#: and the tail carries the return and the error paths, so cutting at the cap
#: would throw away the half that answers "what does this end up doing".
SMART_CAP_CHARS = 6000
_CAP_HEAD_SHARE = 0.68
_CAP_TAIL_SHARE = 0.28
TRUNCATION_MARKER = "\n# ... smart truncation ...\n"

#: File-window geometry. The 40-line overlap (160 - 120) is what stops a
#: service definition or a function that straddles a boundary from reaching the
#: index only in halves.
WINDOW_LINES = 160
WINDOW_STRIDE = 120

#: Which lane a chunk came from. Stored on the row so a retriever can tell a
#: symbol body apart from a slice of a config file.
SOURCE_SYMBOL = "symbol"
SOURCE_FILE_WINDOW = "file_window"

#: Files past this are not windowed. A megabyte of one file is either generated
#: or data, and either way it would flood the corpus with near-identical rows.
MAX_WINDOW_FILE_BYTES = 1024 * 1024

# Text formats the wiki pipeline has no grammar for and typically excludes, but
# which carry the operational half of an infrastructure repo: what a service
# runs, which port it binds, what a script does on failure. Suffixes are matched
# case-insensitively; ``Dockerfile``/``Makefile`` have no suffix at all, and
# ``Dockerfile.gpu``-style variants are matched on the stem.
_WINDOW_SUFFIXES: frozenset[str] = frozenset(
    {
        ".sh",
        ".bash",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".xml",
        ".conf",
        ".service",
        ".timer",
        ".ps1",
        ".rb",
        ".erb",
        ".sql",
    }
)
_WINDOW_BASENAMES: frozenset[str] = frozenset({"Dockerfile", "Makefile"})

# Machine-written dependency resolutions. Eligible by suffix (``.json``,
# ``.toml``) and worthless in a retrieval corpus: thousands of lines of pinned
# hashes that match every query about a package name and answer none of them.
_LOCKFILE_NAMES: frozenset[str] = frozenset(
    {
        "Cargo.lock",
        "Gemfile.lock",
        "composer.lock",
        "package-lock.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "yarn.lock",
    }
)

#: repowise's own index directory. It sits inside the repo and git may well be
#: tracking parts of it, but indexing the index is circular.
_SELF_DIRNAME = ".repowise"


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    """The persisted-symbol fields the recipe reads, and nothing else.

    A plain record rather than the ORM row, so the recipe stays testable
    without a database and cannot accidentally lazy-load a relationship while
    building text.
    """

    symbol_id: str
    file_path: str
    name: str
    qualified_name: str
    kind: str
    signature: str
    docstring: str | None
    start_line: int
    end_line: int
    language: str


@dataclass(frozen=True, slots=True)
class SourceChunk:
    """One retrievable unit of source text, plus what a ranker needs about it."""

    chunk_id: str
    text: str
    file_path: str
    name: str
    qualified_name: str
    kind: str
    start_line: int
    end_line: int
    language: str
    is_test: bool
    source: str
    content_hash: str


def recipe_parameters() -> tuple[tuple[str, str], ...]:
    """Every recipe knob that changes chunk text, as ``(name, value)`` pairs.

    Consumed by :func:`repowise.core.source_search.manifest.recipe_fingerprint`.
    It lives here because this module owns the values: a knob added above and
    forgotten in the fingerprint would let a rebuilt index silently mix two
    recipes, which is the one failure the fingerprint exists to prevent.
    """
    return (
        ("recipe_version", RECIPE_VERSION),
        ("smart_cap_chars", str(SMART_CAP_CHARS)),
        ("cap_head_share", f"{_CAP_HEAD_SHARE:.4f}"),
        ("cap_tail_share", f"{_CAP_TAIL_SHARE:.4f}"),
        ("window_lines", str(WINDOW_LINES)),
        ("window_stride", str(WINDOW_STRIDE)),
    )


def chunk_content_hash(text: str) -> str:
    """SHA-256 of a chunk's text, hex.

    Deliberately not :func:`repowise.core.ingestion.models.compute_content_hash`,
    which is the same digest over a different thing (a whole file's bytes):
    reaching it means importing ``repowise.core.ingestion``, whose package init
    pulls the parser, the graph builder and the change detector onto what is
    otherwise a stdlib-only module. Same algorithm, so the two digests remain
    comparable if a caller ever wants to.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_smart_cap(text: str, cap: int = SMART_CAP_CHARS) -> str:
    """Hold *text* to *cap* characters by dropping its middle, not its tail.

    Under the cap the text is returned unchanged, so the common case is
    identity. Over it, the head and tail shares are taken of the *cap* rather
    than of the input, which is what makes the output length a constant.
    """
    if len(text) <= cap:
        return text
    head = int(cap * _CAP_HEAD_SHARE)
    tail = int(cap * _CAP_TAIL_SHARE)
    return text[:head] + TRUNCATION_MARKER + text[-tail:]


def looks_binary(data: bytes) -> bool:
    """A NUL byte in the first 8 KB, the rule ``ingestion.traverser`` uses.

    Restated on bytes the caller already holds rather than imported: the
    traverser's version is private, takes a path, and re-opens the file.
    """
    return b"\x00" in data[:8192]


@cache
def _code_language_tags() -> frozenset[str]:
    """Language tags that have a tree-sitter grammar behind them.

    Resolved on first use, not at import: the registry lives under
    ``ingestion/``, and importing anything from there runs that package's init.
    :mod:`repowise.core.test_paths` defers for the same reason.
    """
    from ..ingestion.languages.registry import REGISTRY

    return REGISTRY.code_languages()


@cache
def _language_lookup() -> tuple[dict[str, str], dict[str, str]]:
    """``({ext: tag}, {filename: tag})`` from the language registry."""
    from ..ingestion.languages.registry import REGISTRY

    return REGISTRY.all_extensions(), REGISTRY.all_special_filenames()


def language_for_path(rel_path: str) -> str:
    """The registry's language tag for *rel_path*, or ``""`` if it has none."""
    name = PurePosixPath(rel_path).name
    by_ext, by_name = _language_lookup()
    tag = by_name.get(name) or by_ext.get(PurePosixPath(name).suffix.lower())
    return tag or ""


def _is_window_text_format(rel_path: str) -> bool:
    """Whether *rel_path* is one of the text formats the symbol lane never sees."""
    name = PurePosixPath(rel_path).name
    if name in _WINDOW_BASENAMES:
        return True
    if any(name.startswith(f"{stem}.") for stem in _WINDOW_BASENAMES):
        return True
    return PurePosixPath(name).suffix.lower() in _WINDOW_SUFFIXES


def window_eligible(rel_path: str, *, indexed_symbols: int) -> bool:
    """Whether the window lane should cover *rel_path*.

    Two arms. A text format the wiki pipeline has no grammar for is covered
    unconditionally — that is the whole point of this lane. A file that *does*
    have a grammar is covered only when the symbol lane came back empty for it,
    which is how a parse failure, an unsupported dialect or a file of bare
    statements still reaches the index instead of vanishing.

    *indexed_symbols* is how many chunks the symbol lane produced for this
    path, so a file the wiki pipeline never walked (excluded by config, which
    is the normal case for the formats above) arrives here with zero and is
    judged on its own shape rather than on the pipeline's exclusion list.
    """
    parts = PurePosixPath(rel_path).parts
    if _SELF_DIRNAME in parts:
        return False
    if PurePosixPath(rel_path).name in _LOCKFILE_NAMES:
        return False
    if _is_window_text_format(rel_path):
        return True
    return indexed_symbols == 0 and language_for_path(rel_path) in _code_language_tags()


def build_symbol_chunk(symbol: SymbolRecord, lines: Sequence[str]) -> SourceChunk:
    """Render one persisted symbol as a chunk.

    *lines* is the containing file's text already split into lines; the body is
    ``lines[start_line - 1 : end_line]``, the 1-indexed-inclusive slice
    ``ContextAssembler.assemble_symbol_spotlight`` uses. A row whose bounds
    were never populated (both default to 0) yields a header-only chunk rather
    than a slice from the top of the file.
    """
    start = max(symbol.start_line, 1)
    end = max(symbol.end_line, start)
    parts = [
        f"# File: {symbol.file_path}",
        f"# {symbol.kind}: {symbol.qualified_name or symbol.name}",
        f"# Lines: {symbol.start_line}-{symbol.end_line}",
    ]
    signature = (symbol.signature or "").strip()
    if signature:
        parts.append(signature)
    docstring = (symbol.docstring or "").strip()
    if docstring:
        parts.append(f'"""{docstring}"""')
    if symbol.end_line >= 1:
        parts.extend(lines[start - 1 : end])
    text = apply_smart_cap("\n".join(parts))
    return SourceChunk(
        chunk_id=symbol.symbol_id,
        text=text,
        file_path=symbol.file_path,
        name=symbol.name,
        qualified_name=symbol.qualified_name,
        kind=symbol.kind,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        language=symbol.language,
        is_test=_is_test(symbol.file_path, symbol.language),
        source=SOURCE_SYMBOL,
        content_hash=chunk_content_hash(text),
    )


def iter_file_windows(
    rel_path: str, text: str, language: str | None = None
) -> Iterator[SourceChunk]:
    """Cut *text* into overlapping line windows, one chunk each.

    An empty file yields nothing. The last window is short rather than
    back-extended, so a chunk's declared line range is always the range it
    actually holds.
    """
    lines = text.splitlines()
    if not lines:
        return
    tag = language if language is not None else language_for_path(rel_path)
    name = PurePosixPath(rel_path).name
    is_test = _is_test(rel_path, tag)
    total = len(lines)
    for start in range(1, total + 1, WINDOW_STRIDE):
        end = min(start + WINDOW_LINES - 1, total)
        body = [f"# File: {rel_path}", f"# file_window: {start}-{end}"]
        body.extend(lines[start - 1 : end])
        window_text = apply_smart_cap("\n".join(body))
        yield SourceChunk(
            chunk_id=f"file:{rel_path}:{start}-{end}",
            text=window_text,
            file_path=rel_path,
            name=name,
            qualified_name=rel_path,
            kind=SOURCE_FILE_WINDOW,
            start_line=start,
            end_line=end,
            language=tag or "",
            is_test=is_test,
            source=SOURCE_FILE_WINDOW,
            content_hash=chunk_content_hash(window_text),
        )
        if end >= total:
            return


def _is_test(rel_path: str, language: str | None) -> bool:
    """Whether a chunk from *rel_path* is test material rather than product code.

    Delegates to :mod:`repowise.core.test_paths`, the single home for this
    question (#1103). ``is_test_related_path`` rather than ``is_test_path``
    because the flag is a "not production code" signal for a later ranker, and
    a fixture module is not production code even though it is not itself a test.
    """
    from ..test_paths import is_test_related_path

    return is_test_related_path(rel_path, language or None)
