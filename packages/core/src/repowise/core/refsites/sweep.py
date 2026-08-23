"""A comment- and string-aware lexer used to place occurrences in source text.

Two jobs, both of which need to know where the code *isn't*:

1. **Column resolution for AST sites.** The parse layer hands back a line and
   a name, never a column. To turn that into a range a rewriter could use, we
   have to find the identifier on that line — and skip the copy of it sitting
   in a trailing comment.
2. **The textual sweep.** The parse layer drops occurrences: it dedups call
   sites by ``(line, target, receiver)``, so two calls to the same function on
   one line become one, and it deletes symbols declared inside anonymous
   callables outright. The sweep recovers those positions. It is bounded by
   the AST-derived name universe — it can only find names the parser already
   saw somewhere — so it cannot invent a reference, only relocate one.

This is a lexer, not a parser. It knows comments, string literals and
identifiers, and nothing else. Everything it produces is scored on the two
textual floors of the rubric precisely because a lexer cannot tell a
reference from a coincidence.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

_IDENT_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_CONT = _IDENT_START | frozenset("0123456789")

#: Region a token was found in. ``code`` is a real program position;
#: ``string`` is inside a literal; ``comment`` is prose.
REGION_CODE = "code"
REGION_STRING = "string"
REGION_COMMENT = "comment"


@dataclass(frozen=True, slots=True)
class LexConfig:
    """The little that has to be true about a language to lex it safely."""

    line_comment: tuple[str, ...] = ()
    block_comment: tuple[str, str] | None = None
    quotes: tuple[str, ...] = ()
    triple_quotes: tuple[str, ...] = ()
    #: Backslash escapes inside string literals.
    escapes: bool = True
    #: Extra characters legal in an identifier beyond ``[A-Za-z0-9_]``.
    ident_extra: str = ""
    #: Delimiters whose bodies contain code — JS/TS template interpolation.
    interpolating_quotes: tuple[str, ...] = ()
    #: Shell's ``#`` only opens a comment at a word boundary, so ``a#b`` is one
    #: word. Python's ``#`` opens one anywhere outside a literal.
    comment_needs_word_boundary: bool = False


_PY = LexConfig(
    line_comment=("#",),
    quotes=('"', "'"),
    triple_quotes=('"""', "'''"),
)
_TS = LexConfig(
    line_comment=("//",),
    block_comment=("/*", "*/"),
    quotes=('"', "'", "`"),
    ident_extra="$",
    interpolating_quotes=("`",),
)
_SH = LexConfig(
    line_comment=("#",),
    quotes=('"', "'"),
    comment_needs_word_boundary=True,
)
_SQL = LexConfig(
    line_comment=("--",),
    block_comment=("/*", "*/"),
    quotes=("'", '"'),
    escapes=False,
)

_CONFIGS: dict[str, LexConfig] = {
    "python": _PY,
    "typescript": _TS,
    "javascript": _TS,
    "shell": _SH,
    "sql": _SQL,
}


def lex_config_for(language: str) -> LexConfig | None:
    """Return the lexer config for *language*, or ``None`` if we cannot lex it.

    ``None`` is a refusal, not a default. Guessing a comment syntax is how a
    sweep starts reporting identifiers inside prose as references.
    """
    return _CONFIGS.get(language)


@dataclass(frozen=True, slots=True)
class Token:
    """An identifier, or the body of a string literal."""

    kind: str  # "identifier" | "string"
    text: str
    line: int  # 1-indexed
    end_line: int
    start_col: int  # 0-indexed, exclusive end
    end_col: int
    region: str


class _Cursor:
    """Position tracking shared by the scan loop."""

    __slots__ = ("col", "line", "pos")

    def __init__(self) -> None:
        self.pos = 0
        self.line = 1
        self.col = 0

    def advance(self, source: str, count: int) -> None:
        for _ in range(count):
            if self.pos >= len(source):
                return
            if source[self.pos] == "\n":
                self.line += 1
                self.col = 0
            else:
                self.col += 1
            self.pos += 1


def _starts_with(source: str, pos: int, needle: str) -> bool:
    return source.startswith(needle, pos)


def _read_identifier(source: str, cursor: _Cursor, cfg: LexConfig, region: str) -> Token:
    start_pos, start_line, start_col = cursor.pos, cursor.line, cursor.col
    extra = frozenset(cfg.ident_extra)
    length = 0
    while start_pos + length < len(source):
        char = source[start_pos + length]
        if char in _IDENT_CONT or char in extra:
            length += 1
        else:
            break
    cursor.advance(source, length)
    return Token(
        kind="identifier",
        text=source[start_pos : start_pos + length],
        line=start_line,
        end_line=start_line,
        start_col=start_col,
        end_col=start_col + length,
        region=region,
    )


def _scan_string(source: str, cursor: _Cursor, cfg: LexConfig, delim: str) -> Iterator[Token]:
    """Consume a string literal, yielding its body and any interpolated code."""
    cursor.advance(source, len(delim))
    body_start_pos, body_start_line, body_start_col = cursor.pos, cursor.line, cursor.col
    interpolating = delim in cfg.interpolating_quotes
    inner: list[Token] = []

    while cursor.pos < len(source):
        char = source[cursor.pos]
        if cfg.escapes and char == "\\":
            cursor.advance(source, 2)
            continue
        if interpolating and _starts_with(source, cursor.pos, "${"):
            cursor.advance(source, 2)
            depth = 1
            while cursor.pos < len(source) and depth:
                inner_char = source[cursor.pos]
                if inner_char == "{":
                    depth += 1
                    cursor.advance(source, 1)
                elif inner_char == "}":
                    depth -= 1
                    cursor.advance(source, 1)
                elif inner_char in _IDENT_START:
                    inner.append(_read_identifier(source, cursor, cfg, REGION_CODE))
                else:
                    cursor.advance(source, 1)
            continue
        if _starts_with(source, cursor.pos, delim):
            break
        if len(delim) == 1 and char == "\n":
            # An unterminated single-quoted literal. Stop at the newline rather
            # than swallowing the rest of the file as one string.
            break
        cursor.advance(source, 1)

    body_end_pos, body_end_line, body_end_col = cursor.pos, cursor.line, cursor.col
    if _starts_with(source, cursor.pos, delim):
        cursor.advance(source, len(delim))

    yield Token(
        kind="string",
        text=source[body_start_pos:body_end_pos],
        line=body_start_line,
        end_line=body_end_line,
        start_col=body_start_col,
        end_col=body_end_col,
        region=REGION_STRING,
    )
    yield from inner


def _skip_line_comment(source: str, cursor: _Cursor, cfg: LexConfig) -> Iterator[Token]:
    while cursor.pos < len(source) and source[cursor.pos] != "\n":
        if source[cursor.pos] in _IDENT_START:
            yield _read_identifier(source, cursor, cfg, REGION_COMMENT)
        else:
            cursor.advance(source, 1)


def _skip_block_comment(source: str, cursor: _Cursor, cfg: LexConfig) -> Iterator[Token]:
    assert cfg.block_comment is not None
    close = cfg.block_comment[1]
    cursor.advance(source, len(cfg.block_comment[0]))
    while cursor.pos < len(source) and not _starts_with(source, cursor.pos, close):
        if source[cursor.pos] in _IDENT_START:
            yield _read_identifier(source, cursor, cfg, REGION_COMMENT)
        else:
            cursor.advance(source, 1)
    cursor.advance(source, len(close))


def _at_word_boundary(source: str, pos: int) -> bool:
    return pos == 0 or source[pos - 1] in " \t\n;|&(" or source[pos - 1] == "\r"


def scan(source: str, language: str) -> Iterator[Token]:
    """Yield every identifier and string body in *source*, tagged by region.

    Yields nothing for a language with no lexer config: refusing is the point.
    """
    cfg = lex_config_for(language)
    if cfg is None:
        return
    cursor = _Cursor()
    while cursor.pos < len(source):
        char = source[cursor.pos]

        if cfg.block_comment is not None and _starts_with(source, cursor.pos, cfg.block_comment[0]):
            yield from _skip_block_comment(source, cursor, cfg)
            continue

        opened_comment = False
        for marker in cfg.line_comment:
            if _starts_with(source, cursor.pos, marker) and (
                not cfg.comment_needs_word_boundary or _at_word_boundary(source, cursor.pos)
            ):
                yield from _skip_line_comment(source, cursor, cfg)
                opened_comment = True
                break
        if opened_comment:
            continue

        triple = next((t for t in cfg.triple_quotes if _starts_with(source, cursor.pos, t)), None)
        if triple is not None:
            yield from _scan_string(source, cursor, cfg, triple)
            continue

        quote = next((q for q in cfg.quotes if _starts_with(source, cursor.pos, q)), None)
        if quote is not None:
            yield from _scan_string(source, cursor, cfg, quote)
            continue

        if char in _IDENT_START or char in cfg.ident_extra:
            yield _read_identifier(source, cursor, cfg, REGION_CODE)
            continue

        cursor.advance(source, 1)


def code_identifier_positions(source: str, language: str) -> dict[str, list[tuple[int, int]]]:
    """Map identifier text -> ``[(line, start_col), ...]`` for code regions only.

    Comment and string occurrences are excluded; callers that want string
    literals use :func:`string_literals` so the two never get conflated.
    """
    positions: dict[str, list[tuple[int, int]]] = {}
    for token in scan(source, language):
        if token.kind == "identifier" and token.region == REGION_CODE:
            positions.setdefault(token.text, []).append((token.line, token.start_col))
    return positions


def string_literals(source: str, language: str) -> list[Token]:
    """Every string-literal body in *source*."""
    return [token for token in scan(source, language) if token.kind == "string"]


__all__ = [
    "REGION_CODE",
    "REGION_COMMENT",
    "REGION_STRING",
    "LexConfig",
    "Token",
    "code_identifier_positions",
    "lex_config_for",
    "scan",
    "string_literals",
]
