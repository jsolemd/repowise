"""JSDoc belongs to the declaration inside an export statement."""

from __future__ import annotations

import pytest

from repowise.core.ingestion.parser import ASTParser
from tests.unit.ingestion.parser._helpers import _make_file_info

_LANGUAGES = [("sample.js", "javascript"), ("sample.ts", "typescript"), ("sample.tsx", "typescript")]


@pytest.mark.parametrize(("path", "language"), _LANGUAGES)
@pytest.mark.parametrize("prefix", ["", "export ", "export default "])
@pytest.mark.parametrize(
    "declaration",
    [
        "function named() { return 1; }",
        "async function named() { return 1; }",
        "function* named() { yield 1; }",
        "class named {}",
    ],
)
def test_jsdoc_preceding_a_declaration_or_its_export(
    parser: ASTParser, path: str, language: str, prefix: str, declaration: str,
) -> None:
    source = f"/** Describe the public operation. */\n{prefix}{declaration}\n".encode()
    parsed = parser.parse_file(_make_file_info(path, language), source)
    assert parsed.parse_errors == []
    symbol = next(s for s in parsed.symbols if s.name == "named")
    assert symbol.docstring and symbol.docstring.startswith("Describe the public operation.")
    assert symbol.start_line == 2


@pytest.mark.parametrize("prefix", ["export ", "export default "])
def test_jsdoc_preceding_an_exported_abstract_class(parser: ASTParser, prefix: str) -> None:
    source = f"/** Shared service contract. */\n{prefix}abstract class Service {{}}\n".encode()
    parsed = parser.parse_file(_make_file_info("service.ts", "typescript"), source)
    assert parsed.parse_errors == []
    docstring = next(s for s in parsed.symbols if s.name == "Service").docstring
    assert docstring and docstring.startswith("Shared service contract.")


@pytest.mark.parametrize(("path", "language"), _LANGUAGES)
@pytest.mark.parametrize("prefix", ["", "export ", "export default "])
@pytest.mark.parametrize(
    "leading",
    [
        "/** Another declaration. */\nconst other = 1;\n",
        "/** Detached documentation. */\n// Intervening comment.\n",
        "/* Ordinary block comment. */\n",
        "// Ordinary line comment.\n",
    ],
)
def test_jsdoc_does_not_cross_intervening_nodes_or_accept_ordinary_comments(
    parser: ASTParser, path: str, language: str, prefix: str, leading: str,
) -> None:
    source = f"{leading}{prefix}function named() {{ return 1; }}\n".encode()
    parsed = parser.parse_file(_make_file_info(path, language), source)
    assert parsed.parse_errors == []
    assert next(s for s in parsed.symbols if s.name == "named").docstring is None


@pytest.mark.parametrize(("path", "language"), _LANGUAGES)
@pytest.mark.parametrize("prefix", ["export ", "export default "])
@pytest.mark.parametrize(
    ("comment", "expected"),
    [("/** Closer documentation. */", "Closer documentation."), ("/* Ordinary comment. */", None)],
)
def test_comment_inside_export_takes_precedence_over_outer_jsdoc(
    parser: ASTParser, path: str, language: str, prefix: str, comment: str, expected: str | None,
) -> None:
    source = (
        f"/** Outer documentation. */\n{prefix}{comment} function named() {{ return 1; }}\n"
    ).encode()
    parsed = parser.parse_file(_make_file_info(path, language), source)
    assert parsed.parse_errors == []
    docstring = next(s for s in parsed.symbols if s.name == "named").docstring
    if expected is None:
        assert docstring is None
    else:
        assert docstring and docstring.startswith(expected)


@pytest.mark.parametrize(("path", "language"), _LANGUAGES)
def test_exported_class_jsdoc_does_not_leak_to_undocumented_members(
    parser: ASTParser, path: str, language: str,
) -> None:
    source = b"""/** The exported service. */
export class Service {
  run() { return 1; }
  /** Stop this service. */
  stop() { return 0; }
}
"""
    parsed = parser.parse_file(_make_file_info(path, language), source)
    assert parsed.parse_errors == []
    symbols = {s.name: s for s in parsed.symbols}
    assert symbols["Service"].docstring and symbols["Service"].docstring.startswith("The exported service.")
    assert symbols["run"].docstring is None
    assert symbols["stop"].docstring and symbols["stop"].docstring.startswith("Stop this service.")
