"""AST-based symbol extraction for source chunking.

This module provides lightweight declaration extraction using:
- Python stdlib AST for Python
- tree-sitter-language-pack for JS/TS/TSX/Java/Go/Rust/Kotlin/Scala

All functions are best-effort and return ``None`` when AST parsing is
unavailable or does not produce useful declarations.
"""

from __future__ import annotations

import ast
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# (start_line, symbol_name, symbol_kind)
AstDeclaration = tuple[int, str, str]

# Tree-sitter is optional at import time. The source chunker falls back to
# heuristic segmentation when this dependency is unavailable.
try:
    from tree_sitter_language_pack import get_parser as _ts_get_parser
except Exception:  # pragma: no cover - optional dependency guard
    _ts_get_parser = None


_TS_LANGUAGE_MAP: dict[str, str] = {
    "javascript": "javascript",
    "jsx": "javascript",
    "typescript": "typescript",
    "tsx": "tsx",
    "java": "java",
    "go": "go",
    "rust": "rust",
    "kotlin": "kotlin",
    "scala": "scala",
}

_TS_SYMBOL_NODES: dict[str, str] = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "type_alias_declaration": "type",
    "function_declaration": "function",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "method_definition": "method",
    "method_signature": "method",
    "abstract_method_signature": "method",
    "type_spec": "type",
    "struct_item": "class",
    "trait_item": "interface",
    "enum_item": "enum",
    "type_item": "type",
    "function_signature_item": "method",
    "class_definition": "class",
    "trait_definition": "interface",
    "object_definition": "class",
}

_TS_FUNCTION_VALUE_TYPES: set[str] = {
    "arrow_function",
    "function",
    "function_expression",
    "generator_function",
    "generator_function_declaration",
}

_SYMBOL_KIND_ORDER: dict[str, int] = {
    "class": 0,
    "interface": 0,
    "enum": 0,
    "record": 0,
    "type": 1,
    "constructor": 2,
    "method": 2,
    "function": 3,
}

_RUST_METHOD_PARENT_NODES: set[str] = {"impl_item", "trait_item"}
_KOTLIN_METHOD_PARENT_NODES: set[str] = {"class_body"}
_SCALA_METHOD_PARENT_NODES: set[str] = {"template_body"}

_KOTLIN_CLASS_NAME_CHILD_TYPES: tuple[str, ...] = ("type_identifier", "simple_identifier")


@lru_cache(maxsize=8)
def _get_tree_sitter_parser(language: str):
    """Get cached tree-sitter parser for a language, if available."""
    if _ts_get_parser is None:
        return None

    try:
        return _ts_get_parser(language)
    except Exception as exc:  # pragma: no cover - depends on runtime package availability
        logger.debug("tree-sitter parser unavailable for %s: %s", language, exc)
        return None


def _normalize_name(name: str) -> str:
    normalized = " ".join(name.strip().split())
    if not normalized:
        return ""
    return normalized[:120]


def _node_text(content_bytes: bytes, node) -> str:
    return content_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _sort_declarations(declarations: list[AstDeclaration]) -> list[AstDeclaration]:
    return sorted(
        declarations,
        key=lambda d: (d[0], _SYMBOL_KIND_ORDER.get(d[2], 9), d[1]),
    )


def _has_ancestor_type(node, ancestor_types: set[str], max_depth: int = 12) -> bool:
    parent = node.parent
    depth = 0
    while parent is not None and depth < max_depth:
        if parent.type in ancestor_types:
            return True
        parent = parent.parent
        depth += 1
    return False


def _first_child_text_by_type(content_bytes: bytes, node, child_types: tuple[str, ...]) -> str:
    for child in node.children:
        if child.type in child_types:
            symbol_name = _normalize_name(_node_text(content_bytes, child))
            if symbol_name:
                return symbol_name
    return ""


def _resolve_kotlin_symbol(content_bytes: bytes, node) -> tuple[str, str] | None:
    node_type = node.type

    if node_type == "class_declaration":
        symbol_name = _first_child_text_by_type(content_bytes, node, _KOTLIN_CLASS_NAME_CHILD_TYPES)
        if not symbol_name:
            return None

        if any(child.type == "interface" for child in node.children):
            return (symbol_name, "interface")
        if any(child.type == "enum" for child in node.children):
            return (symbol_name, "enum")
        return (symbol_name, "class")

    if node_type == "object_declaration":
        symbol_name = _first_child_text_by_type(content_bytes, node, _KOTLIN_CLASS_NAME_CHILD_TYPES)
        if symbol_name:
            return (symbol_name, "class")
        return None

    if node_type == "function_declaration":
        symbol_name = _first_child_text_by_type(
            content_bytes,
            node,
            ("simple_identifier", "identifier"),
        )
        if not symbol_name:
            return None

        kind = "method" if _has_ancestor_type(node, _KOTLIN_METHOD_PARENT_NODES) else "function"
        return (symbol_name, kind)

    return None


def _resolve_scala_symbol(content_bytes: bytes, node) -> tuple[str, str] | None:
    node_type = node.type
    name_node = node.child_by_field_name("name")

    if node_type in {"class_definition", "trait_definition", "object_definition"}:
        if name_node is None:
            return None
        symbol_name = _normalize_name(_node_text(content_bytes, name_node))
        if not symbol_name:
            return None
        if node_type == "trait_definition":
            return (symbol_name, "interface")
        return (symbol_name, "class")

    if node_type in {"function_definition", "function_declaration"}:
        if name_node is None:
            return None
        symbol_name = _normalize_name(_node_text(content_bytes, name_node))
        if not symbol_name:
            return None
        kind = "method" if _has_ancestor_type(node, _SCALA_METHOD_PARENT_NODES) else "function"
        return (symbol_name, kind)

    return None


def _extract_python_declarations(content: str) -> list[AstDeclaration] | None:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    declarations: list[AstDeclaration] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_depth = 0

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            declarations.append((max(node.lineno - 1, 0), node.name, "class"))
            self.class_depth += 1
            self.generic_visit(node)
            self.class_depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            kind = "method" if self.class_depth > 0 else "function"
            declarations.append((max(node.lineno - 1, 0), node.name, kind))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            kind = "method" if self.class_depth > 0 else "function"
            declarations.append((max(node.lineno - 1, 0), node.name, kind))
            self.generic_visit(node)

    Visitor().visit(tree)

    if not declarations:
        return None

    return _sort_declarations(declarations)


def _extract_tree_sitter_declarations(content: str, language: str) -> list[AstDeclaration] | None:
    parser = _get_tree_sitter_parser(language)
    if parser is None:
        return None

    content_bytes = content.encode("utf-8", errors="ignore")
    tree = parser.parse(content_bytes)
    root = tree.root_node

    declarations: list[AstDeclaration] = []
    stack = [root]

    while stack:
        node = stack.pop()
        node_type = node.type
        resolved: tuple[str, str] | None = None

        if language == "kotlin":
            resolved = _resolve_kotlin_symbol(content_bytes, node)
        elif language == "scala":
            resolved = _resolve_scala_symbol(content_bytes, node)

        if resolved is not None:
            symbol_name, kind = resolved
            declarations.append((node.start_point[0], symbol_name, kind))
        elif node_type in _TS_SYMBOL_NODES or node_type == "function_item":
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                symbol_name = _normalize_name(_node_text(content_bytes, name_node))
                if symbol_name:
                    if node_type == "function_item":
                        kind = "function"
                        if language == "rust" and _has_ancestor_type(
                            node, _RUST_METHOD_PARENT_NODES
                        ):
                            kind = "method"
                    else:
                        kind = _TS_SYMBOL_NODES[node_type]

                    declarations.append((node.start_point[0], symbol_name, kind))

        elif node_type == "variable_declarator":
            value_node = node.child_by_field_name("value")
            if value_node is not None and value_node.type in _TS_FUNCTION_VALUE_TYPES:
                name_node = node.child_by_field_name("name")
                if name_node is not None:
                    symbol_name = _normalize_name(_node_text(content_bytes, name_node))
                    if symbol_name and "{" not in symbol_name:
                        declarations.append((node.start_point[0], symbol_name, "function"))

        stack.extend(reversed(node.children))

    if not declarations:
        return None

    return _sort_declarations(declarations)


def extract_ast_declarations(content: str, language: str) -> list[AstDeclaration] | None:
    """Extract declarations for a supported language using AST parsing.

    Returns:
        Sorted declarations as ``(line, name, kind)`` or ``None`` when AST
        extraction is unavailable/unsupported.
    """
    normalized = language.lower()

    if normalized == "python":
        return _extract_python_declarations(content)

    ts_language = _TS_LANGUAGE_MAP.get(normalized)
    if ts_language is None:
        return None

    return _extract_tree_sitter_declarations(content, ts_language)
