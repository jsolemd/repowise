"""Closed-schema validation for the standalone documentation MCP host."""

from __future__ import annotations

import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

_BOOL_TRUE = {"true", "1", "yes", "on"}
_BOOL_FALSE = {"false", "0", "no", "off"}
_INT_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)$")
_NUMBER_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$")
_ALIASES = {
    "libraryName": "library_name",
    "libraryId": "library_id",
    "libraryIds": "library_ids",
    "chunkTypes": "chunk_types",
    "includeCodeBlocks": "include_code_blocks",
    "exactMatch": "exact_match",
    "limitPerLibrary": "limit_per_library",
    "chunkId": "chunk_id",
    "linesBefore": "lines_before",
    "linesAfter": "lines_after",
    "includeStats": "include_stats",
    "maxTokens": "max_tokens",
    "docsPath": "docs_path",
    "includePatterns": "include_patterns",
    "excludePatterns": "exclude_patterns",
    "bundlePath": "bundle_path",
}
_TOOL_ALIASES = {
    "read_doc": {"file_path": "path"},
    "resolve_library_id": {"query": "library_name"},
}


def _apply_aliases(arguments: dict[str, Any], tool_name: str) -> None:
    aliases = {**_ALIASES, **_TOOL_ALIASES.get(tool_name, {})}
    for alias, canonical in aliases.items():
        if alias not in arguments:
            continue
        if canonical not in arguments:
            arguments[canonical] = arguments[alias]
        del arguments[alias]


def _coerce(value: object, expected_type: str) -> tuple[object, bool]:
    if not isinstance(value, str):
        return value, False
    text = value.strip()
    if expected_type == "integer" and _INT_RE.match(text):
        return int(text), True
    if expected_type == "number" and _NUMBER_RE.match(text):
        return (float(text) if "." in text or "e" in text.lower() else int(text)), True
    if expected_type == "boolean":
        lowered = text.lower()
        if lowered in _BOOL_TRUE:
            return True, True
        if lowered in _BOOL_FALSE:
            return False, True
    return value, False


def _coerce_in_place(arguments: dict[str, Any], schema: dict[str, Any]) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return
    for key, specification in properties.items():
        key = str(key)
        if key not in arguments or not isinstance(specification, dict):
            continue
        value = arguments[key]
        expected_type = specification.get("type")
        if expected_type in {"integer", "number", "boolean"}:
            coerced, changed = _coerce(value, str(expected_type))
            if changed:
                arguments[key] = coerced
                value = coerced
        if expected_type == "array" and isinstance(value, list):
            items = specification.get("items")
            item_type = items.get("type") if isinstance(items, dict) else None
            if item_type in {"integer", "number", "boolean"}:
                for index, item in enumerate(value):
                    coerced, changed = _coerce(item, str(item_type))
                    if changed:
                        value[index] = coerced


def _format_error(tool_name: str, error: ValidationError) -> dict[str, Any]:
    details: dict[str, Any] = {
        "tool": tool_name,
        "reason": error.message,
        "path": list(error.absolute_path),
        "schema_keyword": error.validator,
    }
    if error.validator_value is not None:
        details["schema_constraint"] = error.validator_value
    if error.instance is not None:
        details["received_value"] = error.instance
        details["received_type"] = type(error.instance).__name__
    return details


def validate_tool_arguments(
    schemas: dict[str, dict],
    tool_name: str,
    arguments_raw: object,
) -> dict | None:
    """Validate and normalize one tool argument object."""
    schema = schemas.get(tool_name) or {}
    if not isinstance(arguments_raw, dict):
        return {
            "tool": tool_name,
            "reason": "arguments must be an object",
            "expected_type": "object",
            "received_type": type(arguments_raw).__name__,
        }
    _apply_aliases(arguments_raw, tool_name)
    _coerce_in_place(arguments_raw, schema)
    if not schema:
        return None
    errors = sorted(
        Draft202012Validator(schema).iter_errors(arguments_raw),
        key=lambda error: list(error.absolute_path),
    )
    if not errors:
        return None
    formatted = [_format_error(tool_name, error) for error in errors]
    if len(formatted) == 1:
        return formatted[0]
    return {
        "tool": tool_name,
        "reason": f"{len(formatted)} validation errors",
        "errors": formatted,
    }


__all__ = ["validate_tool_arguments"]
