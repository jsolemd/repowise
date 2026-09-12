"""Shared helpers for doc-search tool handlers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Token estimation: ~4 chars per token (conservative)
CHARS_PER_TOKEN = 4


class ToolError(Exception):
    """Error raised by tool handlers."""

    def __init__(self, message: str, code: int = -32602):
        super().__init__(message)
        self.code = code  # JSON-RPC error code


def clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    """Parse an integer argument with bounds."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def get_arg(
    arguments: dict[str, Any],
    key: str,
    *,
    aliases: tuple[str, ...] = (),
    default: Any = None,
) -> Any:
    """Fetch one argument, supporting legacy alias keys."""
    if key in arguments:
        return arguments[key]
    for alias in aliases:
        if alias in arguments:
            return arguments[alias]
    return default


def require_str(arguments: dict[str, Any], key: str, *, aliases: tuple[str, ...] = ()) -> str:
    """Fetch a required non-empty string argument."""
    value = get_arg(arguments, key, aliases=aliases, default="")
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if not value:
        raise ToolError(f"{key} is required")
    return value


def read_text_with_fallback(path: Path) -> str:
    """Read a text file using UTF-8 first, then latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:  # pragma: no cover - defensive fallback
            raise ToolError(f"Cannot read file: {e}") from e
    except Exception as e:  # pragma: no cover - defensive fallback
        raise ToolError(f"Cannot read file: {e}") from e


def resolve_repo_file_path(repo_path: Path, file_path: str) -> Path:
    """Resolve a file path within a cached repository safely."""
    full_path = (repo_path / file_path).resolve()
    repo_path_resolved = repo_path.resolve()
    if not str(full_path).startswith(str(repo_path_resolved)):
        raise ToolError(f"Invalid path: {file_path}")
    return full_path
