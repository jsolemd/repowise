"""Shared file-format registry for chunking, classification, and indexing defaults."""

from __future__ import annotations

from pathlib import Path

# Documentation formats
MARKDOWN_EXTENSIONS: set[str] = {".md", ".markdown", ".mdown", ".mkd", ".mkdn"}
MDX_EXTENSIONS: set[str] = {".mdx"}
ASCIIDOC_EXTENSIONS: set[str] = {".adoc", ".asciidoc", ".asc"}
RST_EXTENSIONS: set[str] = {".rst", ".rest"}
TEXT_DOC_EXTENSIONS: set[str] = {".txt", ".text", ".org"}
HTML_DOC_EXTENSIONS: set[str] = {".html", ".htm", ".xhtml"}
XML_DOC_EXTENSIONS: set[str] = {".xml"}
NOTEBOOK_EXTENSIONS: set[str] = {".ipynb"}

DOC_EXTENSIONS: set[str] = (
    MARKDOWN_EXTENSIONS
    | MDX_EXTENSIONS
    | ASCIIDOC_EXTENSIONS
    | RST_EXTENSIONS
    | TEXT_DOC_EXTENSIONS
    | HTML_DOC_EXTENSIONS
    | XML_DOC_EXTENSIONS
    | NOTEBOOK_EXTENSIONS
)

# Source code formats
SOURCE_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".groovy": "groovy",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".lua": "lua",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".fish": "fish",
    ".ps1": "powershell",
}

SOURCE_SUFFIX_LANGUAGE: dict[str, str] = {
    ".d.ts": "typescript",
}

SOURCE_EXTENSIONS: set[str] = set(SOURCE_LANGUAGE_BY_EXTENSION)

# Configuration and build formats
CONFIG_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".json5": "json5",
    ".jsonc": "jsonc",
    ".ini": "ini",
    ".cfg": "config",
    ".conf": "config",
    ".properties": "properties",
    ".gradle": "gradle",
    ".hcl": "hcl",
    ".tf": "terraform",
    ".tfvars": "terraform",
}

CONFIG_SUFFIX_LANGUAGE: dict[str, str] = {
    ".gradle.kts": "gradle",
}

CONFIG_FILENAME_LANGUAGE: dict[str, str] = {
    "dockerfile": "dockerfile",
    "docker-compose.yml": "yaml",
    "docker-compose.yaml": "yaml",
    "compose.yml": "yaml",
    "compose.yaml": "yaml",
    "makefile": "make",
    "cmakelists.txt": "cmake",
    "justfile": "just",
    "procfile": "procfile",
    "requirements.txt": "requirements",
    "package-lock.json": "lockfile",
    "pnpm-lock.yaml": "lockfile",
    "yarn.lock": "lockfile",
    "cargo.toml": "toml",
    "cargo.lock": "lockfile",
    "go.mod": "gomod",
    "go.sum": "gosum",
    "pyproject.toml": "toml",
    "setup.cfg": "ini",
    "tsconfig.json": "json",
    "webpack.config.js": "javascript",
    "vite.config.ts": "typescript",
}

CONFIG_EXTENSIONS: set[str] = set(CONFIG_LANGUAGE_BY_EXTENSION)
CONFIG_FILENAMES: set[str] = set(CONFIG_FILENAME_LANGUAGE)


def _glob_patterns_from_extensions(extensions: set[str]) -> list[str]:
    return [f"**/*{ext}" for ext in sorted(extensions)]


def _glob_patterns_from_suffixes(suffixes: dict[str, str]) -> list[str]:
    return [f"**/*{suffix}" for suffix in sorted(suffixes)]


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


# Default include patterns for newly registered libraries.
# These are intentionally text-first and can be overridden per library.
DEFAULT_INCLUDE_PATTERNS: list[str] = _unique_keep_order(
    _glob_patterns_from_extensions(DOC_EXTENSIONS)
    + _glob_patterns_from_extensions(SOURCE_EXTENSIONS)
    + _glob_patterns_from_suffixes(SOURCE_SUFFIX_LANGUAGE)
    + _glob_patterns_from_extensions(CONFIG_EXTENSIONS)
    + _glob_patterns_from_suffixes(CONFIG_SUFFIX_LANGUAGE)
    + [
        "**/Dockerfile",
        "**/docker-compose.yml",
        "**/docker-compose.yaml",
        "**/compose.yml",
        "**/compose.yaml",
        "**/Makefile",
        "**/CMakeLists.txt",
        "**/Justfile",
        "**/Procfile",
        "**/.env",
        "**/.env.*",
        "**/.editorconfig",
        "**/.gitignore",
        "**/.gitattributes",
    ]
)

# Patterns used to discover likely docs roots.
DOC_DISCOVERY_PATTERNS: list[str] = [
    "**/*.md",
    "**/*.mdx",
    "**/*.rst",
    "**/*.adoc",
    "**/*.asciidoc",
    "**/*.txt",
    "**/*.html",
    "**/*.htm",
]

# Generated directories that should not be indexed by default.
AUTO_EXCLUDE_PATTERNS: list[str] = [
    "**/.git/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/__tests__/**",
    "**/__mocks__/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.ruff_cache/**",
    "**/.tox/**",
    "**/.venv/**",
    "**/venv/**",
    "**/test/**",
    "**/tests/**",
    "**/testing/**",
    "**/cypress/**",
    "**/playwright/**",
    "**/fixtures/**",
    "**/fixture/**",
    "**/mocks/**",
    "**/mock-data/**",
    "**/dist/**",
    "**/build/**",
    "**/target/**",
    "**/.next/**",
    "**/.nuxt/**",
    "**/.output/**",
    "**/coverage/**",
    "**/_static/**",
    "**/*.story.*",
    "**/*.stories.*",
    "**/*.min.js",
    "**/*.min.css",
]


def _normalize_path(file_path: str) -> str:
    return file_path.replace("\\", "/").lower()


def file_extension(file_path: str) -> str:
    """Return lowercase suffix for a file path."""
    return Path(file_path).suffix.lower()


def detect_source_language(file_path: str) -> str | None:
    """Infer source language from file path."""
    path_lower = _normalize_path(file_path)

    for suffix, language in SOURCE_SUFFIX_LANGUAGE.items():
        if path_lower.endswith(suffix):
            return language

    ext = Path(path_lower).suffix
    return SOURCE_LANGUAGE_BY_EXTENSION.get(ext)


def detect_config_language(file_path: str) -> str | None:
    """Infer config/build language from file path."""
    path_lower = _normalize_path(file_path)
    filename = Path(path_lower).name

    if filename.startswith(".env"):
        return "dotenv"

    if filename in CONFIG_FILENAME_LANGUAGE:
        return CONFIG_FILENAME_LANGUAGE[filename]

    for suffix, language in CONFIG_SUFFIX_LANGUAGE.items():
        if path_lower.endswith(suffix):
            return language

    ext = Path(path_lower).suffix
    return CONFIG_LANGUAGE_BY_EXTENSION.get(ext)
