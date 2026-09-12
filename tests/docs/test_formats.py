"""Tests for shared format registry helpers."""

from repowise.docs.formats import (
    DEFAULT_INCLUDE_PATTERNS,
    detect_config_language,
    detect_source_language,
)


def test_detect_source_language_for_common_extensions():
    assert detect_source_language("src/app.py") == "python"
    assert detect_source_language("src/index.ts") == "typescript"
    assert detect_source_language("src/types.d.ts") == "typescript"


def test_detect_config_language_for_special_files():
    assert detect_config_language("Dockerfile") == "dockerfile"
    assert detect_config_language("settings.gradle.kts") == "gradle"
    assert detect_config_language(".env.local") == "dotenv"
    assert detect_config_language("config/settings.yaml") == "yaml"


def test_default_include_patterns_cover_docs_code_and_config():
    required = {
        "**/*.md",
        "**/*.rst",
        "**/*.adoc",
        "**/*.ipynb",
        "**/*.java",
        "**/*.py",
        "**/*.pyi",
        "**/*.lua",
        "**/*.ps1",
        "**/*.yaml",
        "**/*.json",
        "**/*.gradle.kts",
        "**/*.org",
    }
    assert required.issubset(set(DEFAULT_INCLUDE_PATTERNS))
