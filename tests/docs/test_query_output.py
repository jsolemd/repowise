"""Tests for LLM-facing doc-search output formatting."""

from repowise.docs.tools.handlers import _format_as_markdown


def test_markdown_output_omits_redundant_summary_prefix() -> None:
    output = _format_as_markdown(
        [
            {
                "title": "Tracing",
                "summary": "Trace LLM calls with the observe decorator.",
                "content": "Trace LLM calls with the observe decorator.\n\nThen inspect spans in the UI.",
                "file_path": "docs/tracing.md",
                "score": 0.93,
                "chunk_type": "doc",
                "doc_category": "reference",
            }
        ],
        "observe decorator",
        "/langfuse/langfuse-docs",
    )

    assert "*Trace LLM calls with the observe decorator.*" not in output
    assert "Then inspect spans in the UI." in output
    assert "---" not in output


def test_markdown_output_keeps_distinct_summary() -> None:
    output = _format_as_markdown(
        [
            {
                "title": "Tracing",
                "summary": "Start here for the shortest setup path.",
                "content": "Trace LLM calls with the observe decorator.",
                "file_path": "docs/tracing.md",
                "score": 0.93,
                "chunk_type": "doc",
                "doc_category": "reference",
            }
        ],
        "observe decorator",
        "/langfuse/langfuse-docs",
    )

    assert "*Start here for the shortest setup path.*" in output
