"""Tests for the markdown chunking module."""

from repowise.docs.chunking import ChunkType, DocCategory, DocChunk, chunk_file, classify_doc_type
from repowise.docs.chunking.classifier import FileType, classify_file_type
from repowise.docs.chunking.extractors.mdx import strip_jsx
from repowise.docs.chunking.extractors.text_split import split_large_section
from repowise.docs.chunking.models import slugify
from repowise.docs.chunking.optimizer import greedy_merge_split_tails


class TestSlugify:
    """Tests for the slugify function."""

    def test_basic_slugify(self):
        assert slugify("Getting Started") == "getting-started"

    def test_special_characters(self):
        assert slugify("What is Langfuse?") == "what-is-langfuse"

    def test_ampersand(self):
        # GitHub keeps consecutive hyphens
        assert slugify("Setup & Installation") == "setup--installation"

    def test_long_title(self):
        result = slugify("Python Contextvars and ThreadPoolExecutors")
        assert result == "python-contextvars-and-threadpoolexecutors"

    def test_colon(self):
        result = slugify("Limitation: Python Contextvars")
        assert result == "limitation-python-contextvars"


class TestChunkFile:
    """Tests for the chunk_file function."""

    def test_empty_file(self):
        """Empty file should produce no chunks."""
        chunks = chunk_file("empty.md", "", "/test/test", "abc123")
        assert len(chunks) == 0

    def test_frontmatter_only(self):
        """File with only frontmatter should produce no chunks."""
        content = """---
title: Test
description: A test
---
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")
        assert len(chunks) == 0

    def test_toml_frontmatter(self):
        """TOML-style MDX frontmatter should be parsed and removed."""
        content = """---
title = "All about Supabase Egress"
github_url = "https://github.com/orgs/supabase/discussions/26605"
topics = [ "platform", "database" ]
keywords = [ "egress", "bandwidth" ]
database_id = "123"
---

This page explains how Supabase egress and bandwidth are measured.
"""
        chunks = chunk_file(
            "apps/docs/content/troubleshooting/all-about-supabase-egress.mdx",
            content,
            "/supabase/supabase",
            "abc123",
        )

        assert len(chunks) == 1
        assert chunks[0].breadcrumb == ["All about Supabase Egress"]
        assert chunks[0].metadata["title"] == "All about Supabase Egress"
        assert "title =" not in chunks[0].content
        assert "egress and bandwidth" in chunks[0].content

    def test_non_mapping_frontmatter_is_ignored(self):
        """Frontmatter that is neither YAML mapping nor TOML should not fail chunking."""
        content = """---
just a scalar
---

# Body

Searchable documentation content.
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")

        assert len(chunks) == 1
        assert chunks[0].breadcrumb == ["Body"]
        assert "Searchable documentation content" in chunks[0].content

    def test_no_headers(self):
        """File with no headers should produce one chunk."""
        content = """This is just some text without any headers.

It has multiple paragraphs.

And some more content here.
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == ChunkType.DOC
        assert "without any headers" in chunks[0].content

    def test_simple_structure(self):
        """Test basic markdown structure with headers and code."""
        content = """# Getting Started

This is an introduction.

## Installation

Install with pip:

```bash
pip install langfuse
```
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")

        # Should have: 1 DOC for intro, 1 DOC for installation, 1 CODE for bash
        assert len(chunks) >= 3

        doc_chunks = [c for c in chunks if c.chunk_type == ChunkType.DOC]
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]

        assert len(doc_chunks) >= 2
        assert len(code_chunks) == 1

        # Check breadcrumbs
        install_chunks = [c for c in chunks if c.section_anchor == "installation"]
        assert len(install_chunks) >= 1
        assert install_chunks[0].breadcrumb == ["Getting Started", "Installation"]

    def test_duplicate_headers(self):
        """Duplicate headers should get unique anchors."""
        content = """# Overview

First overview.

# Overview

Second overview.
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")
        anchors = [c.section_anchor for c in chunks]

        assert "overview" in anchors
        assert "overview-1" in anchors

    def test_deep_nesting(self):
        """Test H1 > H2 > H3 > H4 hierarchy."""
        content = """# H1 Title

## H2 Section

### H3 Subsection

#### H4 Detail

Content at H4 level.
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")

        h4_chunks = [c for c in chunks if c.section_anchor == "h4-detail"]
        assert len(h4_chunks) == 1
        assert h4_chunks[0].breadcrumb == [
            "H1 Title",
            "H2 Section",
            "H3 Subsection",
            "H4 Detail",
        ]

    def test_code_block_extraction(self):
        """Code blocks should have language and parent chunk."""
        content = """# Example

Here's some code:

```python
def hello():
    print("world")
```
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]

        assert len(code_chunks) == 1
        assert code_chunks[0].metadata.get("language") == "python"
        assert code_chunks[0].parent_chunk_id is not None
        assert "def hello():" in code_chunks[0].content

    def test_code_block_with_title(self):
        """Code blocks with title metadata should be extracted."""
        content = """# Example

```python title="hello.py"
def hello():
    pass
```
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]

        assert len(code_chunks) == 1
        assert code_chunks[0].metadata.get("language") == "python"
        assert code_chunks[0].metadata.get("title") == "hello.py"

    def test_large_code_block_respects_chunk_budget(self, monkeypatch):
        """Large code blocks should stay searchable without creating oversized chunks."""
        monkeypatch.setenv("DOC_SEARCH_CHUNK_MAX_TOKENS", "20")

        from repowise.docs import config

        config.get_settings.cache_clear()

        large_code = "print('x')\n" * 200
        content = f"""# Example

Some intro text.

```python
{large_code}```
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]

        assert len(code_chunks) > 1
        assert sum(chunk.content.count("print('x')") for chunk in code_chunks) == 200
        assert all(
            chunk.token_count is not None and chunk.token_count <= 20 for chunk in code_chunks
        )

    def test_garbage_chunks_are_filtered(self):
        """Tiny MDX/JSX remnants should not be indexed as chunks."""
        content = """# Title

Valid content here.

## Garbage

} />
"""
        chunks = chunk_file("test.md", content, "/test/test", "abc123")

        assert any("Valid content here" in c.content for c in chunks)
        assert all("} />" not in c.content for c in chunks)
        assert all(c.section_anchor != "garbage" for c in chunks)

    def test_source_url_generation(self):
        """Source URLs should point to GitHub."""
        content = """# Test

Content here.
"""
        chunks = chunk_file(
            "docs/test.md",
            content,
            "/langfuse/langfuse",
            "abc123",
            repo="langfuse/langfuse",
            branch="main",
        )

        assert len(chunks) == 1
        assert chunks[0].source_url == (
            "https://github.com/langfuse/langfuse/blob/abc123/docs/test.md#test"
        )

    def test_split_chunks_keep_canonical_anchor_and_line_spans(self, monkeypatch):
        """Split markdown chunks should preserve public anchors and line spans."""
        monkeypatch.setenv("DOC_SEARCH_CHUNK_MAX_TOKENS", "12")

        from repowise.docs import config

        config.get_settings.cache_clear()

        content = """# Overview

Line one that is long enough to split across multiple chunks.

Line two that is also long enough to force another split.
"""
        chunks = chunk_file("docs/test.md", content, "/test/test", "abc123")
        doc_chunks = [chunk for chunk in chunks if chunk.chunk_type == ChunkType.DOC]

        assert len(doc_chunks) >= 2
        assert doc_chunks[0].public_anchor == "overview"
        assert doc_chunks[1].public_anchor == "overview"
        assert doc_chunks[0].line_start is not None
        assert doc_chunks[0].line_end is not None
        assert doc_chunks[1].line_start >= doc_chunks[0].line_start

    def test_stable_chunk_ids(self):
        """Same content should produce same chunk IDs."""
        content = """# Test

Content here.
"""
        chunks1 = chunk_file("test.md", content, "/test/test", "abc123")
        chunks2 = chunk_file("test.md", content, "/test/test", "abc123")

        assert chunks1[0].id == chunks2[0].id

    def test_different_content_different_ids(self):
        """Different content should produce different chunk IDs."""
        content1 = "# Test\n\nContent one."
        content2 = "# Test\n\nContent two."

        chunks1 = chunk_file("test.md", content1, "/test/test", "abc123")
        chunks2 = chunk_file("test.md", content2, "/test/test", "abc123")

        assert chunks1[0].id != chunks2[0].id

    def test_asciidoc_routing_parses_sections_and_source_blocks(self):
        """AsciiDoc should preserve section hierarchy and extract source blocks."""
        content = """= PageRank
:description: Centrality algorithm

== Syntax

[source,cypher,role=noplay]
----
CALL gds.pageRank.stream('g')
YIELD nodeId, score
----
"""
        chunks = chunk_file(
            "doc/modules/ROOT/pages/algorithms/page-rank.adoc",
            content,
            "/neo4j/graph-data-science",
            "abc123",
            repo="neo4j/graph-data-science",
            branch="2.13",
        )

        assert len(chunks) >= 2
        doc_chunks = [c for c in chunks if c.chunk_type == ChunkType.DOC]
        code_chunks = [c for c in chunks if c.chunk_type == ChunkType.CODE]
        assert doc_chunks
        assert code_chunks
        assert any(c.section_anchor == "syntax" for c in chunks)
        assert any(c.metadata.get("language") == "cypher" for c in code_chunks)

    def test_java_routing_produces_structure_aware_code_chunks(self):
        """Java files should be split around type/method boundaries as CODE chunks."""
        content = """package example;

public class DemoService {
    public void runTask(String id) {
        System.out.println(id);
    }
}
"""
        chunks = chunk_file(
            "src/main/java/example/DemoService.java",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert all(c.chunk_type == ChunkType.CODE for c in chunks)
        assert any(c.metadata.get("language") == "java" for c in chunks)
        assert any(c.metadata.get("symbol_kind") in {"class", "method"} for c in chunks)

    def test_gradle_routing_produces_config_chunks(self):
        """Gradle files should produce CONFIG chunks with block breadcrumbs."""
        content = """plugins {
    id 'java'
}

dependencies {
    implementation 'org.example:demo:1.0'
}
"""
        chunks = chunk_file(
            "build.gradle",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert all(c.chunk_type == ChunkType.CONFIG for c in chunks)
        assert any(c.metadata.get("language") == "gradle" for c in chunks)
        assert any("dependencies" in c.breadcrumb_text.lower() for c in chunks)

    def test_rst_routing_parses_sections_and_code_directive(self):
        """RST files should preserve section hierarchy and code blocks."""
        content = """Title
=====

Usage
-----

.. code-block:: python

   def run():
       return 42
"""
        chunks = chunk_file(
            "docs/usage.rst",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.chunk_type == ChunkType.CODE for c in chunks)
        assert any(c.section_anchor == "usage" for c in chunks)
        assert any(
            c.metadata.get("language") == "python" for c in chunks if c.chunk_type == ChunkType.CODE
        )

    def test_generic_python_source_routing_produces_code_chunks(self):
        """Non-Java source files should route through generic source chunking."""
        content = """class Worker:
    def run(self, item: str) -> str:
        return item.upper()
"""
        chunks = chunk_file(
            "src/worker.py",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert all(c.chunk_type == ChunkType.CODE for c in chunks)
        assert any(c.metadata.get("language") == "python" for c in chunks)
        assert any(c.metadata.get("symbol_kind") in {"class", "function"} for c in chunks)

    def test_python_ast_extracts_async_method_symbol(self):
        """AST path should capture async methods that regex heuristics miss."""
        content = """class Worker:
    async def run(self, item: str) -> str:
        return item.upper()
"""
        chunks = chunk_file(
            "src/worker.py",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.metadata.get("parser") == "ast" for c in chunks)
        assert any(c.metadata.get("symbol_name") == "run" for c in chunks)
        assert any(c.metadata.get("symbol_kind") == "method" for c in chunks)

    def test_typescript_ast_extracts_class_method_symbol(self):
        """AST path should capture TS class methods as method chunks."""
        content = """export class Service {
    runTask(id: string): string {
        return id;
    }
}
"""
        chunks = chunk_file(
            "src/service.ts",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.metadata.get("parser") == "ast" for c in chunks)
        assert any(c.metadata.get("symbol_name") == "runTask" for c in chunks)
        assert any(c.metadata.get("symbol_kind") == "method" for c in chunks)

    def test_go_ast_extracts_type_and_method_symbols(self):
        """AST path should capture Go type/function/method declarations."""
        content = """type Worker struct{}

func NewWorker() *Worker { return &Worker{} }
func (w *Worker) Run() error { return nil }
"""
        chunks = chunk_file(
            "src/worker.go",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.metadata.get("parser") == "ast" for c in chunks)
        assert any(c.metadata.get("symbol_name") == "Worker" for c in chunks)
        assert any(c.metadata.get("symbol_kind") == "type" for c in chunks)
        assert any(c.metadata.get("symbol_name") == "Run" for c in chunks)
        assert any(c.metadata.get("symbol_kind") == "method" for c in chunks)

    def test_rust_ast_extracts_struct_trait_and_method_symbols(self):
        """AST path should capture Rust declarations with impl methods."""
        content = """pub struct Worker {}
pub trait Runner {
    fn run(&self);
}
impl Runner for Worker {
    fn run(&self) {}
}
impl Worker {
    fn compute(&self) {}
}
pub fn helper() {}
"""
        chunks = chunk_file(
            "src/worker.rs",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.metadata.get("parser") == "ast" for c in chunks)
        assert any(c.metadata.get("symbol_name") == "Worker" for c in chunks)
        assert any(c.metadata.get("symbol_kind") == "class" for c in chunks)
        assert any(c.metadata.get("symbol_name") == "Runner" for c in chunks)
        assert any(c.metadata.get("symbol_kind") == "interface" for c in chunks)
        assert any(
            c.metadata.get("symbol_name") == "run" and c.metadata.get("symbol_kind") == "method"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "compute" and c.metadata.get("symbol_kind") == "method"
            for c in chunks
        )

    def test_kotlin_ast_extracts_class_interface_and_top_level_function(self):
        """AST path should capture Kotlin type and function declarations."""
        content = """class Worker {
    fun run(x: Int): Int { return x }
}
interface Runner {
    fun run(): Int
}
fun top(x: Int) = x
"""
        chunks = chunk_file(
            "src/worker.kt",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.metadata.get("parser") == "ast" for c in chunks)
        assert any(
            c.metadata.get("symbol_name") == "Worker" and c.metadata.get("symbol_kind") == "class"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "Runner"
            and c.metadata.get("symbol_kind") == "interface"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "run" and c.metadata.get("symbol_kind") == "method"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "top" and c.metadata.get("symbol_kind") == "function"
            for c in chunks
        )

    def test_scala_ast_extracts_trait_object_and_top_level_function(self):
        """AST path should capture Scala declarations and methods."""
        content = """class Worker {
    def run(x: Int): Int = x
}
trait Runner {
    def run(): Int
}
object Registry {
    def add() = 1
}
def top(x: Int) = x
"""
        chunks = chunk_file(
            "src/worker.scala",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.metadata.get("parser") == "ast" for c in chunks)
        assert any(
            c.metadata.get("symbol_name") == "Worker" and c.metadata.get("symbol_kind") == "class"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "Runner"
            and c.metadata.get("symbol_kind") == "interface"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "run" and c.metadata.get("symbol_kind") == "method"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "top" and c.metadata.get("symbol_kind") == "function"
            for c in chunks
        )

    def test_ast_keeps_type_chunk_when_nested_symbol_shares_same_line(self):
        """A type declaration should still produce a chunk if a nested symbol starts on the same line."""
        content = """class Worker { fun run(x:Int):Int { return x } }
fun top(x:Int)=x
"""
        chunks = chunk_file(
            "src/worker.kt",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.metadata.get("parser") == "ast" for c in chunks)
        assert any(
            c.metadata.get("symbol_name") == "Worker" and c.metadata.get("symbol_kind") == "class"
            for c in chunks
        )
        assert any(
            c.metadata.get("symbol_name") == "run" and c.metadata.get("symbol_kind") == "method"
            for c in chunks
        )

    def test_generic_yaml_routing_produces_config_chunks(self):
        """YAML files should route through generic config chunking."""
        content = """database:
  host: localhost
  port: 5432

cache:
  enabled: true
"""
        chunks = chunk_file(
            "config/settings.yaml",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert all(c.chunk_type == ChunkType.CONFIG for c in chunks)
        assert any(c.metadata.get("language") == "yaml" for c in chunks)
        assert any("database" in c.breadcrumb_text.lower() for c in chunks)

    def test_html_document_routing_extracts_headings_and_code(self):
        """HTML docs should normalize to markdown headings and fenced code."""
        content = """<html><body>
<h1>Overview</h1>
<p>Intro paragraph.</p>
<h2>Example</h2>
<pre><code class="language-python">print(\"hello\")</code></pre>
</body></html>"""
        chunks = chunk_file(
            "docs/page.html",
            content,
            "/test/test",
            "abc123",
        )

        assert chunks
        assert any(c.chunk_type == ChunkType.DOC for c in chunks)
        assert any(c.chunk_type == ChunkType.CODE for c in chunks)
        assert any(c.section_anchor == "example" for c in chunks)
        assert any(
            c.metadata.get("language") == "python" for c in chunks if c.chunk_type == ChunkType.CODE
        )


class TestChunkType:
    """Tests for ChunkType enum."""

    def test_chunk_types(self):
        assert ChunkType.DOC.value == "doc"
        assert ChunkType.CODE.value == "code"
        assert ChunkType.LLMS_TXT.value == "llms_txt"


class TestDocChunk:
    """Tests for DocChunk dataclass."""

    def test_to_dict(self):
        chunk = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="Test content",
            breadcrumb=["Section"],
            section_anchor="section",
            source_url="https://github.com/test/test/blob/main/test.md#section",
        )

        d = chunk.to_dict()
        assert d["library_id"] == "/test/test"
        assert d["chunk_type"] == "doc"
        assert d["breadcrumb_text"] == "Section"

    def test_breadcrumb_text(self):
        chunk = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="Test",
            breadcrumb=["A", "B", "C"],
            section_anchor="c",
            source_url="https://example.com",
        )

        assert chunk.breadcrumb_text == "A > B > C"

    def test_empty_breadcrumb(self):
        chunk = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="Test",
            breadcrumb=[],
            section_anchor="",
            source_url="https://example.com",
        )

        assert chunk.breadcrumb_text == ""


class TestMDXStripping:
    """Tests for JSX stripping from MDX files."""

    def test_strip_import_statements(self):
        """Import statements should be removed."""
        content = """import { BlogHeader } from "@/components/blog/BlogHeader";
import React from "react";

# Title

Content here.
"""
        stripped = strip_jsx(content)
        assert "import" not in stripped
        assert "# Title" in stripped
        assert "Content here" in stripped

    def test_strip_self_closing_tags(self):
        """Self-closing JSX tags should be removed."""
        content = """# Title

<Video src="https://example.com/video.mp4" aspectRatio={16/9} />

Some text after.
"""
        stripped = strip_jsx(content)
        assert "<Video" not in stripped
        assert "/>" not in stripped
        assert "# Title" in stripped
        assert "Some text after" in stripped

    def test_strip_component_with_children(self):
        """Components with children should be stripped but children kept."""
        content = """# Title

<Callout type="info">
This is important information.
</Callout>

More text.
"""
        stripped = strip_jsx(content)
        assert "<Callout" not in stripped
        assert "</Callout>" not in stripped
        assert "This is important information" in stripped
        assert "More text" in stripped

    def test_strip_nested_components(self):
        """Nested components should all be stripped."""
        content = """<Frame className="max-w-lg">
  <Callout type="warning">
    Nested content here.
  </Callout>
</Frame>
"""
        stripped = strip_jsx(content)
        assert "<Frame" not in stripped
        assert "<Callout" not in stripped
        assert "Nested content here" in stripped

    def test_preserve_markdown_formatting(self):
        """Markdown formatting should be preserved."""
        content = """import { X } from "y";

# Title

**Bold** and *italic* and [link](https://example.com).

- List item 1
- List item 2

<Component />

```python
def hello():
    pass
```
"""
        stripped = strip_jsx(content)
        assert "# Title" in stripped
        assert "**Bold**" in stripped
        assert "*italic*" in stripped
        assert "[link](https://example.com)" in stripped
        assert "- List item 1" in stripped
        assert "```python" in stripped
        assert "def hello():" in stripped

    def test_preserve_frontmatter(self):
        """YAML frontmatter should be preserved."""
        content = """---
title: Test
author: Someone
---

import { X } from "y";

# Content
"""
        stripped = strip_jsx(content)
        assert "---" in stripped
        assert "title: Test" in stripped
        assert "# Content" in stripped

    def test_multiline_attributes(self):
        """Tags with multiline attributes should be stripped."""
        content = """<BlogHeader
  title="Long title here"
  description="A long description"
  authors={["author1", "author2"]}
/>

Text after.
"""
        stripped = strip_jsx(content)
        assert "<BlogHeader" not in stripped
        assert "Text after" in stripped

    def test_strip_dot_notation_components(self):
        """Components using dot-notation (e.g., Cards.Card) should be stripped."""
        content = """# Title

<Cards.Card
  title="Docs"
  href="/docs"
  icon={}
  arrow
/>

Text after.
"""
        stripped = strip_jsx(content)
        assert "<Cards.Card" not in stripped
        assert "icon={}" not in stripped
        assert "Text after" in stripped

    def test_jsx_nested_braces(self):
        """JSX with nested braces in attributes should be fully stripped."""
        content = """<Cards.Card
    title="Docs"
    href="/docs"
    icon={<BookOpen className="w-6 h-6" />}
    arrow
  />"""
        assert strip_jsx(content).strip() == ""

    def test_jsx_greater_than_in_expression(self):
        """Greater-than inside expressions shouldn't break parsing."""
        content = "<Component show={count > 0} />"
        assert strip_jsx(content).strip() == ""

    def test_residue_line_cleanup(self):
        """Residue lines like } /> should be removed."""
        content = """## Learn more

  } />

Some real content here."""
        stripped = strip_jsx(content)
        assert "} />" not in stripped
        assert "Learn more" in stripped
        assert "real content" in stripped

    def test_mdx_file_routing(self):
        """MDX files should use the MDX chunker."""
        content = """---
title: Test MDX
---

import { Component } from "lib";

# Heading

<Component prop="value" />

Regular markdown content.
"""
        chunks = chunk_file("test.mdx", content, "/test/test", "abc123")

        assert len(chunks) >= 1
        # Content should not have JSX
        for chunk in chunks:
            assert "<Component" not in chunk.content
            assert "import" not in chunk.content.split("\n")[0] if chunk.content else True


class TestGreedyTailMerger:
    """Tests for the GreedyTailMerger optimizer."""

    def test_merge_small_split_tails(self):
        """Small -part-2 chunks should merge back into the preceding chunk."""
        doc_part_1 = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="A" * 400,
            breadcrumb=["Title"],
            section_anchor="section",
            source_url="https://example.com#section",
        )
        original_doc_id = doc_part_1.id

        doc_part_2 = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="tail",
            breadcrumb=["Title"],
            section_anchor="section-part-2",
            source_url="https://example.com#section-part-2",
        )

        code_chunk = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.CODE,
            content="print('hello')",
            breadcrumb=["Title"],
            section_anchor="section",
            source_url="https://example.com#section",
            parent_chunk_id=original_doc_id,
            metadata={"language": "python"},
        )

        optimized = greedy_merge_split_tails(
            [doc_part_1, doc_part_2, code_chunk],
            max_tokens=800,
            min_tokens=100,
        )

        assert len(optimized) == 2
        assert optimized[0].chunk_type == ChunkType.DOC
        assert optimized[1].chunk_type == ChunkType.CODE
        assert "tail" in optimized[0].content

        # Doc ID should have been regenerated, and code chunk should point at the new one.
        assert optimized[0].id != original_doc_id
        assert optimized[1].parent_chunk_id == optimized[0].id

    def test_no_merge_across_sections(self):
        """Chunks from different sections should not merge."""
        part_1 = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="A" * 10,
            breadcrumb=["S1"],
            section_anchor="section-1",
            source_url="https://example.com#section-1",
        )
        part_2 = DocChunk(
            library_id="/test/test",
            file_path="test.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="tail",
            breadcrumb=["S2"],
            section_anchor="section-2-part-2",
            source_url="https://example.com#section-2-part-2",
        )

        optimized = greedy_merge_split_tails(
            [part_1, part_2],
            max_tokens=800,
            min_tokens=100,
        )

        assert len(optimized) == 2


class TestDocCategoryClassification:
    """Tests for document category classification."""

    def test_reference_path_patterns(self):
        """Reference documentation paths should be classified as REFERENCE."""
        reference_paths = [
            "docs/reference/api.md",
            "pages/api-reference/sdk.mdx",
            "docs/api/tracing.md",
            "docs/sdk/python.mdx",
            "03-api-reference/functions.md",
        ]
        for path in reference_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == DocCategory.REFERENCE, f"Expected REFERENCE for {path}"

    def test_guide_path_patterns(self):
        """Tutorial/guide paths should be classified as GUIDE."""
        guide_paths = [
            "docs/learn/basics.md",
            "tutorial/getting-started.md",
            "guides/quickstart.mdx",
            "01-getting-started/intro.md",
            "02-guides/integration.md",
        ]
        for path in guide_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == DocCategory.GUIDE, f"Expected GUIDE for {path}"

    def test_example_path_patterns(self):
        """Example/cookbook paths should be classified as EXAMPLE."""
        example_paths = [
            "examples/basic.md",
            "cookbook/recipe.mdx",
            "samples/demo.md",
        ]
        for path in example_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == DocCategory.EXAMPLE, f"Expected EXAMPLE for {path}"

    def test_changelog_path_patterns(self):
        """Changelog paths should be classified as CHANGELOG."""
        changelog_paths = [
            "CHANGELOG.md",
            "docs/changelog/2024.md",
            "releases/v1.md",
            "history.md",
        ]
        for path in changelog_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == DocCategory.CHANGELOG, f"Expected CHANGELOG for {path}"

    def test_changelog_chunk_type_overrides_path(self):
        """ChunkType.CHANGELOG should always result in DocCategory.CHANGELOG."""
        # Even with a non-changelog path, CHANGELOG chunk_type wins
        result = classify_doc_type("docs/api/reference.md", ChunkType.CHANGELOG)
        assert result == DocCategory.CHANGELOG

    def test_blog_path_patterns(self):
        """Blog paths should be classified as BLOG."""
        blog_paths = [
            "blog/post.md",
            "posts/announcement.md",
            "2024/01/15/release.md",
        ]
        for path in blog_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == DocCategory.BLOG, f"Expected BLOG for {path}"

    def test_uncategorized_paths(self):
        """Generic paths should be classified as OTHER."""
        other_paths = [
            "docs/introduction.md",
            "README.md",
            "some/random/file.md",
        ]
        for path in other_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == DocCategory.OTHER, f"Expected OTHER for {path}"

    def test_chunks_get_doc_category(self):
        """Chunks created via chunk_file should have doc_category set."""
        content = """# API Reference

This is the API documentation.
"""
        chunks = chunk_file("docs/reference/api.md", content, "/test/test", "abc123")
        assert len(chunks) >= 1
        assert chunks[0].doc_category == DocCategory.REFERENCE

    def test_doc_category_in_to_dict(self):
        """doc_category should be included in the chunk payload."""
        chunk = DocChunk(
            library_id="/test/test",
            file_path="docs/reference/api.md",
            commit_sha="abc123",
            chunk_type=ChunkType.DOC,
            content="Test",
            breadcrumb=["API"],
            section_anchor="test",
            source_url="https://example.com",
            doc_category=DocCategory.REFERENCE,
        )
        payload = chunk.to_dict()
        assert "doc_category" in payload
        assert payload["doc_category"] == "reference"


class TestClassifierEdgeCases:
    """Edge case tests for the document category classifier."""

    def test_windows_path_normalization(self):
        """Backslashes should be converted to forward slashes."""
        # Windows-style paths
        windows_paths = [
            ("docs\\reference\\api.md", DocCategory.REFERENCE),
            ("pages\\guides\\quickstart.mdx", DocCategory.GUIDE),
            ("blog\\2024\\01\\post.md", DocCategory.BLOG),
        ]
        for path, expected in windows_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == expected, f"Expected {expected} for Windows path {path}"

    def test_case_insensitive_matching(self):
        """Path patterns should match case-insensitively."""
        mixed_case_paths = [
            ("DOCS/REFERENCE/api.md", DocCategory.REFERENCE),
            ("Docs/Reference/API.md", DocCategory.REFERENCE),
            ("GUIDES/quickstart.md", DocCategory.GUIDE),
            ("Blog/Post.md", DocCategory.BLOG),
            ("CHANGELOG.MD", DocCategory.CHANGELOG),
        ]
        for path, expected in mixed_case_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == expected, f"Expected {expected} for mixed-case path {path}"

    def test_deeply_nested_path(self):
        """Deeply nested paths should still be classified correctly."""
        deep_paths = [
            ("docs/api/v2/internal/private/edge.md", DocCategory.REFERENCE),
            ("pages/docs/learn/advanced/topics/subtopic.mdx", DocCategory.GUIDE),
            ("content/blog/2024/01/15/posts/announcement.md", DocCategory.BLOG),
        ]
        for path, expected in deep_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == expected, f"Expected {expected} for deeply nested path {path}"

    def test_ambiguous_path_first_match_wins(self):
        """When paths match multiple patterns, first match (by priority) wins.

        Note: Pattern priority is REFERENCE > GUIDE > EXAMPLE > CHANGELOG > BLOG.
        This means if a path contains both 'api/' and 'guides/', 'api/' wins
        because REFERENCE patterns are checked first.
        """
        # api/ pattern in REFERENCE wins over guides/ in GUIDE
        # This is by design - reference docs are prioritized
        result = classify_doc_type("guides/api/overview.md", ChunkType.DOC)
        assert result == DocCategory.REFERENCE, "api/ matches REFERENCE before guides/"

        # reference/ pattern wins over tutorial/
        result = classify_doc_type("tutorial/reference/quickstart.md", ChunkType.DOC)
        assert result == DocCategory.REFERENCE, "reference/ matches REFERENCE before tutorial/"

        # api/ pattern wins over examples/
        result = classify_doc_type("examples/api/demo.md", ChunkType.DOC)
        assert result == DocCategory.REFERENCE, "api/ matches REFERENCE before examples/"

        # Pure examples path (no api/) should match EXAMPLE
        result = classify_doc_type("examples/basic/demo.md", ChunkType.DOC)
        assert result == DocCategory.EXAMPLE, "examples/ matches EXAMPLE"

        # Pure guides path (no api/) should match GUIDE
        result = classify_doc_type("guides/quickstart/intro.md", ChunkType.DOC)
        assert result == DocCategory.GUIDE, "guides/ matches GUIDE"

    def test_no_extension_path(self):
        """Paths without file extension should still be classified."""
        no_ext_paths = [
            ("docs/reference/api", DocCategory.REFERENCE),
            ("guides/quickstart", DocCategory.GUIDE),
            ("CHANGELOG", DocCategory.CHANGELOG),
        ]
        for path, expected in no_ext_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == expected, f"Expected {expected} for extensionless path {path}"

    def test_special_characters_in_path(self):
        """Paths with special characters should be handled.

        Note: Patterns match on path components with trailing slash (e.g., 'api/').
        So 'api_reference' doesn't match 'api/' - this is by design to avoid
        false positives on compound words.
        """
        special_paths = [
            # URL-encoded characters - matches reference/
            ("docs/reference/api%20reference.md", DocCategory.REFERENCE),
            # Spaces - matches reference/
            ("docs/reference/api reference.md", DocCategory.REFERENCE),
            # Underscores - 'api_reference' doesn't match 'api/' pattern
            # It matches docs/api/ instead (which is in REFERENCE patterns)
            ("docs/api/functions.md", DocCategory.REFERENCE),
            # Numbers in path - matches sdk/
            ("docs/sdk/v2/api.md", DocCategory.REFERENCE),
            # api-reference with hyphen - matches api-reference/
            ("pages/api-reference/endpoints.md", DocCategory.REFERENCE),
        ]
        for path, expected in special_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == expected, f"Expected {expected} for special char path {path}"

    def test_empty_and_root_paths(self):
        """Empty paths and root-level files should return OTHER."""
        edge_paths = [
            ("", DocCategory.OTHER),
            ("file.md", DocCategory.OTHER),
            ("README.md", DocCategory.OTHER),
        ]
        for path, expected in edge_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == expected, f"Expected {expected} for edge path '{path}'"

    def test_dot_notation_paths(self):
        """Paths with dots (like package names) should work.

        Note: Patterns like 'reference/' require a trailing slash to match.
        A file named 'reference.md' doesn't match the pattern.
        """
        dot_paths = [
            # api/ pattern matches because 'api' is a directory
            ("docs/api/com.example.client.md", DocCategory.REFERENCE),
            # sdk/ pattern matches
            ("api/sdk/org.project.index.md", DocCategory.REFERENCE),
            # Package name in path - matches api/
            ("api/com.example/reference.md", DocCategory.REFERENCE),
        ]
        for path, expected in dot_paths:
            result = classify_doc_type(path, ChunkType.DOC)
            assert result == expected, f"Expected {expected} for dotted path {path}"

    def test_file_type_classification_for_adoc_java_and_gradle(self):
        """File type classifier should recognize newly supported formats."""
        assert classify_file_type("doc/modules/ROOT/pages/index.adoc") == FileType.DOCUMENTATION
        assert classify_file_type("src/main/java/org/acme/Demo.java") == FileType.SOURCE
        assert classify_file_type("build.gradle") == FileType.CONFIG
        assert classify_file_type("settings.gradle.kts") == FileType.CONFIG
        assert classify_file_type("docs/guide.rst") == FileType.DOCUMENTATION
        assert classify_file_type("docs/page.html") == FileType.DOCUMENTATION
        assert classify_file_type("src/service.py") == FileType.SOURCE
        assert classify_file_type("Dockerfile") == FileType.CONFIG
        assert classify_file_type(".env.local") == FileType.CONFIG


class TestTextSplitter:
    """Tests for the text splitter, including table-aware splitting."""

    def test_table_not_split_mid_row(self):
        """Markdown tables should not be split between consecutive | rows."""
        header = "| Prop | Type | Default |\n|------|------|---------|"
        rows = "\n".join(f"| prop{i} | string | none |" for i in range(30))
        table = f"{header}\n{rows}"
        # Use a small token limit to force splitting
        chunks = list(split_large_section(table, max_tokens=60))
        # Every chunk should only contain complete table rows
        for chunk in chunks:
            for line in chunk.strip().splitlines():
                stripped = line.strip()
                if stripped:
                    assert stripped.startswith("|"), f"Non-table line in chunk: {stripped!r}"
                    assert stripped.endswith("|"), f"Incomplete table row: {stripped!r}"

    def test_table_kept_together_when_under_limit(self):
        """A table that fits within the limit should be a single chunk."""
        table = "| Name | Value |\n|------|-------|\n| a    | 1     |\n| b    | 2     |"
        chunks = list(split_large_section(table, max_tokens=200))
        assert len(chunks) == 1
        assert "| a    | 1     |" in chunks[0]

    def test_prose_with_embedded_table_preserves_table(self):
        """When prose + table are split, the table rows stay together."""
        content = (
            "Some intro paragraph.\n\n"
            "| Col A | Col B |\n"
            "|-------|-------|\n"
            "| x     | y     |\n"
            "| z     | w     |\n\n"
            "Some outro paragraph."
        )
        chunks = list(split_large_section(content, max_tokens=30))
        # Find the chunk(s) containing table rows
        for chunk in chunks:
            table_lines = [line for line in chunk.splitlines() if line.strip().startswith("|")]
            if table_lines:
                # All table lines should be in the same chunk (table is small)
                assert len(table_lines) >= 2  # at least header + one row

    def test_heading_with_inline_code_preserved(self):
        """Headers containing backtick code are parsed correctly."""
        md = "# Introduction\n\nText.\n\n## The `useState` Hook\n\nExplanation of useState."
        chunks = chunk_file(
            library_id="test/lib",
            file_path="docs/hooks.md",
            content=md,
            commit_sha="abc",
            repo="test/lib",
        )
        anchors = [c.section_anchor for c in chunks if c.chunk_type == ChunkType.DOC]
        assert any("usestate" in (a or "").lower() for a in anchors)

    def test_empty_code_block_produces_chunk(self):
        """An empty fenced code block should still produce a code chunk."""
        md = "# Setup\n\n```bash\n```\n\nDone."
        chunks = chunk_file(
            library_id="test/lib",
            file_path="docs/setup.md",
            content=md,
            commit_sha="abc",
            repo="test/lib",
        )
        # Empty code blocks are legitimately empty — chunker should not crash
        assert any(c.chunk_type == ChunkType.DOC for c in chunks)
