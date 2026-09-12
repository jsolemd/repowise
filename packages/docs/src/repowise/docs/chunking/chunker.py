"""Main chunking router for documentation files."""

import logging
from pathlib import Path

from repowise.docs.chunking.classifier import FileType, classify_doc_type, classify_file_type
from repowise.docs.chunking.extractors.asciidoc import chunk_asciidoc
from repowise.docs.chunking.extractors.changelog import chunk_changelog
from repowise.docs.chunking.extractors.markdown import chunk_markdown
from repowise.docs.chunking.extractors.mdx import chunk_mdx
from repowise.docs.chunking.extractors.rst import chunk_rst
from repowise.docs.chunking.extractors.source import chunk_source_file
from repowise.docs.chunking.extractors.text import chunk_text_document
from repowise.docs.chunking.models import ChunkType, DocChunk
from repowise.docs.chunking.optimizer import greedy_merge_split_tails
from repowise.docs.config import get_settings
from repowise.docs.formats import (
    ASCIIDOC_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    MDX_EXTENSIONS,
    RST_EXTENSIONS,
    detect_config_language,
    detect_source_language,
)

logger = logging.getLogger(__name__)

# File names that indicate changelog content
CHANGELOG_FILENAMES = {
    "changelog.md",
    "changelog.mdx",
    "changelog",
    "changes.md",
    "history.md",
    "releases.md",
    "release-notes.md",
    "releasenotes.md",
}


def chunk_file(
    file_path: str | Path,
    content: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """
    Chunk a documentation file based on its extension.

    Routes to the appropriate chunker based on file type:
    - markdown family -> Markdown chunker
    - .mdx -> MDX chunker (JSX stripped)
    - .adoc/.asciidoc -> AsciiDoc chunker
    - .rst -> RST chunker
    - .java -> source-aware Java chunker
    - .gradle/.gradle.kts -> structure-aware Gradle config chunker
    - other source/config extensions -> generic structure-aware chunkers
    - llms.txt -> LLMs.txt chunker (future)
    - CHANGELOG.md -> Changelog chunker (future)

    Args:
        file_path: Relative file path within repo
        content: Raw file content
        library_id: Library identifier (e.g., "/langfuse/langfuse")
        commit_sha: Git commit SHA
        repo: Optional repo path for URL generation
        branch: Git branch name
        source_path_prefix: Optional repo-relative prefix prepended before file_path for source URLs

    Returns:
        List of DocChunk objects
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    extension = file_path.suffix.lower()
    filename = file_path.name.lower()

    # Convert path to string for chunker
    file_path_str = str(file_path)
    file_type = classify_file_type(file_path_str)

    settings = get_settings()

    # Route based on file type
    if filename == "llms.txt":
        # LLMs.txt files - special handling (future)
        # For now, treat as plain text
        chunks = _chunk_llms_txt(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    elif filename in CHANGELOG_FILENAMES or filename.startswith("changelog"):
        # Changelog files - use specialized changelog chunker
        logger.debug(f"Processing changelog file: {file_path}")
        chunks = chunk_changelog(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    elif extension in MDX_EXTENSIONS:
        # MDX files - strip JSX, then parse as markdown
        logger.debug(f"Processing MDX file: {file_path}")
        chunks = chunk_mdx(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    elif extension in MARKDOWN_EXTENSIONS:
        # Standard markdown
        chunks = chunk_markdown(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    elif extension in ASCIIDOC_EXTENSIONS:
        # AsciiDoc section/listing block aware chunking
        chunks = chunk_asciidoc(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    elif extension in RST_EXTENSIONS:
        # RST directive + section normalization
        chunks = chunk_rst(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    elif extension == ".java":
        # Java structure-aware chunking (types + methods)
        chunks = chunk_source_file(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
            source_kind="java",
        )

    elif extension == ".gradle" or file_path_str.endswith(".gradle.kts"):
        # Gradle script block-aware chunking (plugins/dependencies/tasks/etc.)
        chunks = chunk_source_file(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
            source_kind="gradle",
        )

    elif file_type == FileType.SOURCE:
        # Generic source chunking for additional languages.
        chunks = chunk_source_file(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
            source_kind="generic-code",
            language=detect_source_language(file_path_str),
        )

    elif file_type == FileType.CONFIG:
        # Generic config chunking for YAML/JSON/TOML/etc.
        chunks = chunk_source_file(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
            source_kind="generic-config",
            language=detect_config_language(file_path_str),
        )

    elif file_type == FileType.DOCUMENTATION:
        # Text-like docs (txt/html/xml/etc.) go through markdown splitter fallback.
        chunks = chunk_text_document(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    else:
        # Unknown extension - use markdown parser as final fallback
        logger.warning(
            f"Unknown file extension '{extension}' for {file_path}, using markdown parser"
        )
        chunks = chunk_markdown(
            content,
            file_path_str,
            library_id,
            commit_sha,
            repo,
            branch,
            source_path_prefix,
        )

    # Apply doc category and file type classification to all chunks
    for chunk in chunks:
        chunk.doc_category = classify_doc_type(file_path_str, chunk.chunk_type)
        chunk.file_type = file_type

    return greedy_merge_split_tails(
        chunks,
        max_tokens=settings.chunk_max_tokens,
        min_tokens=settings.merge_min_tokens,
    )


def _chunk_llms_txt(
    content: str,
    file_path: str,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """
    Chunk an llms.txt file.

    LLMs.txt is typically a single file meant to be consumed whole,
    so we create one chunk for the entire content.

    Args:
        content: Raw file content
        file_path: Relative file path
        library_id: Library identifier
        commit_sha: Git commit SHA
        repo: Optional repo path for URL generation
        branch: Git branch name

    Returns:
        List with single DocChunk
    """
    from repowise.docs.chunking.models import build_source_url

    if repo is None:
        repo = library_id.lstrip("/")

    return [
        DocChunk(
            library_id=library_id,
            file_path=file_path,
            commit_sha=commit_sha,
            chunk_type=ChunkType.LLMS_TXT,
            content=content,
            breadcrumb=[],
            section_anchor="",
            source_url=build_source_url(
                repo,
                commit_sha,
                file_path,
                path_prefix=source_path_prefix,
            ),
            metadata={},
            canonical_anchor="",
            line_start=1,
            line_end=max(1, content.count("\n") + 1),
        )
    ]


def chunk_file_from_path(
    repo_dir: Path,
    rel_path: Path,
    library_id: str,
    commit_sha: str,
    repo: str | None = None,
    branch: str = "main",
    source_path_prefix: str = "",
) -> list[DocChunk]:
    """
    Chunk a file by reading it from disk.

    Convenience wrapper that reads the file content first.

    Args:
        repo_dir: Root directory of the repository
        rel_path: Relative path within repo
        library_id: Library identifier
        commit_sha: Git commit SHA
        repo: Optional repo path for URL generation
        branch: Git branch name

    Returns:
        List of DocChunk objects
    """
    abs_path = repo_dir / rel_path

    try:
        content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with latin-1 fallback
        content = abs_path.read_text(encoding="latin-1")
    except Exception as e:
        logger.error(f"Failed to read {rel_path}: {e}")
        return []

    return chunk_file(
        rel_path,
        content,
        library_id,
        commit_sha,
        repo,
        branch,
        source_path_prefix,
    )
