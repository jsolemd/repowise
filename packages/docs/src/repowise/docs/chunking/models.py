"""Data models for document chunks."""

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum


class ChunkType(StrEnum):
    """Type of document chunk."""

    DOC = "doc"  # Prose/documentation text
    CODE = "code"  # Code block
    CONFIG = "config"  # Configuration/YAML/JSON
    CHANGELOG = "changelog"  # Changelog entries
    LLMS_TXT = "llms_txt"  # LLMs.txt file


class FileType(StrEnum):
    """File type classification for search result boosting.

    Used to boost documentation files over source code in typical usage queries,
    while still allowing source code to rank higher for implementation queries.
    """

    DOCUMENTATION = "documentation"  # .md, .mdx, .rst, .adoc, .txt, .html, .xml
    EXAMPLE = "example"  # Files in examples/, cookbook/, tutorial/ paths
    SOURCE = "source"  # .py, .ts, .js, .go, .rs (actual source code)
    CONFIG = "config"  # .yaml, .yml, .toml, pyproject.toml
    TEST = "test"  # Files in test/, tests/, *_test.*, *.spec.*
    OTHER = "other"  # Uncategorized


class DocCategory(StrEnum):
    """Document category for relevance boosting.

    Used to boost canonical reference documentation over tutorials,
    changelogs, and blog posts in search results.
    """

    REFERENCE = "reference"  # API docs, canonical reference
    GUIDE = "guide"  # Tutorials, getting started
    EXAMPLE = "example"  # Cookbooks, examples
    CHANGELOG = "changelog"  # Release notes, changelogs
    BLOG = "blog"  # Blog posts, announcements
    OTHER = "other"  # Uncategorized


# Namespace UUID for generating stable chunk IDs (UUID5)
CHUNK_NAMESPACE = uuid.UUID("b5c7e8d4-9a3f-4b2e-8c1d-6e5f4a3b2c1d")
ANCHOR_SPLIT_SUFFIX_PATTERN = re.compile(r"(-part-\d+|-code-part-\d+)$")


@dataclass
class DocChunk:
    """
    A chunk of documentation content for indexing.

    Attributes:
        id: Stable UUID for this chunk
        library_id: Library identifier (e.g., "/langfuse/langfuse")
        file_path: Relative path within the repo
        commit_sha: Git commit SHA when indexed
        chunk_type: Type of chunk (doc, code, config, etc.)
        doc_category: Document category for relevance boosting
        file_type: File type for source vs docs boosting
        content: The actual text content
        breadcrumb: Header hierarchy leading to this chunk
        section_anchor: GitHub-style slug for deep linking
        source_url: Full GitHub URL to this section
        parent_chunk_id: Optional parent chunk ID (for code blocks)
        metadata: Additional metadata (language, title, etc.)
        heading_level: Heading level (1-6) of the section this chunk belongs to
        token_count: Estimated token count of the content
        canonical_anchor: Public section anchor for navigation (without split suffix)
        line_start: 1-based starting line number in the source file
        line_end: 1-based ending line number in the source file
    """

    library_id: str
    file_path: str
    commit_sha: str
    chunk_type: ChunkType
    content: str
    breadcrumb: list[str]
    section_anchor: str
    source_url: str
    doc_category: DocCategory = DocCategory.OTHER
    file_type: FileType = FileType.OTHER
    parent_chunk_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    heading_level: int | None = None
    token_count: int | None = None
    canonical_anchor: str | None = None
    line_start: int | None = None
    line_end: int | None = None

    # Generated field - computed in __post_init__
    id: str = field(default="", init=False)

    def __post_init__(self) -> None:
        """Generate stable chunk ID after initialization."""
        self.id = self._generate_id()

    def _generate_id(self) -> str:
        """
        Generate a stable UUID5 for this chunk.

        The ID is based on: library_id + file_path + chunk_index + content_hash
        This ensures the same content always gets the same ID, enabling
        stable vector point IDs in Qdrant.
        """
        # Include content hash to distinguish chunks from same location
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]

        # Build unique string for this chunk
        unique_string = f"{self.library_id}:{self.file_path}:{self.section_anchor}:{content_hash}"

        return str(uuid.uuid5(CHUNK_NAMESPACE, unique_string))

    @property
    def breadcrumb_text(self) -> str:
        """Get breadcrumb as a readable string."""
        return " > ".join(self.breadcrumb) if self.breadcrumb else ""

    @property
    def public_anchor(self) -> str:
        """Return the canonical public anchor for navigation."""
        if self.canonical_anchor:
            return self.canonical_anchor
        if not self.section_anchor:
            return ""
        return canonicalize_section_anchor(self.section_anchor)

    def to_dict(self) -> dict:
        """Convert to dictionary for Qdrant payload."""
        result = {
            "id": self.id,
            "library_id": self.library_id,
            "file_path": self.file_path,
            "commit_sha": self.commit_sha,
            "chunk_type": self.chunk_type.value,
            "doc_category": self.doc_category.value,
            "file_type": self.file_type.value,
            "content": self.content,
            "breadcrumb": self.breadcrumb,
            "breadcrumb_text": self.breadcrumb_text,
            "section_anchor": self.section_anchor,
            "canonical_anchor": self.public_anchor,
            "source_url": self.source_url,
            "parent_chunk_id": self.parent_chunk_id,
            "metadata": self.metadata,
        }
        # Include quality signals if set
        if self.heading_level is not None:
            result["heading_level"] = self.heading_level
        if self.token_count is not None:
            result["token_count"] = self.token_count
        if self.line_start is not None:
            result["line_start"] = self.line_start
        if self.line_end is not None:
            result["line_end"] = self.line_end
        return result


def slugify(text: str) -> str:
    """
    Convert text to GitHub-style anchor slug.

    GitHub's algorithm:
    1. Strip leading/trailing whitespace
    2. Convert to lowercase
    3. Replace spaces with hyphens
    4. Remove everything except alphanumeric, hyphens, underscores
    5. Remove consecutive hyphens

    Examples:
        "Getting Started" -> "getting-started"
        "What is Langfuse?" -> "what-is-langfuse"
        "Setup & Installation" -> "setup--installation"
    """
    text = text.strip().lower()
    # Replace spaces with hyphens
    text = text.replace(" ", "-")
    # Keep only alphanumeric, hyphens, underscores
    text = re.sub(r"[^a-z0-9_-]", "", text)
    # Remove consecutive hyphens (but GitHub actually keeps them)
    # text = re.sub(r"-+", "-", text)
    return text


def build_source_url(
    repo: str,
    ref: str,
    file_path: str,
    section_anchor: str = "",
    path_prefix: str = "",
) -> str:
    """
    Build a GitHub source URL for a document section.

    Args:
        repo: Repository path (e.g., "langfuse/langfuse")
        ref: Git ref for the permalink target (commit SHA or branch)
        file_path: File path within repo
        section_anchor: Optional section anchor
        path_prefix: Optional repo-relative prefix prepended before file_path

    Returns:
        GitHub permalink URL
    """
    normalized_prefix = path_prefix.strip("/")
    normalized_path = file_path.lstrip("/")
    if normalized_prefix:
        normalized_path = f"{normalized_prefix}/{normalized_path}"
    base_url = f"https://github.com/{repo}/blob/{ref}/{normalized_path}"
    if section_anchor:
        return f"{base_url}#{section_anchor}"
    return base_url


def append_anchor(source_url: str, section_anchor: str = "") -> str:
    """Append a section anchor to an existing source URL when appropriate."""
    if not section_anchor or not source_url:
        return source_url
    if "#" in source_url:
        return source_url
    return f"{source_url}#{section_anchor}"


def canonicalize_section_anchor(section_anchor: str) -> str:
    """Strip internal split suffixes from a chunk anchor for public navigation."""
    if not section_anchor:
        return ""
    return ANCHOR_SPLIT_SUFFIX_PATTERN.sub("", section_anchor)
