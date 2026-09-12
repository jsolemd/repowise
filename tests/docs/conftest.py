"""Shared test fixtures for doc-search tests.

This module provides reusable fixtures for mocking Qdrant points,
library states, and other common test dependencies.
"""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from repowise.docs.library.models import LibraryState, LibraryStatus


@pytest.fixture
def mock_qdrant_point():
    """Factory for creating mock Qdrant scored points.

    Usage:
        point = mock_qdrant_point("chunk-1", score=0.9, doc_category="reference")
        point = mock_qdrant_point("chunk-2", content="Custom content", file_path="docs/api.md")
    """

    def _create(
        chunk_id: str,
        score: float = 0.85,
        doc_category: str = "other",
        **payload_overrides: Any,
    ) -> MagicMock:
        point = MagicMock()
        point.id = chunk_id
        point.score = score
        point.payload = {
            "content": "test content",
            "breadcrumb": ["Test"],
            "breadcrumb_text": "Test",
            "file_path": "test.md",
            "section_anchor": "test",
            "source_url": "https://github.com/test/test/blob/main/test.md#test",
            "chunk_type": "doc",
            "doc_category": doc_category,
            "library_id": "/test/test",
            **payload_overrides,
        }
        return point

    return _create


@pytest.fixture
def mock_library_ready() -> LibraryState:
    """A library with READY status."""
    return LibraryState(
        library_id="/test/test",
        repo="test/test",
        name="Test Library",
        description="A test library for unit tests",
        docs_path="docs/",
        branch="main",
        status=LibraryStatus.READY,
        chunk_count=100,
        file_count=10,
        current_sha="abc123def456",
        indexed_at=datetime(2024, 1, 15, 12, 0, 0),
    )


@pytest.fixture
def mock_library_pending() -> LibraryState:
    """A library with PENDING status."""
    return LibraryState(
        library_id="/pending/lib",
        repo="pending/lib",
        name="Pending Library",
        description="A library awaiting indexing",
        docs_path="docs/",
        branch="main",
        status=LibraryStatus.PENDING,
        chunk_count=0,
        file_count=0,
    )


@pytest.fixture
def mock_library_indexing() -> LibraryState:
    """A library with INDEXING status."""
    return LibraryState(
        library_id="/indexing/lib",
        repo="indexing/lib",
        name="Indexing Library",
        description="A library currently being indexed",
        docs_path="docs/",
        branch="main",
        status=LibraryStatus.INDEXING,
        chunk_count=50,
        file_count=5,
        current_sha="partial123",
    )


@pytest.fixture
def mock_library_error() -> LibraryState:
    """A library with ERROR status."""
    return LibraryState(
        library_id="/error/lib",
        repo="error/lib",
        name="Error Library",
        description="A library with indexing errors",
        docs_path="docs/",
        branch="main",
        status=LibraryStatus.ERROR,
        chunk_count=0,
        file_count=0,
        error_message="GitHub API rate limit exceeded",
    )


@pytest.fixture
def mock_qdrant_record():
    """Factory for creating mock Qdrant records (for get_chunk_by_id).

    Unlike mock_qdrant_point, this creates a Record without a score,
    used for direct chunk retrieval.
    """

    def _create(
        chunk_id: str,
        library_id: str = "/test/test",
        file_path: str = "test.md",
        content: str = "Test chunk content",
        **payload_overrides: Any,
    ) -> MagicMock:
        record = MagicMock()
        record.id = chunk_id
        record.payload = {
            "content": content,
            "breadcrumb": ["Test"],
            "breadcrumb_text": "Test",
            "file_path": file_path,
            "section_anchor": "test",
            "source_url": f"https://github.com/{library_id.lstrip('/')}/blob/main/{file_path}#test",
            "chunk_type": "doc",
            "doc_category": "other",
            "library_id": library_id,
            **payload_overrides,
        }
        return record

    return _create


@pytest.fixture
def sample_file_content() -> str:
    """Sample markdown file content for expand-chunk tests."""
    return """# Getting Started

Welcome to the documentation.

## Installation

Install the package using pip:

```bash
pip install example-package
```

## Configuration

Configure your settings in `config.yaml`:

```yaml
debug: true
port: 8080
```

## Usage

Here's how to use the package:

```python
from example import Client

client = Client()
client.connect()
```

### Advanced Usage

For advanced features, see the API reference.

## Troubleshooting

Common issues and solutions.
"""
