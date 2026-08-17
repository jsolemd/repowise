"""What a source index was made of, written next to it.

Two hashes carry the weight. The **recipe fingerprint** covers everything
except the repository: chunk geometry, tokenizer version, embedder identity and
width. The **corpus hash** covers the repository and nothing else: every
chunk id paired with the hash of its text. Together they answer the only two
questions a rebuild has to ask — "were these vectors computed the way I would
compute them now?" and "has anything actually changed?" — without reading a
single vector.

Written atomically, because a half-written manifest beside a complete index is
indistinguishable from a complete manifest beside a half-written one, and the
next run would have to assume the worse of the two.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunks import SourceChunk, recipe_parameters
from .fts import tokenizer_parameters

__all__ = [
    "MANIFEST_FILENAME",
    "EmbedderIdentity",
    "SourceIndexManifest",
    "corpus_hash",
    "default_manifest_path",
    "read_manifest",
    "recipe_fingerprint",
    "write_manifest",
]

MANIFEST_FILENAME = "source_index.json"


def default_manifest_path(repo_path: Path | str) -> Path:
    """Path to *repo_path*'s source-index manifest (not created here)."""
    return Path(repo_path) / ".repowise" / MANIFEST_FILENAME


@dataclass(frozen=True, slots=True)
class EmbedderIdentity:
    """Which embedder produced the vectors, and how wide they are."""

    provider: str
    model: str
    dims: int


def recipe_fingerprint(embedder: EmbedderIdentity) -> str:
    """Hash of everything that decides a vector except the repository's text.

    A change to any of it means the stored vectors were computed by a
    different recipe and cannot be reused, however unchanged the source is.
    Built from ``(name, value)`` pairs rather than a positional tuple so that
    adding a knob cannot silently reorder the ones already there and
    invalidate every index in the field for no reason.
    """
    parts = [
        *recipe_parameters(),
        *tokenizer_parameters(),
        ("embedder_provider", embedder.provider),
        ("embedder_model", embedder.model),
        ("embedder_dims", str(embedder.dims)),
    ]
    digest = hashlib.sha256()
    for name, value in parts:
        digest.update(f"{name}={value}\n".encode())
    return digest.hexdigest()


def corpus_hash(chunks: Iterable[SourceChunk]) -> str:
    """Hash of the corpus: every chunk id with the hash of its text.

    Sorted, so the same repository hashes the same however the chunks were
    enumerated — which is what makes "did anything change?" answerable without
    depending on filesystem or database ordering.
    """
    lines = sorted(f"{chunk.chunk_id}\x00{chunk.content_hash}" for chunk in chunks)
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceIndexManifest:
    """The record of one completed build."""

    recipe_fingerprint: str
    corpus_hash: str
    symbol_chunks: int
    file_window_chunks: int
    files_covered: int
    indexed_commit: str | None
    built_at: str
    embedder: EmbedderIdentity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SourceIndexManifest:
        embedder = raw.get("embedder") or {}
        if not isinstance(embedder, dict):
            embedder = {}
        return cls(
            recipe_fingerprint=str(raw.get("recipe_fingerprint") or ""),
            corpus_hash=str(raw.get("corpus_hash") or ""),
            symbol_chunks=int(raw.get("symbol_chunks") or 0),
            file_window_chunks=int(raw.get("file_window_chunks") or 0),
            files_covered=int(raw.get("files_covered") or 0),
            indexed_commit=(
                str(raw["indexed_commit"]) if raw.get("indexed_commit") is not None else None
            ),
            built_at=str(raw.get("built_at") or ""),
            embedder=EmbedderIdentity(
                provider=str(embedder.get("provider") or ""),
                model=str(embedder.get("model") or ""),
                dims=int(embedder.get("dims") or 0),
            ),
        )


def write_manifest(path: Path | str, manifest: SourceIndexManifest) -> None:
    """Write *manifest* to *path*, atomically.

    Same directory for the temporary file, because ``os.replace`` is only
    atomic within a filesystem and ``/tmp`` is routinely a different one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_manifest(path: Path | str) -> SourceIndexManifest | None:
    """Read a manifest, or None when there is none or it is unreadable.

    Never raises: the only caller is a rebuild deciding whether it may reuse
    vectors, and "cannot tell" has to mean "re-embed", not "crash".
    """
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return SourceIndexManifest.from_dict(raw)
    except (TypeError, ValueError):
        return None
