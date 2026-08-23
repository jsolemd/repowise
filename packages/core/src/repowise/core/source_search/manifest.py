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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .chunks import SourceChunk, recipe_parameters
from .fts import tokenizer_parameters

__all__ = [
    "MANIFEST_FILENAME",
    "EmbedderIdentity",
    "ManifestReadResult",
    "SourceIndexManifest",
    "corpus_hash",
    "corpus_hash_entries",
    "default_manifest_path",
    "identify_embedder",
    "inspect_manifest",
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
    """Everything about the embedding adapter that changes vector meaning."""

    provider: str
    model: str
    dims: int
    document_prefix: str = ""
    query_prefix: str = ""


def identify_embedder(
    embedder: Any,
    *,
    provider: str | None = None,
    document_prefix: str = "",
    query_prefix: str = "",
) -> EmbedderIdentity:
    """Describe the concrete adapter that will produce source vectors.

    Provider/model are runtime facts, not configuration aspirations.  The
    fallback inference is used by hosts that already own an embedder object
    (server jobs and resumed CLI runs) and therefore should not construct a
    second adapter merely to fingerprint the first one.
    """

    resolved_provider = (provider or "").strip().lower()
    if not resolved_provider:
        module_leaf = type(embedder).__module__.rsplit(".", 1)[-1].lower()
        resolved_provider = (
            "mock" if type(embedder).__name__ in {"MockEmbedder", "KeylessEmbedder"} else module_leaf
        )
    model = ""
    for attribute in ("_model", "model", "_model_name"):
        value = getattr(embedder, attribute, None)
        if isinstance(value, str) and value:
            model = value
            break
    return EmbedderIdentity(
        provider=resolved_provider,
        model=model or type(embedder).__name__,
        dims=int(embedder.dimensions),
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )


def recipe_fingerprint(embedder: EmbedderIdentity) -> str:
    """Hash of everything that decides a vector except the repository's text.

    A change to any of it means the stored vectors were computed by a
    different recipe and cannot be reused, however unchanged the source is.
    Built from ``(name, value)`` pairs rather than a positional tuple so that
    adding a knob cannot silently reorder the ones already there and
    invalidate every index in the field for no reason.
    """
    # Symbol text is sliced at parser-produced bounds.  A query/extractor
    # change can therefore change the corpus even when every source byte and
    # chunking constant stayed fixed; the parser build belongs in the recipe.
    from repowise.core.ingestion.parse_cache import parser_fingerprint

    parts = [
        *recipe_parameters(),
        *tokenizer_parameters(),
        ("parser_fingerprint", parser_fingerprint()),
        ("embedder_provider", embedder.provider),
        ("embedder_model", embedder.model),
        ("embedder_dims", str(embedder.dims)),
        ("embedder_document_prefix", embedder.document_prefix),
        ("embedder_query_prefix", embedder.query_prefix),
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
    return corpus_hash_entries((chunk.chunk_id, chunk.content_hash) for chunk in chunks)


def corpus_hash_entries(entries: Iterable[tuple[str, str]]) -> str:
    """Hash ``(chunk_id, content_hash)`` pairs from an already-persisted corpus."""

    lines = sorted(f"{chunk_id}\x00{content_hash}" for chunk_id, content_hash in entries)
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
    generation_id: str = "legacy"
    generation_sequence: int = 0
    lance_table: str = "source_chunks"
    fts_path: str = ".repowise/source_search/source_fts.db"
    stale_files: dict[str, str] = field(default_factory=dict)
    #: Path -> content hash of the bytes this generation ingested for it, for
    #: every path that was dirty against HEAD at build time. The live
    #: freshness check compares disk bytes against THIS, not against git:
    #: a reconciled edit is fresh while it still matches, and a post-build
    #: revert is stale even though ``git diff HEAD`` is clean.
    working_tree_ingest: dict[str, str] = field(default_factory=dict)

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
                document_prefix=str(embedder.get("document_prefix") or ""),
                query_prefix=str(embedder.get("query_prefix") or ""),
            ),
            generation_id=str(raw.get("generation_id") or "legacy"),
            generation_sequence=int(raw.get("generation_sequence") or 0),
            lance_table=str(raw.get("lance_table") or "source_chunks"),
            fts_path=str(raw.get("fts_path") or ".repowise/source_search/source_fts.db"),
            stale_files={
                str(path): str(reason) for path, reason in (raw.get("stale_files") or {}).items()
            }
            if isinstance(raw.get("stale_files") or {}, dict)
            else {},
            working_tree_ingest={
                str(path): str(digest)
                for path, digest in (raw.get("working_tree_ingest") or {}).items()
            }
            if isinstance(raw.get("working_tree_ingest") or {}, dict)
            else {},
        )


@dataclass(frozen=True, slots=True)
class ManifestReadResult:
    """Discriminated manifest read for status surfaces.

    Rebuild callers deliberately use :func:`read_manifest`, where both an
    absent and an unreadable manifest mean "do not reuse vectors". Observers
    need the stronger distinction: a broken publication pointer must never be
    reported as an empty index.
    """

    state: Literal["missing", "unreadable", "ok"]
    manifest: SourceIndexManifest | None = None
    error: str | None = None


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
        # ``fsync`` above makes the bytes durable; syncing the containing
        # directory makes the rename itself durable.  Without both, a power
        # loss may resurrect the prior publication pointer even though every
        # derived-store transaction for the new generation committed.
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def inspect_manifest(path: Path | str) -> ManifestReadResult:
    """Read a manifest while preserving missing vs unreadable state.

    This is the observer-facing sibling of :func:`read_manifest`. It never
    raises, but unlike that rebuild-oriented API it carries the parse or I/O
    failure so health checks cannot mistake corruption for absence.
    """

    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ManifestReadResult(state="missing")
    except (OSError, ValueError) as exc:
        return ManifestReadResult(
            state="unreadable",
            error=f"{type(exc).__name__}: {exc}",
        )
    if not isinstance(raw, dict):
        return ManifestReadResult(
            state="unreadable",
            error=f"manifest root must be an object, found {type(raw).__name__}",
        )
    try:
        manifest = SourceIndexManifest.from_dict(raw)
    except (TypeError, ValueError) as exc:
        return ManifestReadResult(
            state="unreadable",
            error=f"{type(exc).__name__}: {exc}",
        )
    return ManifestReadResult(state="ok", manifest=manifest)


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
