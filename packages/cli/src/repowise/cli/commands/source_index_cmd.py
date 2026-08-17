"""``repowise source-index`` — build the source-chunk retrieval corpus.

Reads the wiki index's symbols and the repository's own worktree, embeds the
result, and writes a LanceDB table, an FTS5 sidecar and a manifest. No LLM
calls: embeddings only, which is why it is cheap enough to rerun.

Gated on ``REPOWISE_SOURCE_SEARCH``. Off is the default and the command
refuses rather than doing nothing quietly, so a scripted run that forgot the
flag fails where the mistake was made.
"""

from __future__ import annotations

import click

from repowise.cli.helpers import ensure_repowise_dir, resolve_repo_path, run_async
from repowise.cli.output import emit_json
from repowise.core.source_search import SOURCE_SEARCH_ENV, source_search_enabled


@click.command("source-index")
@click.option(
    "--repo",
    "repo",
    default=None,
    metavar="PATH",
    help="Repository to index. Defaults to the current directory.",
)
@click.option(
    "--embedder",
    "embedder_name",
    type=click.Choice(["gemini", "openai", "openrouter", "ollama", "mock", "auto"]),
    default="auto",
    help="Embedder to use. 'auto' detects from config / env vars.",
)
@click.option(
    "--batch-size",
    type=click.IntRange(1, 256),
    default=None,
    help="Chunks per embedding request. Defaults to the recipe's 16.",
)
def source_index_command(repo: str | None, embedder_name: str, batch_size: int | None) -> None:
    """Build the source-code retrieval corpus for a repository.

    One chunk per indexed symbol, plus overlapping line windows over the
    tracked files no grammar covers (shell, compose, systemd, SQL). Rebuilds
    the whole corpus; vectors whose chunk text is unchanged are carried over
    from the previous build instead of being recomputed.
    """
    if not source_search_enabled():
        raise click.ClickException(
            f"Source search is disabled. Set {SOURCE_SEARCH_ENV}=1 to enable it."
        )

    _log_progress_to_stderr()
    repo_path = resolve_repo_path(repo)
    ensure_repowise_dir(repo_path)

    from repowise.cli.ui import load_dotenv

    load_dotenv(repo_path)

    run_async(_run(repo_path, embedder_name, batch_size))


def _log_progress_to_stderr() -> None:
    """Move structlog output to stderr, leaving stdout for the JSON summary.

    structlog's default ``PrintLoggerFactory`` writes to stdout, so the
    progress lines of an indexing run land inside the document this command
    exists to print and ``repowise source-index | jq`` fails on the first one.

    The house helper for a machine-readable command,
    ``silence_logs_for_machine_output``, solves that by pinning the level to
    ERROR — right for a command that finishes in a second, wrong for one that
    spends minutes embedding and whose progress is the only sign it is alive.
    Moving the stream keeps both, and is what ``mcp_server/_stdio_logging``
    does where stdout is likewise a channel with a format.
    """
    import sys

    import structlog

    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


async def _run(repo_path, embedder_name: str, batch_size: int | None) -> None:
    from pathlib import Path

    from repowise.cli.providers.embedders import build_embedder, resolve_embedder_for_repo
    from repowise.core.providers.embedding.base import KeylessEmbedder
    from repowise.core.source_search.indexer import EMBED_BATCH_SIZE, build_source_index
    from repowise.core.source_search.manifest import EmbedderIdentity

    requested = embedder_name
    if embedder_name == "auto":
        # The pin in ``config.yaml`` is what wrote the repo's other vectors, so
        # it is what this index should be built with too.
        embedder_name = resolve_embedder_for_repo(repo_path)

    embedder = build_embedder(embedder_name, repo_path)
    if isinstance(embedder, KeylessEmbedder) and requested != "mock":
        # An 8-wide keyless vector is distinct per text and not discriminative
        # (see ``providers.embedding.base``), so an index built on it would
        # rank by noise while looking like it worked.
        raise click.ClickException(
            "No real embedder is configured. Set an embedder key, configure Ollama, "
            "or pass --embedder mock to build a deterministic test index."
        )

    identity = EmbedderIdentity(
        provider=embedder_name,
        model=_model_name(embedder),
        dims=int(embedder.dimensions),
    )

    result = await build_source_index(
        Path(repo_path),
        embedder=embedder,
        embedder_identity=identity,
        batch_size=batch_size or EMBED_BATCH_SIZE,
    )

    emit_json(
        {
            "repo": str(repo_path),
            "chunks": {
                "symbol": result.symbol_chunks,
                "file_window": result.file_window_chunks,
                "total": result.symbol_chunks + result.file_window_chunks,
                "files_covered": result.files_covered,
            },
            "embedding": {
                "provider": identity.provider,
                "model": identity.model,
                "dims": identity.dims,
                "embedded": result.embedded,
                "reused": result.reused,
            },
            "timings_seconds": {
                "load_and_build": result.load_seconds,
                "embed": result.embed_seconds,
                "write_fts": result.write_seconds,
                "total": result.total_seconds,
            },
            "indexed_commit": result.indexed_commit,
            "corpus_hash": result.corpus_hash,
            "recipe_fingerprint": result.recipe_fingerprint,
            "paths": {
                "manifest": result.manifest_path,
                "lancedb": result.lancedb_path,
                "fts": result.fts_path,
            },
        }
    )


def _model_name(embedder: object) -> str:
    """The embedder's model identifier, or its class name when it has none.

    ``Embedder`` is a structural protocol with no model attribute, and the
    concrete classes spell it privately. The name goes into the recipe
    fingerprint, so falling back to the class is what keeps a mock index from
    claiming the same fingerprint as a real one.
    """
    for attribute in ("_model", "model", "_model_name"):
        value = getattr(embedder, attribute, None)
        if isinstance(value, str) and value:
            return value
    return type(embedder).__name__
