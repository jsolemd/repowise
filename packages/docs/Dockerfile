# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /usr/local/bin/uv
WORKDIR /app
ENV XDG_CACHE_HOME=/app/.cache
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked uv sync --frozen --no-dev --no-install-project
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked uv sync --frozen --no-dev --no-editable
# Grammar-pack releases fetch native parsers on demand. Bake exactly the docs
# extractor's supported grammars into the image so indexing works offline.
RUN .venv/bin/python -c "from tree_sitter_language_pack import prefetch; from repowise.docs.chunking.extractors.ast_symbols import _TS_LANGUAGE_MAP; prefetch(sorted(set(_TS_LANGUAGE_MAP.values())))"

FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl git && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache /app/.cache
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/app/.cache
RUN --network=none python -c "from tree_sitter_language_pack import get_parser; from repowise.docs.chunking.extractors.ast_symbols import _TS_LANGUAGE_MAP; [get_parser(language) for language in sorted(set(_TS_LANGUAGE_MAP.values()))]"
EXPOSE 8200
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -sf http://localhost:8200/livez || exit 1
CMD ["repowise-docs-worker"]
