"""Configuration for the library documentation runtime."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DOC_SEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    port: Annotated[int, Field(description="HTTP server port")] = 8200
    host: Annotated[str, Field(description="HTTP server host")] = "0.0.0.0"
    log_level: Annotated[str, Field(description="Logging level")] = "INFO"
    max_concurrent_tool_calls: Annotated[
        int, Field(description="Maximum concurrent MCP tool calls")
    ] = 4
    tool_timeout_search_seconds: Annotated[
        float, Field(description="Timeout for interactive search/read calls")
    ] = 20.0
    tool_timeout_seconds: Annotated[
        float, Field(description="Timeout for other documentation tool calls")
    ] = 30.0

    # Mutation auth for the natively served docs admin endpoint. This host is
    # the single docs writer, so /docs/action/index changes state here and is
    # gated by repowise.docs.transport.auth. Fail-closed: with neither a token nor
    # the explicit opt-in, the endpoint refuses.
    webhook_token: Annotated[
        str, Field(description="Token required in the X-Doc-Search-Token header for mutations")
    ] = ""
    allow_unauthenticated: Annotated[
        bool, Field(description="Allow token-less docs mutations on a trusted network")
    ] = False

    # Qdrant configuration
    qdrant_host: Annotated[str, Field(description="Qdrant API host")] = "http://qdrant:6333"
    qdrant_collection: Annotated[str, Field(description="Qdrant collection name")] = "doc-search"

    # TEI (Text Embeddings Inference) configuration
    tei_host: Annotated[str, Field(description="TEI API host")] = "http://tei:80"
    embedding_dimensions: Annotated[int, Field(description="Embedding vector dimensions")] = 768

    # Supabase configuration
    supabase_url: Annotated[str, Field(description="Supabase API URL")] = (
        "http://supabase-kong:8000"
    )
    supabase_service_key: Annotated[str, Field(description="Supabase service role key")] = ""

    # PostgreSQL direct connection (for asyncpg)
    postgres_dsn: Annotated[str, Field(description="PostgreSQL connection string")] = ""

    # Cache configuration
    cache_root: Annotated[Path, Field(description="Root directory for cached repos")] = Path(
        "/var/cache/code-search/doc-search"
    )

    # GitHub configuration - uses GITHUB_TOKEN (no prefix) as fallback
    github_token: Annotated[str, Field(description="GitHub API token")] = ""

    # Job queue configuration
    stale_timeout_sec: Annotated[
        int, Field(description="Seconds before a job is considered stale")
    ] = 300  # 5 minutes
    freshness_check_interval: Annotated[
        int, Field(description="Base scheduler interval in seconds for docs freshness checks")
    ] = 3600

    # Indexing configuration
    chunk_max_tokens: Annotated[int, Field(description="Maximum tokens per chunk")] = 800
    merge_min_tokens: Annotated[int, Field(description="Merge chunks smaller than this")] = 100
    embedding_batch_size: Annotated[int, Field(description="Batch size for embeddings")] = 32

    # Search relevance boosting by document category (applied post-RRF)
    # Higher values = more boost. 1.0 = baseline (no change)
    doc_category_boosts: Annotated[
        dict[str, float],
        Field(description="Boost multipliers by document category"),
    ] = {
        "reference": 1.25,  # +25% for canonical API/SDK reference docs
        "guide": 1.0,  # Baseline for tutorials/getting-started
        "example": 0.9,  # -10% for cookbooks/examples
        "changelog": 0.6,  # -40% for release notes/changelogs
        "blog": 0.7,  # -30% for blog posts/announcements
        "other": 1.0,  # Baseline for uncategorized
    }

    # Path hierarchy boosting - prioritize core docs over integrations/blog
    # Patterns are matched with re.search (substring match)
    # Order matters for priority - first match is used
    path_hierarchy_boosts: Annotated[
        dict[str, float],
        Field(description="Boost multipliers by path pattern (regex)"),
    ] = {
        # SDK docs - highest priority
        r"pages/docs/sdk/": 1.25,  # Langfuse-style SDK docs: +25%
        r"(?:^|/)docs/sdk/": 1.25,  # Generic SDK docs: +25%
        # Overview files - boost core explanatory docs
        r"overview\.mdx?$": 1.15,  # Overview files: +15%
        # Integration docs - match both /docs/integrations/ and /integrations/
        r"pages/docs/integrations/": 1.20,  # Langfuse-style API integrations: +20%
        r"(?:^|/)docs/integrations/": 1.20,  # Generic API integrations: +20%
        r"pages/integrations/": 1.15,  # Langfuse framework integrations: +15%
        r"(?:^|/)integrations/frameworks/": 1.15,  # Framework integrations: +15%
        # Core documentation (fallback for other /docs/ paths)
        r"(?:^|/)docs/": 1.10,  # Core documentation: +10%
        r"pages/docs/": 1.10,  # Langfuse-style core docs
        # Other content types
        r"(?:^|/)guides?/cookbook/": 0.95,  # Cookbooks: -5%
        r"(?:^|/)blog/": 0.80,  # Blog: -20%
        r"(?:^|/)changelog/": 0.70,  # Changelog directory: -30%
        r"(?:^|/)faq/": 0.90,  # FAQ: -10%
    }

    # Heading level boosting - prioritize H1/H2 over H4/H5/H6
    # Deeper headings are less authoritative
    heading_level_boosts: Annotated[
        dict[int, float],
        Field(description="Boost multipliers by heading level (1-6)"),
    ] = {
        1: 1.15,  # H1: +15%
        2: 1.10,  # H2: +10%
        3: 1.0,  # H3: baseline
        4: 0.95,  # H4: -5%
        5: 0.95,  # H5: -5%
        6: 0.95,  # H6: -5%
    }

    # Token count thresholds for content quality boosting
    token_count_thresholds: Annotated[
        dict[str, int],
        Field(description="Token count thresholds for boost/penalty"),
    ] = {
        "min_substantive": 400,  # Above this: +5% boost
        "max_fragment": 100,  # Below this: -10% penalty
    }

    # Token count boost multipliers
    token_count_boosts: Annotated[
        dict[str, float],
        Field(description="Boost multipliers for token count tiers"),
    ] = {
        "substantive": 1.05,  # >400 tokens: +5%
        "fragment": 0.90,  # <100 tokens: -10%
        "normal": 1.0,  # 100-400 tokens: baseline
    }

    # File type boosting - prioritize documentation over source code (conservative)
    # Higher values = more boost. 1.0 = baseline (no change)
    file_type_boosts: Annotated[
        dict[str, float],
        Field(description="Boost multipliers by file type (docs vs source)"),
    ] = {
        "documentation": 1.20,  # +20% for markdown/rst docs
        "example": 1.15,  # +15% for example code
        "config": 1.0,  # Baseline for config files
        "source": 0.90,  # -10% for raw source code
        "test": 0.70,  # -30% for test files (indexed but penalized)
        "other": 1.0,  # Baseline
    }

    # Intent-aware file type adjustments (multiplied on top of base file_type_boosts)
    # Only specified intents have adjustments; others use base boosts
    intent_file_type_boosts: Annotated[
        dict[str, dict[str, float]],
        Field(description="Intent-specific file type boost adjustments"),
    ] = {
        "api_lookup": {
            "documentation": 1.10,  # Slight docs preference
            "source": 1.05,  # Minor boost (might want signature)
        },
        "how_to": {
            "documentation": 1.15,  # Docs preference
            "example": 1.20,  # Examples even better for how-to
            "source": 0.90,  # Slight source penalty
        },
        "implementation": {
            "source": 1.25,  # User explicitly wants source
            "documentation": 0.95,  # Slight docs penalty
        },
        # error_debug and concept use base boosts (no adjustments)
    }

    @model_validator(mode="after")
    def apply_env_fallbacks(self) -> "Settings":
        """Apply fallbacks for common env vars without prefix."""
        if not self.github_token:
            self.github_token = os.environ.get("GITHUB_TOKEN", "")
        if not self.postgres_dsn:
            self.postgres_dsn = os.environ.get("DATABASE_URL", "")
        if not self.supabase_service_key:
            self.supabase_service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the settings singleton."""
    return Settings()
