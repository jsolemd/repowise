"""Pydantic models for doc-search library management."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from repowise.docs.formats import DEFAULT_INCLUDE_PATTERNS


class LibraryStatus(StrEnum):
    """Library indexing status."""

    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    ERROR = "error"


class LibrarySourceType(StrEnum):
    """Where the current docs content for a library comes from."""

    GIT = "git"
    SNAPSHOT = "snapshot"


class JobType(StrEnum):
    """Index job type."""

    FULL = "full"
    INCREMENTAL = "incremental"
    FORCE = "force"


class JobStatus(StrEnum):
    """Index job status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobEnqueueDisposition(StrEnum):
    """How a job request affected the queue."""

    QUEUED = "queued"
    COALESCED = "coalesced"
    UPGRADED = "upgraded"


class LibraryConfig(BaseModel):
    """Library configuration from YAML seed file."""

    source_type: LibrarySourceType = Field(
        LibrarySourceType.GIT,
        description="Backing source provider for this library",
    )
    library_id: str | None = Field(
        None,
        description="Stable logical library identifier (e.g., /codeatlas/cosmograph)",
    )
    repo: str = Field("", description="GitHub repo path (e.g., langfuse/langfuse)")
    name: str = Field(..., description="Human-readable name")
    description: str | None = Field(None, description="Library description")
    legacy_ids: list[str] = Field(
        default_factory=list,
        description="Older logical library IDs that should migrate to this library",
    )
    source_subpath: str = Field(
        "",
        description="Optional subdirectory within the source repo that scopes this library",
    )
    docs_path: str = Field("", description="Path to docs within repo")
    branch: str = Field("main", description="Git branch to index")
    include_patterns: list[str] = Field(
        default_factory=lambda: list(DEFAULT_INCLUDE_PATTERNS),
        description="File patterns to include",
    )
    exclude_patterns: list[str] = Field(
        default_factory=list,
        description="File patterns to exclude. Default patterns for old versions (v1/, legacy/, etc.) are always added.",
    )
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Scheduling priority (1-3=hourly, 4-6=6h, 7-10=daily)",
    )

    @model_validator(mode="after")
    def _normalize(self) -> "LibraryConfig":
        self.repo = _normalize_repo(self.repo)
        self.legacy_ids = _normalize_library_ids(self.legacy_ids)
        self.source_subpath = _normalize_rel_path(self.source_subpath)
        self.docs_path = _normalize_rel_path(self.docs_path)
        if self.source_type == LibrarySourceType.SNAPSHOT and not (self.library_id or "").strip():
            raise ValueError("library_id is required for snapshot-backed documentation libraries")
        self.library_id = _normalize_library_id(self.library_id, self.repo, self.source_subpath)
        self.legacy_ids = [
            legacy_id for legacy_id in self.legacy_ids if legacy_id != self.library_id
        ]
        if self.source_type == LibrarySourceType.GIT and not self.repo:
            raise ValueError("repo is required for git-backed documentation libraries")
        return self


class LibraryState(BaseModel):
    """Library state from database."""

    source_type: LibrarySourceType = Field(
        LibrarySourceType.GIT,
        description="Backing source provider for this library",
    )
    library_id: str = Field(..., description="Stable logical library ID")
    repo: str = Field("", description="GitHub repo path")
    name: str = Field(..., description="Human-readable name")
    description: str | None = Field(None, description="Library description")
    source_subpath: str = Field(
        "",
        description="Optional subdirectory within the source repo that scopes this library",
    )
    docs_path: str = Field("", description="Path to docs within repo")
    branch: str = Field("main", description="Git branch to index")
    include_patterns: list[str] = Field(default_factory=list)
    exclude_patterns: list[str] = Field(default_factory=list)
    priority: int = Field(
        5,
        ge=1,
        le=10,
        description="Scheduling priority (1-3=hourly, 4-6=6h, 7-10=daily)",
    )

    # Indexing state
    status: LibraryStatus = Field(LibraryStatus.PENDING)
    current_sha: str | None = Field(
        None,
        description="Current indexed source ref (commit SHA for whole-repo libs, tree SHA for subpath libs)",
    )
    indexed_at: datetime | None = Field(None, description="Last index time")
    freshness_checked_at: datetime | None = Field(
        None,
        description="Last scheduler freshness probe or retry-cadence update time",
    )
    next_freshness_check_at: datetime | None = Field(
        None,
        description="Persisted next scheduler probe time to avoid restart burst checks",
    )
    last_freshness_state: str | None = Field(
        None,
        description="Last freshness outcome: fresh, stale, or unknown",
    )
    last_remote_sha: str | None = Field(
        None,
        description="Most recent upstream source ref observed during freshness checks",
    )
    last_freshness_error: str | None = Field(
        None,
        description="Most recent freshness-check failure or warning",
    )
    error_message: str | None = Field(None, description="Last indexing error message")
    graph_synced_at: datetime | None = Field(
        None,
        description="Last successful docs metadata sync into Neo4j",
    )
    graph_sync_error: str | None = Field(
        None,
        description="Last docs graph sync error",
    )

    # Statistics
    chunk_count: int = Field(0, description="Number of chunks in Qdrant")
    file_count: int = Field(0, description="Number of indexed files")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def _normalize(self) -> "LibraryState":
        self.repo = _normalize_repo(self.repo)
        self.source_subpath = _normalize_rel_path(self.source_subpath)
        self.docs_path = _normalize_rel_path(self.docs_path)
        self.library_id = _normalize_library_id(self.library_id, self.repo, self.source_subpath)
        if self.source_type == LibrarySourceType.GIT and not self.repo:
            raise ValueError("repo is required for git-backed documentation libraries")
        return self


class LibraryFile(BaseModel):
    """File tracking record for incremental indexing."""

    library_id: str = Field(..., description="Library ID")
    file_path: str = Field(..., description="Relative path within repo")
    content_hash: str = Field(..., description="SHA256 hash of file content")
    chunk_count: int = Field(0, description="Number of chunks from this file")
    indexed_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class IndexJob(BaseModel):
    """Index job from the job queue."""

    id: str = Field(..., description="Job UUID")
    library_id: str = Field(..., description="Library ID")
    job_type: JobType = Field(JobType.INCREMENTAL)
    priority: int = Field(5, ge=1, le=10)

    # Job state
    status: JobStatus = Field(JobStatus.PENDING)
    worker_id: str | None = Field(None)
    claimed_at: datetime | None = Field(None)
    heartbeat_at: datetime | None = Field(None)
    started_at: datetime | None = Field(None)
    completed_at: datetime | None = Field(None)
    error_message: str | None = Field(None)

    # Progress
    files_processed: int = Field(0)
    files_total: int = Field(0)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SnapshotState(BaseModel):
    """Current snapshot metadata for a snapshot-backed library."""

    library_id: str = Field(..., description="Library ID")
    source_ref: str = Field(..., description="Current snapshot source ref")
    manifest_hash: str = Field(..., description="Current snapshot manifest hash")
    file_count: int = Field(0, description="Current snapshot file count")
    published_at: datetime = Field(default_factory=datetime.utcnow)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SnapshotFile(BaseModel):
    """Current file content for a snapshot-backed library."""

    library_id: str = Field(..., description="Library ID")
    file_path: str = Field(..., description="Relative path within the logical library")
    content: str = Field(..., description="Current file content")
    content_hash: str = Field(..., description="SHA256 hash of current content")
    source_url: str | None = Field(None, description="Optional upstream source URL")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class JobEnqueueResult(BaseModel):
    """Outcome of a job enqueue request."""

    disposition: JobEnqueueDisposition
    job: IndexJob


class LibrariesConfig(BaseModel):
    """Root model for libraries.yaml."""

    libraries: list[LibraryConfig] = Field(default_factory=list)


def _normalize_repo(repo: str) -> str:
    return repo.strip().strip("/")


def _normalize_rel_path(path: str | None) -> str:
    value = (path or "").strip().replace("\\", "/")
    if value in {"", "."}:
        return ""
    return value.strip("/")


def _normalize_library_id(
    library_id: str | None,
    repo: str,
    source_subpath: str = "",
) -> str:
    value = (library_id or "").strip()
    if not value:
        if source_subpath:
            return f"/{repo}/{source_subpath}".replace("//", "/")
        return f"/{repo}"
    if not value.startswith("/"):
        value = f"/{value}"
    return value.rstrip("/") or "/"


def _normalize_library_ids(values: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        if not str(value).strip():
            continue
        normalized_value = _normalize_library_id(value, repo="", source_subpath="")
        if normalized_value in seen:
            continue
        normalized.append(normalized_value)
        seen.add(normalized_value)
    return normalized
