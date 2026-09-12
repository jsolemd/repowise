-- Doc-search schema initialization (standalone PostgreSQL)
-- Combines 001_initial_schema.sql + 002_add_library_priority.sql
-- + 003_add_freshness_checked_at.sql
-- Stripped of Supabase-specific grants/RLS.

CREATE SCHEMA IF NOT EXISTS doc_search;

-- ============================================================================
-- doc_search.libraries - Library registry
-- ============================================================================
CREATE TABLE doc_search.libraries (
    library_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL DEFAULT 'git'
        CHECK (source_type IN ('git', 'snapshot')),
    repo TEXT NOT NULL DEFAULT '',
    name TEXT NOT NULL,
    description TEXT,
    source_subpath TEXT DEFAULT '',
    docs_path TEXT DEFAULT '',
    branch TEXT DEFAULT 'main',
    include_patterns TEXT[] DEFAULT ARRAY[
        '**/*.md', '**/*.mdx', '**/*.rst', '**/*.adoc',
        '**/*.java', '**/*.gradle', '**/*.yaml', '**/*.json'
    ],
    exclude_patterns TEXT[] DEFAULT ARRAY[]::TEXT[],
    priority INT DEFAULT 5 CHECK (priority BETWEEN 1 AND 10),

    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'indexing', 'ready', 'error')),
    current_sha TEXT,
    indexed_at TIMESTAMPTZ,
    freshness_checked_at TIMESTAMPTZ,
    next_freshness_check_at TIMESTAMPTZ,
    last_freshness_state TEXT
        CHECK (last_freshness_state IN ('fresh', 'stale', 'unknown')),
    last_remote_sha TEXT,
    last_freshness_error TEXT,
    error_message TEXT,
    graph_synced_at TIMESTAMPTZ,
    graph_sync_error TEXT,

    chunk_count INT DEFAULT 0,
    file_count INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_libraries_status ON doc_search.libraries (status);
CREATE INDEX idx_libraries_priority ON doc_search.libraries (priority);
CREATE INDEX idx_libraries_freshness_checked_at ON doc_search.libraries (freshness_checked_at);
CREATE INDEX idx_libraries_next_freshness_check_at ON doc_search.libraries (next_freshness_check_at);
CREATE INDEX idx_libraries_source_type ON doc_search.libraries (source_type);

-- ============================================================================
-- doc_search.library_files - Per-file hash tracking
-- ============================================================================
CREATE TABLE doc_search.library_files (
    library_id TEXT NOT NULL REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    chunk_count INT DEFAULT 0,
    indexed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (library_id, file_path)
);

CREATE INDEX idx_library_files_library ON doc_search.library_files (library_id);

-- ============================================================================
-- doc_search.snapshot_states - Current internal snapshot metadata
-- ============================================================================
CREATE TABLE doc_search.snapshot_states (
    library_id TEXT PRIMARY KEY REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
    source_ref TEXT NOT NULL,
    manifest_hash TEXT NOT NULL,
    file_count INT NOT NULL DEFAULT 0,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_snapshot_states_published_at ON doc_search.snapshot_states (published_at);

-- ============================================================================
-- doc_search.snapshot_files - Current internal file content for snapshot sources
-- ============================================================================
CREATE TABLE doc_search.snapshot_files (
    library_id TEXT NOT NULL REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    source_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (library_id, file_path)
);

CREATE INDEX idx_snapshot_files_library ON doc_search.snapshot_files (library_id);

-- ============================================================================
-- doc_search.index_jobs - Job queue
-- ============================================================================
CREATE TABLE doc_search.index_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library_id TEXT NOT NULL REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
    job_type TEXT NOT NULL DEFAULT 'incremental'
        CHECK (job_type IN ('full', 'incremental', 'force')),
    priority INT DEFAULT 5,

    status TEXT DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    worker_id TEXT,
    claimed_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    error_message TEXT,

    files_processed INT DEFAULT 0,
    files_total INT DEFAULT 0,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_jobs_pending_coalesce
    ON doc_search.index_jobs (library_id)
    WHERE status = 'pending';

CREATE INDEX idx_jobs_claim
    ON doc_search.index_jobs (priority, created_at)
    WHERE status = 'pending';

CREATE INDEX idx_jobs_stale
    ON doc_search.index_jobs (heartbeat_at)
    WHERE status = 'running';

-- ============================================================================
-- doc_search.graph_sync_jobs - Durable docs->Neo4j replay queue
-- ============================================================================
CREATE TABLE IF NOT EXISTS doc_search.graph_sync_jobs (
    library_id TEXT PRIMARY KEY,
    action TEXT NOT NULL
        CHECK (action IN ('upsert', 'delete')),
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'failed')),
    attempts INT NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_error TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_graph_sync_jobs_claim
    ON doc_search.graph_sync_jobs (status, next_attempt_at, updated_at);

-- ============================================================================
-- updated_at triggers
-- ============================================================================
CREATE OR REPLACE FUNCTION doc_search.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER libraries_updated_at
    BEFORE UPDATE ON doc_search.libraries
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

CREATE TRIGGER library_files_updated_at
    BEFORE UPDATE ON doc_search.library_files
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

CREATE TRIGGER snapshot_states_updated_at
    BEFORE UPDATE ON doc_search.snapshot_states
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

CREATE TRIGGER snapshot_files_updated_at
    BEFORE UPDATE ON doc_search.snapshot_files
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

CREATE TRIGGER index_jobs_updated_at
    BEFORE UPDATE ON doc_search.index_jobs
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

CREATE TRIGGER graph_sync_jobs_updated_at
    BEFORE UPDATE ON doc_search.graph_sync_jobs
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

COMMENT ON SCHEMA doc_search IS 'Documentation search MCP server';
