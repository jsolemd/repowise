-- Doc-search schema migrations
-- Creates the doc_search schema with tables for library registry,
-- file tracking, and job queue.

-- Create schema
CREATE SCHEMA IF NOT EXISTS doc_search;

-- ============================================================================
-- doc_search.libraries - Library registry
-- ============================================================================
-- Stores metadata about each documentation library (GitHub repo).
-- library_id format: /{owner}/{repo} (e.g., /langfuse/langfuse)
CREATE TABLE doc_search.libraries (
    library_id TEXT PRIMARY KEY,
    repo TEXT NOT NULL,                        -- GitHub repo path (e.g., langfuse/langfuse)
    name TEXT NOT NULL,                        -- Human-readable name
    description TEXT,                          -- Library description
    docs_path TEXT DEFAULT '',                 -- Path to docs within repo (empty = root)
    branch TEXT DEFAULT 'main',                -- Git branch to index
    include_patterns TEXT[] DEFAULT ARRAY[
        '**/*.md',
        '**/*.mdx',
        '**/*.rst',
        '**/*.adoc',
        '**/*.java',
        '**/*.gradle',
        '**/*.yaml',
        '**/*.json'
    ],  -- File patterns to include
    exclude_patterns TEXT[] DEFAULT ARRAY[]::TEXT[],  -- File patterns to exclude

    -- Indexing state
    status TEXT DEFAULT 'pending'              -- pending, indexing, ready, error
        CHECK (status IN ('pending', 'indexing', 'ready', 'error')),
    current_sha TEXT,                          -- Current indexed commit SHA
    indexed_at TIMESTAMPTZ,                    -- Last successful index time
    freshness_checked_at TIMESTAMPTZ,          -- Last scheduler freshness probe time
    next_freshness_check_at TIMESTAMPTZ,       -- Persisted next freshness probe time
    last_freshness_state TEXT
        CHECK (last_freshness_state IN ('fresh', 'stale', 'unknown')),
    last_remote_sha TEXT,
    last_freshness_error TEXT,
    error_message TEXT,                        -- Last error message (if status = error)
    graph_synced_at TIMESTAMPTZ,
    graph_sync_error TEXT,

    -- Statistics
    chunk_count INT DEFAULT 0,                 -- Number of chunks in Qdrant
    file_count INT DEFAULT 0,                  -- Number of indexed files

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for listing libraries by status
CREATE INDEX idx_libraries_status ON doc_search.libraries (status);
CREATE INDEX idx_libraries_freshness_checked_at ON doc_search.libraries (freshness_checked_at);
CREATE INDEX idx_libraries_next_freshness_check_at ON doc_search.libraries (next_freshness_check_at);

-- ============================================================================
-- doc_search.library_files - Per-file hash tracking
-- ============================================================================
-- Tracks content hashes for incremental indexing.
-- If a file's hash hasn't changed, skip re-indexing.
CREATE TABLE doc_search.library_files (
    library_id TEXT NOT NULL REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
    file_path TEXT NOT NULL,                   -- Relative path within repo
    content_hash TEXT NOT NULL,                -- SHA256 hash of file content
    chunk_count INT DEFAULT 0,                 -- Number of chunks from this file
    indexed_at TIMESTAMPTZ DEFAULT NOW(),      -- When this file was last indexed

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (library_id, file_path)
);

-- Index for listing files by library
CREATE INDEX idx_library_files_library ON doc_search.library_files (library_id);

-- ============================================================================
-- doc_search.index_jobs - Job queue
-- ============================================================================
-- Durable job queue with heartbeat-based stale detection.
-- Jobs are coalesced: multiple requests for the same library become one job.
CREATE TABLE doc_search.index_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    library_id TEXT NOT NULL REFERENCES doc_search.libraries(library_id) ON DELETE CASCADE,
    job_type TEXT NOT NULL DEFAULT 'incremental'  -- full, incremental, force
        CHECK (job_type IN ('full', 'incremental', 'force')),
    priority INT DEFAULT 5,                    -- Lower = higher priority (1-10)

    -- Job state
    status TEXT DEFAULT 'pending'              -- pending, running, completed, failed
        CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    worker_id TEXT,                            -- Worker that claimed this job
    claimed_at TIMESTAMPTZ,                    -- When job was claimed
    heartbeat_at TIMESTAMPTZ,                  -- Last heartbeat from worker
    started_at TIMESTAMPTZ,                    -- When processing started
    completed_at TIMESTAMPTZ,                  -- When processing completed
    error_message TEXT,                        -- Error message (if status = failed)

    -- Progress tracking
    files_processed INT DEFAULT 0,
    files_total INT DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Coalescing index: Only one pending/running job per library
-- This allows enqueueing a job to be idempotent
CREATE UNIQUE INDEX idx_jobs_pending_coalesce
    ON doc_search.index_jobs (library_id)
    WHERE status = 'pending';

-- Claim index: Find next job to claim efficiently
-- Orders by priority (lower first), then creation time
CREATE INDEX idx_jobs_claim
    ON doc_search.index_jobs (priority, created_at)
    WHERE status = 'pending';

-- Stale detection index: Find jobs that haven't heartbeated recently
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

CREATE INDEX IF NOT EXISTS idx_graph_sync_jobs_claim
    ON doc_search.graph_sync_jobs (status, next_attempt_at, updated_at);

-- ============================================================================
-- Triggers for updated_at
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

CREATE TRIGGER index_jobs_updated_at
    BEFORE UPDATE ON doc_search.index_jobs
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

CREATE TRIGGER graph_sync_jobs_updated_at
    BEFORE UPDATE ON doc_search.graph_sync_jobs
    FOR EACH ROW EXECUTE FUNCTION doc_search.update_updated_at();

-- ============================================================================
-- Comments
-- ============================================================================
COMMENT ON SCHEMA doc_search IS 'Documentation search MCP server schema';
COMMENT ON TABLE doc_search.libraries IS 'Registry of GitHub documentation libraries';
COMMENT ON TABLE doc_search.library_files IS 'Per-file content hashes for incremental indexing';
COMMENT ON TABLE doc_search.index_jobs IS 'Durable job queue for indexing tasks';
COMMENT ON TABLE doc_search.graph_sync_jobs IS 'Durable replay queue for docs-to-Neo4j metadata sync';

-- ============================================================================
-- Permissions (for Supabase PostgREST access)
-- ============================================================================

-- Grant usage on schema
GRANT USAGE ON SCHEMA doc_search TO anon, authenticated, service_role;

-- Grant select on all tables
GRANT SELECT ON ALL TABLES IN SCHEMA doc_search TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA doc_search TO service_role;

-- Grant for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA doc_search
  GRANT SELECT ON TABLES TO anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA doc_search
  GRANT ALL ON TABLES TO service_role;

-- Enable Row Level Security
ALTER TABLE doc_search.libraries ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_search.library_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_search.index_jobs ENABLE ROW LEVEL SECURITY;
ALTER TABLE doc_search.graph_sync_jobs ENABLE ROW LEVEL SECURITY;

-- Read access for anon/authenticated
CREATE POLICY "Allow read access" ON doc_search.libraries
  FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Allow read access" ON doc_search.library_files
  FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Allow read access" ON doc_search.index_jobs
  FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY "Allow read access" ON doc_search.graph_sync_jobs
  FOR SELECT TO anon, authenticated USING (true);

-- Service role can do everything (bypasses RLS)
CREATE POLICY "Service role full access" ON doc_search.libraries
  FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON doc_search.library_files
  FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON doc_search.index_jobs
  FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role full access" ON doc_search.graph_sync_jobs
  FOR ALL TO service_role USING (true) WITH CHECK (true);
