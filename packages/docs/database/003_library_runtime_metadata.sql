-- Add persisted runtime metadata for freshness and docs->Neo4j sync.

ALTER TABLE doc_search.libraries
ADD COLUMN IF NOT EXISTS last_freshness_state TEXT
    CHECK (last_freshness_state IN ('fresh', 'stale', 'unknown'));

ALTER TABLE doc_search.libraries
ADD COLUMN IF NOT EXISTS last_remote_sha TEXT;

ALTER TABLE doc_search.libraries
ADD COLUMN IF NOT EXISTS last_freshness_error TEXT;

ALTER TABLE doc_search.libraries
ADD COLUMN IF NOT EXISTS graph_synced_at TIMESTAMPTZ;

ALTER TABLE doc_search.libraries
ADD COLUMN IF NOT EXISTS graph_sync_error TEXT;

CREATE INDEX IF NOT EXISTS idx_libraries_freshness_checked_at
    ON doc_search.libraries (freshness_checked_at);
