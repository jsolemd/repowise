ALTER TABLE doc_search.libraries
    ADD COLUMN IF NOT EXISTS freshness_checked_at TIMESTAMPTZ;

UPDATE doc_search.libraries
SET freshness_checked_at = COALESCE(freshness_checked_at, indexed_at)
WHERE freshness_checked_at IS NULL
  AND indexed_at IS NOT NULL;
