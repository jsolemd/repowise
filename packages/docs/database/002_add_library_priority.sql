-- Add priority column to libraries table
-- Priority determines how frequently the scheduler checks for updates:
--   1-3 (high): Check every 1 hour
--   4-6 (medium/default): Check every 6 hours
--   7-10 (low): Check every 24 hours

-- Add priority column with default of 5 (medium)
ALTER TABLE doc_search.libraries
ADD COLUMN IF NOT EXISTS priority INT DEFAULT 5
    CHECK (priority BETWEEN 1 AND 10);

-- Create index for priority-based scheduling queries
CREATE INDEX IF NOT EXISTS idx_libraries_priority
    ON doc_search.libraries (priority);

-- Comment explaining the priority system
COMMENT ON COLUMN doc_search.libraries.priority IS
    'Scheduling priority (1=highest, 10=lowest). Determines freshness check frequency: 1-3=hourly, 4-6=6h, 7-10=daily.';
