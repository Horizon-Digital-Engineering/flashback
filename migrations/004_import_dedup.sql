-- Idempotent bulk import. A record carrying a stable external id (`source_ref`,
-- e.g. the message id from a ChatGPT/Claude export) dedups on
-- (user_id, source, source_ref) so re-running an import never duplicates.
-- Normal ingest (no source_ref) is unconstrained — legitimate repeat content
-- ("ok", "thanks") is still allowed.
CREATE UNIQUE INDEX IF NOT EXISTS raw_records_import_dedup_idx
    ON raw_records (user_id, source, source_ref)
    WHERE source_ref IS NOT NULL;
