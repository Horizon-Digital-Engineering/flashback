-- ---------------------------------------------------------------------------
-- Playground settings — per-operator, server-side.
--
-- These lived in browser localStorage first, which broke the moment you moved
-- between origins (127.0.0.1 and localhost are different stores) or machines,
-- and made "the page seems dead" the failure mode. The server has a database;
-- settings belong in it.
--
-- IF NOT EXISTS because this table may predate the migration on installs where
-- it was created by hand during bring-up; the statement is idempotent either way.
--
-- The API key deliberately does NOT live here. A third-party credential in
-- plaintext would land in every pg_dump the nightly backup writes, which is a
-- real downgrade from how `tokens` handles secrets (hashed, never recoverable).
-- Local model servers don't need one; when a hosted endpoint does, it stays in
-- the browser.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS playground_settings (
    user_id       TEXT PRIMARY KEY,
    base_url      TEXT,
    model         TEXT,
    system_prompt TEXT,
    -- How many memories to inject per turn. More is not better: every extra
    -- record is prompt tokens and latency, and dilutes what actually matters.
    context_limit INT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
