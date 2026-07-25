-- ---------------------------------------------------------------------------
-- Auth. Tokens are sha256-hashed at rest; plaintext shown once at mint.
-- Roles split the two surfaces:
--   service  — the REST/MCP API (ingest, query, context). What integrations hold.
--   operator — the /admin UI; sees the whole estate, cannot call the API.
-- The middleware enforces the wall both ways.
-- ---------------------------------------------------------------------------
CREATE TABLE tokens (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_hash    TEXT NOT NULL UNIQUE,
    token_prefix  TEXT NOT NULL,           -- first 11 chars, shown in `token list`
    user_id       TEXT NOT NULL,
    name          TEXT,
    role          TEXT NOT NULL DEFAULT 'service' CHECK (role IN ('service','operator')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);
CREATE INDEX tokens_user_idx ON tokens (user_id) WHERE revoked_at IS NULL;
CREATE INDEX tokens_hash_idx ON tokens (token_hash) WHERE revoked_at IS NULL;
