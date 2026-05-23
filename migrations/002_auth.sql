-- Bearer-token auth. Tokens are sha256-hashed at rest.
-- The plaintext is shown ONCE at mint time and never stored or logged.
-- See `flashback token mint --user=<user> --name=<label>`.

CREATE TABLE IF NOT EXISTS tokens (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_hash    TEXT NOT NULL UNIQUE,
    token_prefix  TEXT NOT NULL,           -- first 11 chars ("fb_XXXXXXXX") shown in `token list`
    user_id       TEXT NOT NULL,
    name          TEXT,                    -- e.g. "claude-code laptop", "cursor work"
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at  TIMESTAMPTZ,
    revoked_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS tokens_user_idx
    ON tokens (user_id) WHERE revoked_at IS NULL;

CREATE INDEX IF NOT EXISTS tokens_hash_idx
    ON tokens (token_hash) WHERE revoked_at IS NULL;
