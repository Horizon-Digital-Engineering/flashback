-- ---------------------------------------------------------------------------
-- System provider settings — one row, server-wide.
--
-- The extraction/distillation provider was configurable only through
-- environment variables read once at startup, which meant every model change
-- was an SSH session and a service restart. These columns are the runtime
-- override: NULL means "inherit the environment value", a saved value wins
-- over the environment from the moment it is applied. The environment stays
-- the bootstrap seed so a fresh install still starts with zero database state.
--
-- Distinct from playground_settings, which is a per-operator sandbox for the
-- playground page and configures nothing about the real pipeline.
--
-- The API key deliberately does NOT live here, for the same reason it is
-- absent from playground_settings: a plaintext credential would land in every
-- pg_dump. Keys stay in the environment; local model servers need none.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS system_settings (
    -- Single-row table: the primary key is a constant so a second row is a
    -- constraint violation rather than a silent ambiguity.
    id                 BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    provider           TEXT CHECK (provider IN ('heuristic', 'remote')),
    remote_backend     TEXT CHECK (remote_backend IN ('openai', 'anthropic', 'openrouter')),
    api_base           TEXT,
    extract_model      TEXT,
    distill_model      TEXT,
    extract_timeout_ms INT CHECK (extract_timeout_ms BETWEEN 1000 AND 600000),
    distill_timeout_ms INT CHECK (distill_timeout_ms BETWEEN 1000 AND 600000),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
