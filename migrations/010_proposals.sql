-- ---------------------------------------------------------------------------
-- Proposals: the lake proposes, the host executes, the operator decides.
-- There is deliberately no code path by which the lake acts — the absence is
-- the enforcement. Lifecycle: proposed → approved → executed | denied.
-- ---------------------------------------------------------------------------
CREATE TABLE proposals (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('action','insight')),
    title       TEXT NOT NULL,
    body        JSONB NOT NULL,            -- {action, rationale, evidence:[record_id,...]}
    status      TEXT NOT NULL DEFAULT 'proposed'
                CHECK (status IN ('proposed','approved','denied','executed')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at  TIMESTAMPTZ,
    decided_by  TEXT
);
CREATE INDEX proposals_user_status_idx ON proposals (user_id, status);
