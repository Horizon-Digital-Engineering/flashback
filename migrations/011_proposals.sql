-- PROPOSALS — the "propose, don't act" surface of the Operation layer.
--
-- The lake never executes. It surfaces a decision or an action as a PROPOSAL
-- citing the evidence (raw record ids) that justifies it, and the operator/host
-- decides. Execution is entirely the host's job: the host reports completion
-- back via a status transition. There is deliberately NO column, endpoint, or
-- code path by which the lake itself carries out an action — the absence is the
-- enforcement.
--
-- Lifecycle: proposed → approved → executed, with a proposed → denied terminal.
--   proposed  the lake surfaced it; awaiting a human/host decision.
--   approved  the operator approved it; the host may now act on it.
--   denied    the operator rejected it (terminal).
--   executed  the host reports it carried the action out (terminal).

CREATE TABLE IF NOT EXISTS proposals (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     TEXT NOT NULL,
    kind        TEXT NOT NULL CHECK (kind IN ('action','insight')),
    title       TEXT NOT NULL,
    -- {action, rationale, evidence:[record_id,...]} — the proposal body. The
    -- cited evidence is validated (exists + owned) at create time.
    body        JSONB NOT NULL,
    status      TEXT NOT NULL DEFAULT 'proposed'
                CHECK (status IN ('proposed','approved','denied','executed')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at  TIMESTAMPTZ,
    decided_by  TEXT
);

-- The review-queue read: "this user's proposals in a given status".
CREATE INDEX IF NOT EXISTS proposals_user_status_idx
    ON proposals (user_id, status);
