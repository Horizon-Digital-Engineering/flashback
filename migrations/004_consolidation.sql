-- Audit log for the consolidation worker.
--
-- Each row records one run of `run_daily` / `run_weekly` / `run_monthly` —
-- when it ran, how long it took, what it did. Admin UI surfaces recent runs.
-- The worker can also use this to skip "already ran today" idempotency.

CREATE TABLE IF NOT EXISTS consolidation_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kind            TEXT NOT NULL CHECK (kind IN ('daily','weekly','monthly')),
    trigger         TEXT NOT NULL CHECK (trigger IN ('scheduled','manual')),
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    promoted_count  INT NOT NULL DEFAULT 0,
    expired_count   INT NOT NULL DEFAULT 0,
    distilled_count INT NOT NULL DEFAULT 0,
    clusters_seen   INT NOT NULL DEFAULT 0,
    user_id         TEXT,
    error           TEXT,
    notes           JSONB
);

CREATE INDEX IF NOT EXISTS consolidation_runs_started_idx
    ON consolidation_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS consolidation_runs_kind_user_idx
    ON consolidation_runs (kind, user_id, started_at DESC);
