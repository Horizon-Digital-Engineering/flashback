-- ---------------------------------------------------------------------------
-- Conclusions the system drew about raw records. Wiped and re-derived freely;
-- nothing here is testimony. Each row names WHO decided and WHEN, so a bad
-- decision is traceable to the thing that made it rather than anonymous.
-- ---------------------------------------------------------------------------

-- Which embedding register a record belongs to. It was a column on raw, where a
-- misclassification was permanent and made the record invisible rather than
-- merely mis-ranked. The FK to `modes` is the point: it is impossible on an
-- append-only table, and without it a typo'd register silently orphans a record.
CREATE TABLE derived_record_mode (
    record_id  UUID PRIMARY KEY REFERENCES raw_records(id) ON DELETE CASCADE,
    user_id    TEXT NOT NULL,
    mode       TEXT NOT NULL,
    embedder   TEXT NOT NULL,
    -- Copied from the record so mode-scoped recency does not need the join to
    -- probe raw per row. A copy in a rebuildable table is not a second truth.
    event_time TIMESTAMPTZ NOT NULL,
    origin     TEXT NOT NULL CHECK (origin IN ('writer','llm','default','human')),
    confidence REAL,
    decided_by TEXT NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY (user_id, mode) REFERENCES modes (user_id, name)
);
CREATE INDEX derived_record_mode_scope_time_idx
    ON derived_record_mode (user_id, mode, event_time DESC);

-- Which records are no longer current. Two sources: the writer's own claim on
-- raw, and the server's inference that a newer row won a (kind, key). Keeping
-- both here makes the exclusion an index-only anti-join instead of rebuilding
-- the superseded set on every query.
CREATE TABLE derived_superseded (
    record_id     UUID PRIMARY KEY REFERENCES raw_records(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL,
    superseded_by UUID NOT NULL REFERENCES raw_records(id),
    origin        TEXT NOT NULL CHECK (origin IN ('writer','inferred')),
    decided_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX derived_superseded_user_idx ON derived_superseded (user_id);

-- The causal edge: which record actually followed which. Resolved from the
-- writer's `prev_source_ref`, and re-resolved on every rebuild — so a parent
-- that arrives late gets linked instead of leaving a permanent gap.
CREATE TABLE derived_link (
    record_id  UUID PRIMARY KEY REFERENCES raw_records(id) ON DELETE CASCADE,
    user_id    TEXT NOT NULL,
    prev_id    UUID NOT NULL REFERENCES raw_records(id),
    resolved_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX derived_link_prev_idx ON derived_link (prev_id);
