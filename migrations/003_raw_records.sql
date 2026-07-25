-- ---------------------------------------------------------------------------
-- RAW layer: the universal typed record. APPEND-ONLY; corrections are a new
-- row with `supersedes` = old id (forward pointer, old row untouched).
-- "Superseded" is derived: is this id referenced by a newer row's supersedes.
-- state_kind/state_key promote reference identity for type='state_object'
-- rows (filled from payload at INSERT); the terminal row of
-- (user_id, state_kind, state_key) is the current value.
-- ---------------------------------------------------------------------------
CREATE TABLE raw_records (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type          TEXT NOT NULL CHECK (type IN (
        'episodic','semantic','working','document','procedural','state_object'
    )),
    content       TEXT NOT NULL,
    content_hash  TEXT GENERATED ALWAYS AS (md5(content)) STORED,
    content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- dual time: when it actually happened vs when we stored it.
    event_time    TIMESTAMPTZ NOT NULL,
    ingest_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    source        TEXT NOT NULL,           -- origin: 'ritsu:health' | 'chatgpt' | ...
    source_ref    TEXT,                    -- stable external id (import dedup)

    user_id       TEXT NOT NULL,
    project_id    TEXT,
    session_id    TEXT,
    mode          TEXT,

    importance    REAL CHECK (importance IS NULL OR (importance >= 0.0 AND importance <= 1.0)),
    supersedes    UUID REFERENCES raw_records(id),

    state_kind    TEXT,                    -- state_object rows only
    state_key     TEXT,

    acl           JSONB,                   -- null = private
    ttl           TIMESTAMPTZ,             -- optional expiry (working memory)
    payload       JSONB
);

CREATE INDEX raw_records_scope_type_idx  ON raw_records (user_id, project_id, type);
CREATE INDEX raw_records_event_time_idx  ON raw_records (event_time);
CREATE INDEX raw_records_ingest_time_idx ON raw_records (ingest_time);
CREATE INDEX raw_records_session_idx     ON raw_records (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX raw_records_mode_idx        ON raw_records (mode) WHERE mode IS NOT NULL;
CREATE INDEX raw_records_hash_idx        ON raw_records (content_hash);
CREATE INDEX raw_records_supersedes_idx  ON raw_records (supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX raw_records_ttl_idx         ON raw_records (ttl) WHERE ttl IS NOT NULL;
CREATE INDEX raw_records_tsv_idx         ON raw_records USING GIN (content_tsv);

-- Idempotent bulk import: a record carrying a stable external id dedups on
-- (user_id, source, source_ref); normal ingest (no source_ref) is unconstrained.
CREATE UNIQUE INDEX raw_records_import_dedup_idx
    ON raw_records (user_id, source, source_ref)
    WHERE source_ref IS NOT NULL;

-- Reference-identity lookup for state objects.
CREATE INDEX raw_records_state_ident_idx
    ON raw_records (user_id, state_kind, state_key)
    WHERE type = 'state_object';

-- Append-only enforcement: UPDATE, DELETE, and TRUNCATE all raise.
CREATE FUNCTION raw_records_immutable()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'raw_records is append-only (attempted %)', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER raw_records_no_mutate BEFORE UPDATE OR DELETE ON raw_records
    FOR EACH ROW EXECUTE FUNCTION raw_records_immutable();
CREATE TRIGGER raw_records_no_truncate BEFORE TRUNCATE ON raw_records
    FOR EACH STATEMENT EXECUTE FUNCTION raw_records_immutable();

-- Fill state identity from the payload at INSERT only (disjoint from the
-- immutability triggers; explicit writer-set values win).
CREATE FUNCTION raw_records_fill_state()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.type = 'state_object' AND NEW.payload IS NOT NULL THEN
        NEW.state_kind := COALESCE(NEW.state_kind, NEW.payload->>'kind');
        NEW.state_key  := COALESCE(NEW.state_key,  NEW.payload->>'key');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER raw_records_fill_state BEFORE INSERT ON raw_records
    FOR EACH ROW EXECUTE FUNCTION raw_records_fill_state();
