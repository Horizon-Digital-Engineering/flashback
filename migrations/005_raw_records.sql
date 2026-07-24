-- The immutable RAW layer (medallion "bronze"): the universal typed record every
-- consumer writes. Distinct from `memories` (which conflates raw + curation):
-- raw_records is APPEND-ONLY and holds NO embeddings — embeddings + curated views
-- + references live in the derived tables below and are rebuildable from raw.
--
-- The one move nobody else makes: `supersedes` is a FORWARD pointer on the NEW
-- record; the old row is never mutated. "Superseded" is a derived status (is this
-- id referenced by any newer row's supersedes?). Immutability is enforced by a
-- trigger that blocks UPDATE/DELETE.

CREATE TABLE IF NOT EXISTS raw_records (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type          TEXT NOT NULL CHECK (type IN (
        'episodic','semantic','working','document','procedural','state_object'
    )),
    content       TEXT NOT NULL,
    -- md5 keeps integrity/dedup without pulling a hashing crate into the server.
    content_hash  TEXT GENERATED ALWAYS AS (md5(content)) STORED,
    content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- dual time: when it ACTUALLY happened vs when we stored it.
    event_time    TIMESTAMPTZ NOT NULL,
    ingest_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    source        TEXT NOT NULL,          -- origin: 'ritsu:health' | 'chatgpt' | 'finance-sync'
    source_ref    TEXT,                   -- pointer to the originating external record/file/url

    user_id       TEXT NOT NULL,
    project_id    TEXT,
    session_id    TEXT,
    mode          TEXT,

    importance    REAL CHECK (importance IS NULL OR (importance >= 0.0 AND importance <= 1.0)),
    supersedes    UUID REFERENCES raw_records(id),   -- forward pointer; old row untouched

    acl           JSONB,                  -- visibility/sharing; null = private (governance)
    ttl           TIMESTAMPTZ,            -- optional expiry (working memory)
    payload       JSONB                   -- type-specific structured fields
);

-- Promoted (indexed) columns — everything we sort/filter/scope by.
CREATE INDEX IF NOT EXISTS raw_records_scope_type_idx  ON raw_records (user_id, project_id, type);
CREATE INDEX IF NOT EXISTS raw_records_event_time_idx  ON raw_records (event_time);
CREATE INDEX IF NOT EXISTS raw_records_ingest_time_idx ON raw_records (ingest_time);
CREATE INDEX IF NOT EXISTS raw_records_session_idx     ON raw_records (session_id) WHERE session_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS raw_records_mode_idx        ON raw_records (mode) WHERE mode IS NOT NULL;
CREATE INDEX IF NOT EXISTS raw_records_hash_idx        ON raw_records (content_hash);
CREATE INDEX IF NOT EXISTS raw_records_supersedes_idx  ON raw_records (supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX IF NOT EXISTS raw_records_ttl_idx         ON raw_records (ttl) WHERE ttl IS NOT NULL;
CREATE INDEX IF NOT EXISTS raw_records_tsv_idx         ON raw_records USING GIN (content_tsv);

-- IMMUTABILITY: append-only. Corrections are a new row with supersedes = old.id.
CREATE OR REPLACE FUNCTION raw_records_immutable()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'raw_records is append-only (attempted % on %)', TG_OP, OLD.id;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS raw_records_no_mutate ON raw_records;
CREATE TRIGGER raw_records_no_mutate BEFORE UPDATE OR DELETE ON raw_records
    FOR EACH ROW EXECUTE FUNCTION raw_records_immutable();

-- ---------------------------------------------------------------------------
-- DERIVED tables (medallion "silver/gold"). Rebuildable from raw_records; a
-- consumer never writes these directly — the curation pipeline does. Named now
-- so the raw<->curated boundary (a seam) is fixed even before the pipeline lands.
-- ---------------------------------------------------------------------------

-- Embeddings live OUT of raw (raw is pre-embedding). One row per (record, model)
-- so re-embedding with a new model is additive.
CREATE TABLE IF NOT EXISTS raw_embeddings (
    record_id UUID NOT NULL REFERENCES raw_records(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (record_id, model)
);
CREATE INDEX IF NOT EXISTS raw_embeddings_vec_idx
    ON raw_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Curated summary nodes (RAPTOR-style): level 0 = raw, higher = summaries.
CREATE TABLE IF NOT EXISTS curated_nodes (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kind       TEXT NOT NULL,
    content    TEXT NOT NULL,
    level      INT  NOT NULL DEFAULT 0,
    user_id    TEXT NOT NULL,
    meta       JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- References (HippoRAG-style pointers): edges point at raw records / nodes by id.
-- Records never move; the reference layer is what's rebuilt.
CREATE TABLE IF NOT EXISTS curated_edges (
    from_id UUID NOT NULL,
    to_id   UUID NOT NULL,
    kind    TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id, kind)
);

-- Decay / priority weights live on the reference layer, never on raw
-- (MemoryBank's Ebbinghaus signal, inverted: demote from the active view, never delete raw).
CREATE TABLE IF NOT EXISTS ref_weights (
    ref_id      UUID PRIMARY KEY,
    weight      REAL NOT NULL DEFAULT 1.0,
    last_access TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
