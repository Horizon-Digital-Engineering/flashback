-- The CATALOG / store-registry — the Operation layer's legibility surface.
--
-- The lake's answer to "is my data organized, and can I SEE it?". Every store
-- the lake knows about is a row here: the two built-in layers (raw + curated)
-- auto-register with a live schema + record-count summary, and a user can
-- register operational/external stores that publish slices INTO the lake.
--
-- This is a REGISTRY, not a data table — it holds descriptions of stores, not
-- their contents. The raw/curated summaries are computed live from the actual
-- tables at read time; a registered operational store carries a declared schema
-- and an access descriptor for how the lake reaches it.

CREATE TABLE IF NOT EXISTS catalog_stores (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id        TEXT NOT NULL,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('raw','curated','operational','external')),
    -- Declared column/field shape of the store (free-form JSON schema-ish blob).
    schema         JSONB,
    -- How the lake reaches the store's data:
    --   {interface:'internal'|'http'|'sql', url?, auth_ref?}
    -- 'internal' = a built-in lake table (raw/curated). 'http'/'sql' = an
    -- external system the lake syncs from (live query-through is a later step).
    access         JSONB,
    description    TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Last time this store's published facts were pulled into the lake.
    last_synced_at TIMESTAMPTZ
);

-- One store name per user (the raw/curated built-ins reserve their names).
CREATE UNIQUE INDEX IF NOT EXISTS catalog_stores_user_name_idx
    ON catalog_stores (user_id, name);

-- List/group by kind for the store map.
CREATE INDEX IF NOT EXISTS catalog_stores_user_kind_idx
    ON catalog_stores (user_id, kind);

-- ---------------------------------------------------------------------------
-- catalog_published_facts — the SLICES an operational/external store publishes
-- into the lake. A store declares "here are the facts you may ingest"; a sync
-- pulls them into `raw_records` through the normal idempotent import path. This
-- is the concrete "the lake ingests slices" mechanism (as opposed to live
-- query-through, which is a later step).
--
-- CASCADE on the store FK so deleting a store cleans up its staged slices.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS catalog_published_facts (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    store_id   UUID NOT NULL REFERENCES catalog_stores(id) ON DELETE CASCADE,
    fact       TEXT NOT NULL,
    event_time TIMESTAMPTZ,
    payload    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The sync read: "the facts this store has published", oldest first.
CREATE INDEX IF NOT EXISTS catalog_published_facts_store_idx
    ON catalog_published_facts (store_id, created_at);
