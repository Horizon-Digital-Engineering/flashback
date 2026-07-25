-- ---------------------------------------------------------------------------
-- Catalog / store registry: descriptions of stores, not their contents.
-- Built-in raw/curated layers auto-register; operational/external stores
-- publish fact slices that sync into raw via the idempotent import path.
-- ---------------------------------------------------------------------------
CREATE TABLE catalog_stores (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id        TEXT NOT NULL,
    name           TEXT NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN ('raw','curated','operational','external')),
    schema         JSONB,
    access         JSONB,                  -- {interface:'internal'|'http'|'sql', url?, auth_ref?}
    description    TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_synced_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX catalog_stores_user_name_idx ON catalog_stores (user_id, name);
CREATE INDEX catalog_stores_user_kind_idx ON catalog_stores (user_id, kind);

CREATE TABLE catalog_published_facts (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    store_id   UUID NOT NULL REFERENCES catalog_stores(id) ON DELETE CASCADE,
    fact       TEXT NOT NULL,
    event_time TIMESTAMPTZ,
    payload    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX catalog_published_facts_store_idx
    ON catalog_published_facts (store_id, created_at);
