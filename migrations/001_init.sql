-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Entities table
CREATE TABLE IF NOT EXISTS entities (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    attributes  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS entities_name_idx ON entities (name);
CREATE INDEX IF NOT EXISTS entities_type_idx ON entities (entity_type);
CREATE INDEX IF NOT EXISTS entities_attrs_idx ON entities USING GIN (attributes);

-- Facts table
CREATE TABLE IF NOT EXISTS facts (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content     TEXT NOT NULL,
    source      TEXT,
    confidence  DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    entity_ids  UUID[] NOT NULL DEFAULT '{}',
    embedding   vector(1536),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_from  TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    metadata    JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS facts_entity_ids_idx ON facts USING GIN (entity_ids);
CREATE INDEX IF NOT EXISTS facts_valid_from_idx ON facts (valid_from);
CREATE INDEX IF NOT EXISTS facts_valid_until_idx ON facts (valid_until);
CREATE INDEX IF NOT EXISTS facts_metadata_idx ON facts USING GIN (metadata);

-- Relationships table
CREATE TABLE IF NOT EXISTS relationships (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_entity_id  UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_entity_id    UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type   TEXT NOT NULL,
    weight          DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    attributes      JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_from      TIMESTAMPTZ,
    valid_until     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS relationships_from_idx ON relationships (from_entity_id);
CREATE INDEX IF NOT EXISTS relationships_to_idx ON relationships (to_entity_id);
CREATE INDEX IF NOT EXISTS relationships_type_idx ON relationships (relation_type);

-- Core memory key-value store
CREATE TABLE IF NOT EXISTS core_memory (
    key         TEXT PRIMARY KEY,
    value       JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER entities_updated_at BEFORE UPDATE ON entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER facts_updated_at BEFORE UPDATE ON facts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER relationships_updated_at BEFORE UPDATE ON relationships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER core_memory_updated_at BEFORE UPDATE ON core_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
