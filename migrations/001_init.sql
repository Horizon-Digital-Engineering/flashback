-- Flashback initial schema.
--
-- One table holds every kind of memory (records + references) with a `type`
-- discriminator. The state_object branch carries extra columns for the
-- structured-reference case.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS memories (
    id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type             TEXT NOT NULL CHECK (type IN (
        'episodic','semantic','working','document','procedural','state_object'
    )),
    content          TEXT NOT NULL,
    content_tsv      tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding        vector(384),

    importance       REAL NOT NULL DEFAULT 0.5 CHECK (importance >= 0.0 AND importance <= 1.0),
    access_count     INT  NOT NULL DEFAULT 0,
    decay_class      TEXT NOT NULL DEFAULT 'medium' CHECK (decay_class IN ('none','slow','medium','fast')),

    user_id          TEXT NOT NULL,
    project_id       TEXT,
    session_id       TEXT,
    entities         TEXT[] NOT NULL DEFAULT '{}',

    -- Supersede chain: a memory points back to what it replaced and forward to
    -- whatever replaced it. Both directions are indexed.
    superseded_by    UUID REFERENCES memories(id),
    supersedes       UUID REFERENCES memories(id),

    -- Document branch
    source_path      TEXT,
    chunk_index      INT,
    content_hash     TEXT,

    -- state_object branch — the reference half of memory
    state_kind       TEXT,
    state_key        TEXT,
    state_data       JSONB,

    expires_at       TIMESTAMPTZ,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT state_object_columns_consistent CHECK (
        (type = 'state_object' AND state_kind IS NOT NULL AND state_key IS NOT NULL AND state_data IS NOT NULL)
        OR (type <> 'state_object')
    ),
    CONSTRAINT document_columns_consistent CHECK (
        (type = 'document' AND source_path IS NOT NULL)
        OR (type <> 'document')
    )
);

CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS memories_tsv_idx
    ON memories USING GIN (content_tsv);

CREATE INDEX IF NOT EXISTS memories_user_proj_type_idx
    ON memories (user_id, project_id, type);

CREATE INDEX IF NOT EXISTS memories_session_idx
    ON memories (session_id) WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS memories_expires_idx
    ON memories (expires_at) WHERE expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS memories_supersede_idx
    ON memories (superseded_by) WHERE superseded_by IS NOT NULL;

CREATE INDEX IF NOT EXISTS memories_entities_idx
    ON memories USING GIN (entities);

CREATE INDEX IF NOT EXISTS memories_state_key_idx
    ON memories (user_id, state_kind, state_key) WHERE state_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS memories_terminal_idx
    ON memories (user_id, type, last_accessed_at DESC) WHERE superseded_by IS NULL;

-- Core memory: pinned, always-injected context. Small, hand-curated.
CREATE TABLE IF NOT EXISTS core_memory (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    importance  REAL NOT NULL DEFAULT 1.0 CHECK (importance >= 0.0 AND importance <= 1.0),
    pinned_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS core_memory_user_idx ON core_memory (user_id);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS memories_updated_at ON memories;
CREATE TRIGGER memories_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS core_memory_updated_at ON core_memory;
CREATE TRIGGER core_memory_updated_at BEFORE UPDATE ON core_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
