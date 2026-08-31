-- ---------------------------------------------------------------------------
-- Curated layer (derived, rebuildable from raw; INSERT-only by the pipeline).
--   kind='episodic'|'semantic' at level 0; kind='summary' at level N+1
--   (RAPTOR tiers). Curation crosses topics, never users or modes.
-- ---------------------------------------------------------------------------
CREATE TABLE curated_nodes (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    kind        TEXT NOT NULL,
    content     TEXT NOT NULL,
    level       INT  NOT NULL DEFAULT 0,
    user_id     TEXT NOT NULL,
    mode        TEXT,
    importance  REAL,
    decay_class TEXT,
    event_time  TIMESTAMPTZ,
    meta        JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
-- No topic_id here: a curated node is derived ACROSS topics, so it has none of
-- its own. Topic scoping is answered from the raw evidence a node cites.
CREATE INDEX curated_nodes_scope_idx       ON curated_nodes (user_id, mode, kind);
CREATE INDEX curated_nodes_level_scope_idx ON curated_nodes (user_id, mode, level);

CREATE TABLE curated_embeddings (
    node_id        UUID NOT NULL REFERENCES curated_nodes(id) ON DELETE CASCADE,
    model          TEXT NOT NULL,
    embedding      vector(384),
    embedding_768  vector(768),
    embedding_1024 vector(1024),
    PRIMARY KEY (node_id, model)
);
CREATE INDEX curated_embeddings_vec_idx
    ON curated_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;
CREATE INDEX curated_embeddings_vec768_idx
    ON curated_embeddings USING hnsw (embedding_768 vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE embedding_768 IS NOT NULL;
CREATE INDEX curated_embeddings_vec1024_idx
    ON curated_embeddings USING hnsw (embedding_1024 vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE embedding_1024 IS NOT NULL;

-- Lineage edges. Kinds: 'derived_from' (node -> raw), 'supersedes'
-- (node -> node), 'summarizes' (summary -> child), 'entity' (node -> raw,
-- label = entity string).
CREATE TABLE curated_edges (
    from_id UUID NOT NULL,
    to_id   UUID NOT NULL,
    kind    TEXT NOT NULL,
    label   TEXT,
    PRIMARY KEY (from_id, to_id, kind)
);
CREATE INDEX curated_edges_to_kind_idx   ON curated_edges (to_id, kind);
CREATE INDEX curated_edges_from_kind_idx ON curated_edges (from_id, kind);
