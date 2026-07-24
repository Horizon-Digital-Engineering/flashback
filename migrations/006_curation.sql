-- The CURATED layer built on `raw_records` (medallion "silver/gold").
-- Derived + rebuildable: raw is immutable and never mutated; curation only ever
-- INSERTs here and can be wiped + rebuilt from raw at any time.
--
-- 003 already scaffolded bare `curated_nodes`/`curated_edges`; this migration
-- promotes them to load-bearing: scope + lineage columns, an embeddings table
-- mirroring `raw_embeddings`, and the indexes the pipeline reads by.

-- ---------------------------------------------------------------------------
-- curated_nodes — the derived summary/fact rows.
--   kind='episodic'  (level 0): one per active working raw_record.
--   kind='semantic'  (level 0): a fact distilled from an entity-overlap cluster.
-- Scope columns mirror raw_records so curation never crosses user/project/mode.
-- ---------------------------------------------------------------------------
ALTER TABLE curated_nodes ADD COLUMN IF NOT EXISTS project_id  TEXT;
ALTER TABLE curated_nodes ADD COLUMN IF NOT EXISTS mode        TEXT;
ALTER TABLE curated_nodes ADD COLUMN IF NOT EXISTS importance  REAL;
ALTER TABLE curated_nodes ADD COLUMN IF NOT EXISTS decay_class TEXT;
ALTER TABLE curated_nodes ADD COLUMN IF NOT EXISTS event_time  TIMESTAMPTZ;

-- Scope + lineage lookups: "curated nodes for this user/project/mode of a kind".
CREATE INDEX IF NOT EXISTS curated_nodes_scope_idx
    ON curated_nodes (user_id, project_id, mode, kind);

-- ---------------------------------------------------------------------------
-- curated_embeddings — one row per (node, model), mirroring raw_embeddings.
-- Re-embedding with a new model is additive. CASCADE so a rebuild that drops a
-- node cleans up its vectors.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS curated_embeddings (
    node_id   UUID NOT NULL REFERENCES curated_nodes(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    PRIMARY KEY (node_id, model)
);
CREATE INDEX IF NOT EXISTS curated_embeddings_vec_idx
    ON curated_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- curated_edges lineage indexes. Edge kinds used by the pipeline:
--   'derived_from' : curated_node.id (from) -> raw_record.id (to)
--   'supersedes'   : curated_node.id (from) -> curated_node.id (to)
-- The reverse lookup ("what derived from this raw id?") is the glass-box door,
-- so index to_id + kind.
-- ---------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS curated_edges_to_kind_idx  ON curated_edges (to_id, kind);
CREATE INDEX IF NOT EXISTS curated_edges_from_kind_idx ON curated_edges (from_id, kind);
