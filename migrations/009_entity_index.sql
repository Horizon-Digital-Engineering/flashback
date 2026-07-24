-- The HippoRAG entity pointer table.
--
-- `entity_index` maps an extracted entity string to the raw records it appears
-- in, per user. It is a DERIVED, rebuildable index (like the embeddings and the
-- curated layer): the curation pipeline populates it from `extract_entities` on
-- the raw content, and it can be wiped + repopulated from raw at any time. It
-- exists so entity-overlap clustering reads a table instead of re-extracting
-- entities from scratch on every pass, and so future graph traversal can hop
-- entity → records without a full scan.
--
-- CASCADE on the record FK so it self-cleans if a raw row were ever removed
-- (raw is append-only today, but the FK keeps the index honest regardless).
CREATE TABLE IF NOT EXISTS entity_index (
    user_id   TEXT NOT NULL,
    entity    TEXT NOT NULL,
    record_id UUID NOT NULL REFERENCES raw_records(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, entity, record_id)
);

-- The forward hop: "which records mention this entity, for this user".
CREATE INDEX IF NOT EXISTS entity_index_user_entity_idx
    ON entity_index (user_id, entity);

-- ---------------------------------------------------------------------------
-- curated_edges gains an optional label so an 'entity' edge can carry the
-- entity string directly, rather than manufacturing a synthetic entity node in
-- curated_nodes for every distinct phrase (which would smear the node table).
--
-- New edge kind:
--   'entity' : curated_node.id (from) -> raw_record.id (to), label = entity
-- i.e. "this curated node is tied to this raw record via this entity". The
-- entity pointer table (above) is the scan-friendly index; the edge is the
-- glass-box lineage view ("what entities did this curated node cluster on").
-- Existing edges (derived_from / supersedes / summarizes) leave label NULL.
-- ---------------------------------------------------------------------------
ALTER TABLE curated_edges ADD COLUMN IF NOT EXISTS label TEXT;
