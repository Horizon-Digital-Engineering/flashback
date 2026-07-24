-- Hierarchical summaries + decay/importance weighting over the curated layer:
-- RAPTOR-style summary tiers and the decay signal the active view ranks by.
-- Both are derived and rebuildable — raw stays immutable, and the whole summary
-- tree + weight table can be wiped and re-derived from the level-0 curated
-- nodes at any time.
--
-- 008 promoted `curated_nodes`/`curated_edges` to the load-bearing level-0
-- layer (episodic/semantic, all level 0). This migration adds the levels above
-- them and the reference-weight signal that demotes stale nodes from the active
-- feed without ever deleting anything.

-- ---------------------------------------------------------------------------
-- Summary levels. `curated_nodes.level` already exists (INT DEFAULT 0); this
-- just documents the new usage and indexes the traversal:
--   kind='summary', level = child_level + 1 : a RAPTOR summary of a cluster of
--   lower-level nodes. Recurses until a level has few enough nodes or hits a
--   depth cap. Scope columns (user/project/mode) are inherited from children
--   and never crossed.
--
-- New edge kind:
--   'summarizes' : summary_node.id (from) -> child_node.id (to)
-- The child lookup ("what does this summary cover?") reads from_id+kind; the
-- parent lookup ("which summary owns this node?") reads to_id+kind. 008 already
-- created both curated_edges (from_id,kind) and (to_id,kind) indexes, which
-- cover 'summarizes' too.
-- ---------------------------------------------------------------------------

-- Level-scoped scan: "the top-level summaries for this user/project/mode".
CREATE INDEX IF NOT EXISTS curated_nodes_level_scope_idx
    ON curated_nodes (user_id, project_id, mode, level);

-- ---------------------------------------------------------------------------
-- ref_weights — the Ebbinghaus decay signal on the reference layer (never on
-- raw). 005 scaffolded (ref_id, weight, last_access); this promotes it:
--
--   ref_kind    'raw' | 'curated' — `ref_id` is polymorphic: a raw_records.id
--               OR a curated_nodes.id. The kind disambiguates which table it
--               points at (there is no cross-table FK; the two id spaces are
--               distinct UUIDs but the column carries either).
--   decay_class the named forgetting curve this ref follows (e.g. 'default',
--               'pinned', 'ephemeral'); drives the half-life when s_hours is
--               unset.
--   s_hours     the stored half-life S (hours) for the retention curve
--               R = exp(-t / S), t = hours since last_access. Explicit per-ref
--               override; falls back to a decay_class default when NULL.
--
-- Nothing here is ever deleted: decay only lowers `weight`, which lowers
-- ranking. A ref that drops to a low weight is demoted from the active feed,
-- never removed (tombstone-not-delete).
-- ---------------------------------------------------------------------------
ALTER TABLE ref_weights ADD COLUMN IF NOT EXISTS ref_kind    TEXT NOT NULL DEFAULT 'curated';
ALTER TABLE ref_weights ADD COLUMN IF NOT EXISTS decay_class TEXT NOT NULL DEFAULT 'default';
ALTER TABLE ref_weights ADD COLUMN IF NOT EXISTS s_hours     DOUBLE PRECISION;

-- Look up a weight by (ref_id) is already the PK; index the kind so a scoped
-- sweep ("all curated weights") stays cheap.
CREATE INDEX IF NOT EXISTS ref_weights_kind_idx ON ref_weights (ref_kind);
