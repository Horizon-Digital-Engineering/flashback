-- Structured extraction output stored alongside each memory.
--
-- Populated at ingest time by the configured `AiProvider`. Heuristic provider
-- always fills `entities` and best-effort `intent`/`operation`. LLM providers
-- (remote or local) fill `topic`, `action_target`, `contradicts_hint` with
-- richer semantic content.
--
-- Phase 2d will use `extraction->>'topic'` for semantic-cosine supersede
-- detection. Phase 1 / 2a row-level supersede stays on the `entities` column
-- so legacy and back-compat work.

ALTER TABLE memories
    ADD COLUMN IF NOT EXISTS extraction JSONB;

-- Indexed lookup on intent / topic for future filtering endpoints.
CREATE INDEX IF NOT EXISTS memories_extraction_intent_idx
    ON memories ((extraction ->> 'intent'))
    WHERE extraction IS NOT NULL;

CREATE INDEX IF NOT EXISTS memories_extraction_topic_idx
    ON memories ((extraction ->> 'topic'))
    WHERE extraction IS NOT NULL;
