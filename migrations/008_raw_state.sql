-- References as a first-class citizen of the RAW layer.
--
-- 005 already carries `type='state_object'` and the `{kind, key, data}` payload
-- convention. This migration PROMOTES the reference identity onto raw_records as
-- two indexed, write-time columns so the current-value read is O(1) instead of a
-- payload scan:
--
--   state_kind  the reference family  ('todo_list' | 'plan' | 'decision_log' …)
--   state_key   the name within that family within a user's scope
--
-- A reference's identity is the triple (user_id, state_kind, state_key). Its
-- CURRENT VALUE is the terminal node: the `state_object` raw row for that
-- identity not referenced by any newer row's `supersedes`. Older rows are the
-- audit trail. `payload->'data'` on the terminal row is always the COMPLETE
-- current value (never a delta), so rendering the current state needs no chain
-- walk.
--
-- These columns are:
--   * NULLABLE — only populated for `type='state_object'` rows.
--   * NON-MUTATING — set once at INSERT from the payload, NEVER updated. The
--     005 immutability trigger (blocks UPDATE/DELETE) is left exactly as-is;
--     the BEFORE INSERT trigger below only fills NEW on the way in.

ALTER TABLE raw_records ADD COLUMN IF NOT EXISTS state_kind TEXT;
ALTER TABLE raw_records ADD COLUMN IF NOT EXISTS state_key  TEXT;

-- Populate the two columns from the payload at INSERT time only. This rides in
-- front of the existing immutability trigger (which fires on UPDATE/DELETE, a
-- disjoint event) so there is no interaction with append-only enforcement. A
-- writer that already set the columns explicitly is respected (COALESCE).
CREATE OR REPLACE FUNCTION raw_records_fill_state()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.type = 'state_object' AND NEW.payload IS NOT NULL THEN
        NEW.state_kind := COALESCE(NEW.state_kind, NEW.payload->>'kind');
        NEW.state_key  := COALESCE(NEW.state_key,  NEW.payload->>'key');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS raw_records_fill_state ON raw_records;
CREATE TRIGGER raw_records_fill_state BEFORE INSERT ON raw_records
    FOR EACH ROW EXECUTE FUNCTION raw_records_fill_state();

-- Backfill any pre-existing state_object rows (idempotent; touches only NULLs).
-- A direct UPDATE would trip the immutability trigger, so the backfill is done
-- with the trigger momentarily disabled — this is a one-time migration-time
-- write of columns that were absent when those rows were inserted, not a
-- mutation of any existing value.
ALTER TABLE raw_records DISABLE TRIGGER raw_records_no_mutate;
UPDATE raw_records
   SET state_kind = payload->>'kind',
       state_key  = payload->>'key'
 WHERE type = 'state_object'
   AND payload IS NOT NULL
   AND (state_kind IS NULL OR state_key IS NULL);
ALTER TABLE raw_records ENABLE TRIGGER raw_records_no_mutate;

-- The reference-identity lookup: "the rows for (user, state_kind, state_key)".
-- Partial — only state_object rows carry these columns.
CREATE INDEX IF NOT EXISTS raw_records_state_ident_idx
    ON raw_records (user_id, state_kind, state_key)
    WHERE type = 'state_object';
