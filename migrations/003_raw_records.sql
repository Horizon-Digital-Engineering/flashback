-- ---------------------------------------------------------------------------
-- RAW layer: the universal typed record. APPEND-ONLY; corrections are a new
-- row with `supersedes` = old id (forward pointer, old row untouched).
-- "Superseded" is derived: is this id referenced by a newer row's supersedes.
-- state_kind/state_key promote reference identity for type='state_object'
-- rows (filled from payload at INSERT); the terminal row of
-- (user_id, state_kind, state_key) is the current value.
--
-- RULES FOR THIS LAYER:
--   1. It ARRIVED — it wasn't computed. If Flashback produced it, it isn't raw.
--   2. It's immutable. Corrections are a new row pointing back via `supersedes`.
--   3. Nobody judges it on the way in — no tier, no importance, no expiry from
--      the writer. Those are conclusions, and conclusions are derived.
--   4. `type` is HOW IT MUST BE PROCESSED, not what it's about. A screenshot of
--      a spreadsheet is an image (needs OCR), not a document. Two records share
--      a type exactly when the same extractor handles them.
--   5. Everything records where it came from (`source`) and what it arrived
--      within (`container_id`).
--   6. Arrived metadata is stored VERBATIM in `payload` — never normalized.
--      Normalising is interpreting, and interpreting is deriving.
--
-- Corollary: a conclusion is NEVER written back onto a raw row. Derived facts
-- live in curated_* so a rebuild can reproduce them — and is free to disagree.
--
-- Tier vocabulary (episodic, semantic, summary) exists only in
-- curated_nodes.kind. Only types with a real processing rule are listed here;
-- an accepted-but-unread type is a silent data hole, which is exactly how a
-- writer's records once vanished on arrival.
-- ---------------------------------------------------------------------------
CREATE TABLE raw_records (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type          TEXT NOT NULL CHECK (type IN (
        'conversation','document','state_object'
    )),
    content       TEXT NOT NULL,
    content_hash  TEXT GENERATED ALWAYS AS (md5(content)) STORED,
    content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- dual time: when it actually happened vs when we stored it.
    event_time    TIMESTAMPTZ NOT NULL,
    ingest_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    source        TEXT NOT NULL,           -- origin: 'ritsu:health' | 'chatgpt' | ...
    source_ref    TEXT,                    -- stable external id (import dedup)

    user_id       TEXT NOT NULL,

    -- project_id is a HARD PARTITION: curation never derives across it, so two
    -- records in different projects can never be clustered or distilled
    -- together. Reserve it for genuinely separate bodies of work. A grouping
    -- that is really just a sorter — a ChatGPT "project", an agent name —
    -- belongs in `payload`, which filters without fragmenting.
    project_id    TEXT,

    -- The stream this arrived on: a chat thread, a watched folder, an import
    -- batch, a sensor feed. Episodes form per container, so a folder drop
    -- becomes an episode the same way a conversation does. Namespace it —
    -- it has to stay unique across every writer.
    container_id  TEXT,

    mode          TEXT,                    -- embedding register; also a hard partition
                                           -- (vectors of different dims can't compare)

    importance    REAL CHECK (importance IS NULL OR (importance >= 0.0 AND importance <= 1.0)),
    supersedes    UUID REFERENCES raw_records(id),

    state_kind    TEXT,                    -- state_object rows only
    state_key     TEXT,

    -- No `ttl` and no `acl` here on purpose. An append-only row cannot expire —
    -- a TTL would only hide truth from queries, and deciding how long something
    -- matters is a conclusion (rule 3), which is what the decay model is for. A
    -- source's OWN validity window (a calendar event's end, a cert's expiry) is
    -- an arrived fact and rides in `payload`. Visibility likewise belongs to a
    -- tenancy model that enforces it, not to a column nothing reads.

    -- The ONE metadata bag, holding what the source handed us verbatim: the
    -- exporting model, a conversation title, the folder a file was dropped in,
    -- a device's units. Uninterpreted on purpose — curation reads it, and a
    -- mapping mistake is rebuildable. Anything we CONCLUDED goes in curated_*.
    payload       JSONB
);

CREATE INDEX raw_records_scope_type_idx  ON raw_records (user_id, project_id, type);
CREATE INDEX raw_records_event_time_idx  ON raw_records (event_time);
CREATE INDEX raw_records_ingest_time_idx ON raw_records (ingest_time);
-- Episode formation walks a container's records in order, so the index covers
-- the grouping key and the sort.
CREATE INDEX raw_records_container_idx   ON raw_records (user_id, container_id, event_time)
    WHERE container_id IS NOT NULL;
CREATE INDEX raw_records_mode_idx        ON raw_records (mode) WHERE mode IS NOT NULL;
CREATE INDEX raw_records_hash_idx        ON raw_records (content_hash);
CREATE INDEX raw_records_supersedes_idx  ON raw_records (supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX raw_records_tsv_idx         ON raw_records USING GIN (content_tsv);

-- Arrived-metadata lookup ("everything dropped in the receipts folder").
-- jsonb_path_ops is the smaller/faster index and covers the containment
-- queries this is for (`payload @> '{"folder":"receipts_for_business"}'`).
CREATE INDEX raw_records_payload_idx     ON raw_records
    USING GIN (payload jsonb_path_ops) WHERE payload IS NOT NULL;

-- Idempotent bulk import: a record carrying a stable external id dedups on
-- (user_id, source, source_ref); normal ingest (no source_ref) is unconstrained.
CREATE UNIQUE INDEX raw_records_import_dedup_idx
    ON raw_records (user_id, source, source_ref)
    WHERE source_ref IS NOT NULL;

-- Reference-identity lookup for state objects.
CREATE INDEX raw_records_state_ident_idx
    ON raw_records (user_id, state_kind, state_key)
    WHERE type = 'state_object';

-- Append-only enforcement: UPDATE, DELETE, and TRUNCATE all raise.
CREATE FUNCTION raw_records_immutable()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'raw_records is append-only (attempted %)', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER raw_records_no_mutate BEFORE UPDATE OR DELETE ON raw_records
    FOR EACH ROW EXECUTE FUNCTION raw_records_immutable();
CREATE TRIGGER raw_records_no_truncate BEFORE TRUNCATE ON raw_records
    FOR EACH STATEMENT EXECUTE FUNCTION raw_records_immutable();

-- Fill state identity from the payload at INSERT only (disjoint from the
-- immutability triggers; explicit writer-set values win).
CREATE FUNCTION raw_records_fill_state()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.type = 'state_object' AND NEW.payload IS NOT NULL THEN
        NEW.state_kind := COALESCE(NEW.state_kind, NEW.payload->>'kind');
        NEW.state_key  := COALESCE(NEW.state_key,  NEW.payload->>'key');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER raw_records_fill_state BEFORE INSERT ON raw_records
    FOR EACH ROW EXECUTE FUNCTION raw_records_fill_state();
