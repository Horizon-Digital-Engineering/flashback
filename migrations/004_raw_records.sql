-- ---------------------------------------------------------------------------
-- RAW layer: the universal typed record. APPEND-ONLY; corrections are a new
-- row with `supersedes` = old id (forward pointer, old row untouched).
-- "Superseded" is derived: is this id referenced by a newer row's supersedes.
-- Reference identity for type='state_object' lives in payload ('kind'/'key');
-- the terminal row for a (kind, key) is the current value.
--
-- RULES FOR THIS LAYER:
--   1. It ARRIVED — it wasn't computed. If Flashback produced it, it isn't raw.
--   2. It's immutable. Corrections are a new row pointing back via `supersedes`.
--   3. Nobody judges it on the way in — no tier, no register, no importance, no
--      expiry. Those are conclusions, and conclusions live in derived_*.
--   4. `type` is HOW IT MUST BE PROCESSED, not what it's about. A screenshot of
--      a spreadsheet is an image (needs OCR), not a document. Two records share
--      a type exactly when the same extractor handles them.
--   5. Everything records where it came from (`source`) and what it arrived
--      within (`thread_id`).
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
    content_hash  TEXT,
    -- Original bytes, only when content had to be sanitised (NUL is unstorable).
    content_raw   BYTEA,
    content_tsv   tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- dual time: when it actually happened vs when we stored it.
    event_time    TIMESTAMPTZ NOT NULL,
    ingest_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    source        TEXT NOT NULL,           -- origin: 'ritsu:health' | 'chatgpt' | ...
    source_ref    TEXT,                    -- stable external id (import dedup)

    user_id       TEXT NOT NULL,

    -- Where the thread is filed: a sidebar folder. A filter, not a wall.
    topic_id      TEXT,

    -- The conversation this arrived on. Episodes form per thread. Namespace it.
    thread_id     TEXT,

    -- The writer's claim that this replaces an earlier record. A claim, so it
    -- stays; the server's own guess about which row a state key currently
    -- resolves to is a conclusion and lives in derived_superseded.
    supersedes    UUID REFERENCES raw_records(id),
    -- What the writer said this followed, in the WRITER's id space. We never
    -- resolve it here: a parent can arrive late, and a resolution that cannot be
    -- redone is a conclusion frozen into an immutable row. The resolved edge is
    -- derived and heals on rebuild.
    prev_source_ref TEXT,

    -- Arrival order per user. This is what the tamper chain runs along, NOT the
    -- causal link: order of arrival is dense and total, so every record protects
    -- every later one. Chaining on the causal link instead left an imported
    -- corpus as thousands of one-record chains protecting nothing.
    seq           BIGINT NOT NULL,
    record_hash   TEXT,

    -- No `ttl` and no `acl` here on purpose. An append-only row cannot expire —
    -- a TTL would only hide truth from queries, and deciding how long something
    -- matters is a conclusion (rule 3), which is what the decay model is for. A
    -- source's OWN validity window (a calendar event's end, a cert's expiry) is
    -- an arrived fact and rides in `payload`. Visibility lives in `grants`.

    -- The ONE metadata bag, holding what the source handed us verbatim: the
    -- exporting model, a conversation title, the folder a file was dropped in,
    -- a device's units. Uninterpreted on purpose — curation reads it, and a
    -- mapping mistake is rebuildable. Anything we CONCLUDED goes in curated_*.
    payload       JSONB
);


CREATE INDEX raw_records_event_time_idx  ON raw_records (event_time);
CREATE INDEX raw_records_ingest_time_idx ON raw_records (ingest_time);
-- Episode formation walks a container's records in order, so the index covers
-- the grouping key and the sort.
CREATE INDEX raw_records_thread_idx      ON raw_records (user_id, thread_id, event_time)
    WHERE thread_id IS NOT NULL;
CREATE INDEX raw_records_user_time_idx   ON raw_records (user_id, event_time DESC);
CREATE INDEX raw_records_type_time_idx   ON raw_records (user_id, type, event_time DESC);
CREATE INDEX raw_records_topic_time_idx  ON raw_records (user_id, topic_id, event_time DESC)
    WHERE topic_id IS NOT NULL;

CREATE INDEX raw_records_hash_idx        ON raw_records (content_hash);
CREATE INDEX raw_records_supersedes_idx  ON raw_records (supersedes) WHERE supersedes IS NOT NULL;
CREATE UNIQUE INDEX raw_records_seq_idx  ON raw_records (user_id, seq);
CREATE INDEX raw_records_source_ref_idx  ON raw_records (user_id, source_ref)
    WHERE source_ref IS NOT NULL;
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
-- Reference identity comes straight out of the payload. Copying it into columns
-- needed a BEFORE INSERT trigger, and that trigger fired after the hash one, so
-- every state_object failed its own tamper check.
CREATE INDEX raw_records_state_ident_idx
    ON raw_records (user_id, (payload->>'kind'), (payload->>'key'))
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

-- Length-prefixed so no field can impersonate part of another: "12:" cannot be
-- read as content. NULL encodes as "-:" and the empty string as "0:", so the
-- two stay distinguishable.
CREATE FUNCTION raw_records_hash_field(v TEXT)
RETURNS TEXT AS $$
    SELECT CASE WHEN v IS NULL THEN '-:' ELSE length(v)::text || ':' || v END;
$$ LANGUAGE sql IMMUTABLE;

-- The preimage lives in ONE place so the writer and the verifier cannot drift.
-- Takes the whole row, so adding a column without deciding whether it is covered
-- is a visible edit here rather than a silent omission.
-- Takes the row as jsonb so the same definition serves every schema that clones
-- this table. `content_hash` and `record_hash` are excluded on purpose: both are
-- computed by the trigger, so including them would make the preimage circular.
CREATE FUNCTION raw_records_preimage(prev_hash TEXT, r JSONB)
RETURNS TEXT AS $$
    SELECT raw_records_hash_field(COALESCE(prev_hash, ''))
        || raw_records_hash_field(r->>'id')
        || raw_records_hash_field(r->>'type')
        || raw_records_hash_field(r->>'content')
        || raw_records_hash_field(r->>'content_raw')
        || raw_records_hash_field(r->>'event_time')
        || raw_records_hash_field(r->>'ingest_time')
        || raw_records_hash_field(r->>'source')
        || raw_records_hash_field(r->>'source_ref')
        || raw_records_hash_field(r->>'user_id')
        || raw_records_hash_field(r->>'topic_id')
        || raw_records_hash_field(r->>'thread_id')
        || raw_records_hash_field(r->>'supersedes')
        || raw_records_hash_field(r->>'prev_source_ref')
        || raw_records_hash_field(r->>'seq')
        || raw_records_hash_field(r->>'payload');
$$ LANGUAGE sql IMMUTABLE;

CREATE FUNCTION raw_records_fill_hash()
RETURNS TRIGGER AS $$
DECLARE
    prev_hash TEXT := '';
BEGIN
    -- One writer at a time per user, so seq is dense and gapless. The lock is
    -- held to end of transaction, so a bulk import takes it once, not per row.
    PERFORM pg_advisory_xact_lock(hashtext(NEW.user_id));
    EXECUTE format(
        'SELECT COALESCE(MAX(seq), 0) + 1 FROM %I.raw_records WHERE user_id = $1',
        TG_TABLE_SCHEMA)
        INTO NEW.seq USING NEW.user_id;

    -- Resolve the predecessor in THIS table's schema, never through search_path:
    -- a sandbox row must not chain to a real one, and the hash has to be a
    -- function of the data rather than of the connection that wrote it.
    IF NEW.seq > 1 THEN
        EXECUTE format(
            'SELECT record_hash FROM %I.raw_records WHERE user_id = $1 AND seq = $2',
            TG_TABLE_SCHEMA)
            INTO prev_hash USING NEW.user_id, NEW.seq - 1;
        IF prev_hash IS NULL THEN
            RAISE EXCEPTION 'no record_hash at seq % for user %', NEW.seq - 1, NEW.user_id;
        END IF;
    END IF;

    NEW.content_hash := encode(sha256(convert_to(NEW.content, 'UTF8')), 'hex');
    NEW.record_hash := encode(
        sha256(convert_to(raw_records_preimage(prev_hash, to_jsonb(NEW)), 'UTF8')), 'hex');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER raw_records_fill_hash BEFORE INSERT ON raw_records
    FOR EACH ROW EXECUTE FUNCTION raw_records_fill_hash();

-- Verification walks arrival order forward from seq 1, recomputing as it goes,
-- so a tampered record invalidates every record that arrived after it. Checking
-- a row against its parent's STORED hash would miss exactly that.
CREATE FUNCTION raw_records_verify_chain(p_user_id TEXT)
RETURNS TABLE (record_id UUID, seq BIGINT, stored_hash TEXT, computed_hash TEXT, ok BOOLEAN) AS $$
    WITH RECURSIVE chain AS (
        SELECT r.id, r.seq, r.record_hash,
               encode(sha256(convert_to(raw_records_preimage('', to_jsonb(r)), 'UTF8')), 'hex') AS computed
        FROM raw_records r
        WHERE r.user_id = p_user_id AND r.seq = 1
        UNION ALL
        SELECT r.id, r.seq, r.record_hash,
               encode(sha256(convert_to(raw_records_preimage(c.computed, to_jsonb(r)), 'UTF8')), 'hex')
        FROM raw_records r
        JOIN chain c ON r.seq = c.seq + 1
        WHERE r.user_id = p_user_id
    )
    SELECT id, seq, record_hash, computed, record_hash IS NOT DISTINCT FROM computed
    FROM chain
    ORDER BY seq;
$$ LANGUAGE sql STABLE;
