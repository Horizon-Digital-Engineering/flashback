-- ---------------------------------------------------------------------------
-- What things are called and what they are about. Two tables on purpose: one
-- holds what a person SAID, the other what the system GUESSED, and the boundary
-- between them is a table boundary rather than a column. A rebuild wipes the
-- guesses; it can never touch the statements.
--
-- This replaces a `titles` table and a `classifications` table that were the
-- same shape written twice. A title is a facet, a category is a facet, a
-- sensitivity level is a facet.
-- ---------------------------------------------------------------------------

-- ARRIVED. A person said something about a subject. Append-only, same rule as
-- raw_records: changing your mind is a new line, never an edit.
CREATE TABLE raw_assertions (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      TEXT NOT NULL,
    subject_type TEXT NOT NULL CHECK (subject_type IN ('topic','thread','record')),
    subject_id   TEXT NOT NULL,
    facet        TEXT NOT NULL,
    label        TEXT NOT NULL,
    stance       TEXT NOT NULL CHECK (stance IN ('affirm','retract')),
    event_time   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingest_time  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source       TEXT NOT NULL,
    asserted_by  TEXT NOT NULL,
    -- e.g. {"responding_to":"tagger-v1"} when this answers a suggestion.
    basis        JSONB
);
CREATE INDEX raw_assertions_subject_idx
    ON raw_assertions (user_id, subject_type, subject_id, facet, label, event_time DESC);
CREATE INDEX raw_assertions_label_idx
    ON raw_assertions (user_id, facet, label) WHERE stance = 'affirm';

CREATE TRIGGER raw_assertions_no_mutate BEFORE UPDATE OR DELETE ON raw_assertions
    FOR EACH ROW EXECUTE FUNCTION raw_records_immutable();
CREATE TRIGGER raw_assertions_no_truncate BEFORE TRUNCATE ON raw_assertions
    FOR EACH STATEMENT EXECUTE FUNCTION raw_records_immutable();

-- Whether a facet holds one value or many decides how "current" is read.
CREATE TABLE label_facets (
    facet       TEXT PRIMARY KEY,
    cardinality TEXT NOT NULL CHECK (cardinality IN ('scalar','set'))
);
INSERT INTO label_facets (facet, cardinality) VALUES
    ('title', 'scalar'), ('sensitivity', 'scalar'), ('category', 'set');

-- CONCLUDED. What a tagger thinks. Wiped wholesale by a rebuild.
CREATE TABLE derived_labels (
    user_id      TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id   TEXT NOT NULL,
    facet        TEXT NOT NULL,
    label        TEXT NOT NULL,
    confidence   REAL NOT NULL,
    derived_by   TEXT NOT NULL,
    derived_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, subject_type, subject_id, facet, label)
);
CREATE INDEX derived_labels_lookup_idx ON derived_labels (user_id, facet, label);

-- Set-valued facets. A human affirm stands until a LATER retract names it. A
-- guess is suppressed by ANY human statement on the same label — which is what
-- makes a rejection survive a rebuild that keeps re-suggesting the same thing.
CREATE VIEW current_labels AS
SELECT a.user_id, a.subject_type, a.subject_id, a.facet, a.label,
       'human'::text AS authority, max(a.event_time) AS since
FROM raw_assertions a
WHERE a.stance = 'affirm'
  AND NOT EXISTS (
      SELECT 1 FROM raw_assertions r
      WHERE (r.user_id, r.subject_type, r.subject_id, r.facet, r.label)
          = (a.user_id, a.subject_type, a.subject_id, a.facet, a.label)
        AND r.stance = 'retract' AND r.event_time > a.event_time)
GROUP BY 1,2,3,4,5
UNION ALL
SELECT d.user_id, d.subject_type, d.subject_id, d.facet, d.label,
       'derived', d.derived_at
FROM derived_labels d
WHERE NOT EXISTS (
    SELECT 1 FROM raw_assertions r
    WHERE (r.user_id, r.subject_type, r.subject_id, r.facet, r.label)
        = (d.user_id, d.subject_type, d.subject_id, d.facet, d.label));

-- Scalar facets: latest human statement wins. Human-only by construction, so no
-- ordering accident can let a guess downgrade a sensitivity level someone set.
CREATE VIEW current_scalar_labels AS
SELECT DISTINCT ON (user_id, subject_type, subject_id, facet)
       user_id, subject_type, subject_id, facet, label, event_time
FROM raw_assertions
WHERE stance = 'affirm'
  AND facet IN (SELECT facet FROM label_facets WHERE cardinality = 'scalar')
ORDER BY user_id, subject_type, subject_id, facet, event_time DESC;
