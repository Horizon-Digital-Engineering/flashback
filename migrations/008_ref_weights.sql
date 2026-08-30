-- Decay/priority on the reference layer, never on raw: weight only ever
-- lowers ranking (demote-not-delete). ref_id is polymorphic (raw or curated
-- id); ref_kind disambiguates. R = exp(-t/S), S from s_hours or decay_class.
CREATE TABLE ref_weights (
    ref_id      UUID PRIMARY KEY,
    weight      REAL NOT NULL DEFAULT 1.0,
    last_access TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ref_kind    TEXT NOT NULL DEFAULT 'curated',
    decay_class TEXT NOT NULL DEFAULT 'default',
    s_hours     DOUBLE PRECISION
);
CREATE INDEX ref_weights_kind_idx ON ref_weights (ref_kind);
