-- Entity pointer table (derived, rebuildable): entity string -> raw records,
-- per user. Read by entity-overlap clustering and graph hops.
CREATE TABLE entity_index (
    user_id   TEXT NOT NULL,
    entity    TEXT NOT NULL,
    record_id UUID NOT NULL REFERENCES raw_records(id) ON DELETE CASCADE,
    PRIMARY KEY (user_id, entity, record_id)
);
CREATE INDEX entity_index_user_entity_idx ON entity_index (user_id, entity);
