CREATE SCHEMA IF NOT EXISTS playground;

CREATE TABLE playground.raw_records (LIKE public.raw_records INCLUDING ALL);
ALTER TABLE playground.raw_records
    ADD CONSTRAINT raw_records_supersedes_fkey
        FOREIGN KEY (supersedes) REFERENCES playground.raw_records(id);

-- LIKE does not copy triggers, so all four are re-attached by hand. The sandbox
-- must have the SAME storage semantics as production or it stops being a
-- rehearsal: an app-level UPDATE that passes here would fail in public.
CREATE TRIGGER raw_records_fill_hash BEFORE INSERT ON playground.raw_records
    FOR EACH ROW EXECUTE FUNCTION public.raw_records_fill_hash();
CREATE TRIGGER raw_records_no_mutate BEFORE UPDATE OR DELETE ON playground.raw_records
    FOR EACH ROW EXECUTE FUNCTION public.raw_records_immutable();
CREATE TRIGGER raw_records_no_truncate BEFORE TRUNCATE ON playground.raw_records
    FOR EACH STATEMENT EXECUTE FUNCTION public.raw_records_immutable();

CREATE TABLE playground.raw_embeddings (LIKE public.raw_embeddings INCLUDING ALL);
ALTER TABLE playground.raw_embeddings
    ADD CONSTRAINT raw_embeddings_record_fkey
        FOREIGN KEY (record_id) REFERENCES playground.raw_records(id) ON DELETE CASCADE;

CREATE TABLE playground.curated_nodes (LIKE public.curated_nodes INCLUDING ALL);
CREATE TABLE playground.curated_edges (LIKE public.curated_edges INCLUDING ALL);
CREATE TABLE playground.curated_embeddings (LIKE public.curated_embeddings INCLUDING ALL);
ALTER TABLE playground.curated_embeddings
    ADD CONSTRAINT curated_embeddings_node_fkey
        FOREIGN KEY (node_id) REFERENCES playground.curated_nodes(id) ON DELETE CASCADE;

CREATE TABLE playground.entity_index (LIKE public.entity_index INCLUDING ALL);
ALTER TABLE playground.entity_index
    ADD CONSTRAINT entity_index_record_fkey
        FOREIGN KEY (record_id) REFERENCES playground.raw_records(id) ON DELETE CASCADE;

-- Derived state the sandbox writes through an unqualified name. Without a twin
-- here, `search_path` resolves these to public and playing in the sandbox
-- reorders real memory.
CREATE TABLE playground.ref_weights (LIKE public.ref_weights INCLUDING ALL);
CREATE TABLE playground.modes (LIKE public.modes INCLUDING ALL);
-- LIKE clones the table, not its rows. Without the template registers the
-- sandbox has none at all, every resolve fails, and the one surface built to
-- rehearse mode-scoped behaviour cannot exercise it.
INSERT INTO playground.modes SELECT * FROM public.modes;

CREATE TABLE playground.derived_record_mode (LIKE public.derived_record_mode INCLUDING ALL);
ALTER TABLE playground.derived_record_mode
    ADD CONSTRAINT derived_record_mode_record_fkey
        FOREIGN KEY (record_id) REFERENCES playground.raw_records(id) ON DELETE CASCADE,
    ADD CONSTRAINT derived_record_mode_mode_fkey
        FOREIGN KEY (user_id, mode) REFERENCES playground.modes (user_id, name);

CREATE TABLE playground.derived_superseded (LIKE public.derived_superseded INCLUDING ALL);
ALTER TABLE playground.derived_superseded
    ADD CONSTRAINT derived_superseded_record_fkey
        FOREIGN KEY (record_id) REFERENCES playground.raw_records(id) ON DELETE CASCADE;

CREATE TABLE playground.derived_link (LIKE public.derived_link INCLUDING ALL);
ALTER TABLE playground.derived_link
    ADD CONSTRAINT derived_link_record_fkey
        FOREIGN KEY (record_id) REFERENCES playground.raw_records(id) ON DELETE CASCADE;
CREATE TABLE playground.derived_labels (LIKE public.derived_labels INCLUDING ALL);
