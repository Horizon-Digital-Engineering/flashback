-- Close the append-only guarantee fully: the BEFORE UPDATE/DELETE trigger does
-- NOT fire on TRUNCATE. Make the guard TG_OP-safe (OLD is unset for TRUNCATE —
-- so drop the OLD.id reference) and add a statement-level TRUNCATE guard.
CREATE OR REPLACE FUNCTION raw_records_immutable()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'raw_records is append-only (attempted %)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS raw_records_no_truncate ON raw_records;
CREATE TRIGGER raw_records_no_truncate BEFORE TRUNCATE ON raw_records
    FOR EACH STATEMENT EXECUTE FUNCTION raw_records_immutable();
