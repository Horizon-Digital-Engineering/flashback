-- ---------------------------------------------------------------------------
-- Embeddings (derived; raw is pre-embedding). One row per (record, model);
-- per-mode geometry: exactly one of the three dim columns is populated,
-- matching the mode's embedder. Partial indexes so empty columns cost nothing.
-- ---------------------------------------------------------------------------
CREATE TABLE raw_embeddings (
    record_id      UUID NOT NULL REFERENCES raw_records(id) ON DELETE CASCADE,
    model          TEXT NOT NULL,
    embedding      vector(384),
    embedding_768  vector(768),
    embedding_1024 vector(1024),
    PRIMARY KEY (record_id, model)
);
CREATE INDEX raw_embeddings_vec_idx
    ON raw_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    WHERE embedding IS NOT NULL;
CREATE INDEX raw_embeddings_vec768_idx
    ON raw_embeddings USING ivfflat (embedding_768 vector_cosine_ops) WITH (lists = 100)
    WHERE embedding_768 IS NOT NULL;
CREATE INDEX raw_embeddings_vec1024_idx
    ON raw_embeddings USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists = 100)
    WHERE embedding_1024 IS NOT NULL;
