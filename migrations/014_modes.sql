-- Modes (cognitive registers) — a first-class axis of the memory model.
-- Every raw record and curated node already carries a `mode` column (a filter
-- passthrough until now); this migration makes modes load-bearing: each mode
-- pins an embedder + vector dimension, so a record is embedded in its mode's
-- geometry and only ever compared against records in the same geometry.
--
-- A mode is scoped to a user. Four built-in registers ship out of the box
-- (general/code/journal/research), each with the embedder from docs/EMBEDDINGS.md.
-- The canonical built-in definitions are seeded here under the reserved
-- `user_id = '*'` template row; the server copies them per-user on first use
-- (mirroring how the catalog auto-registers its built-in stores) and users can
-- declare their own registers on top.

CREATE TABLE IF NOT EXISTS modes (
    user_id           TEXT NOT NULL,
    name              TEXT NOT NULL,
    embedder          TEXT NOT NULL,          -- fastembed model key, e.g. "jina-embeddings-v2-base-code"
    embedding_dim     INT  NOT NULL,          -- 384 | 768 | 1024 — which embedding column this mode writes
    description       TEXT,
    default_decay     TEXT,                   -- 'fast' | 'medium' | 'slow' | 'none'
    prompt_overrides  JSONB,                  -- per-mode hints handed to the extraction call
    is_default        BOOLEAN NOT NULL DEFAULT false,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, name)
);

-- Only one default mode per user. Partial unique index so non-default rows are
-- unconstrained. The '*' template's `general` default is exempt from clashing
-- with a real user because the index is per (user_id).
CREATE UNIQUE INDEX IF NOT EXISTS modes_one_default_per_user_idx
    ON modes (user_id) WHERE is_default;

-- The canonical built-in registers. Seeded under the reserved '*' user; the
-- server clones these into a real user's rows on first use. Embedder keys +
-- dims come straight from docs/EMBEDDINGS.md.
INSERT INTO modes (user_id, name, embedder, embedding_dim, description, default_decay, is_default)
VALUES
    ('*', 'general',  'sentence-transformers/all-MiniLM-L6-v2', 384,
        'Default English conversational memory.', 'medium', true),
    ('*', 'code',     'jinaai/jina-embeddings-v2-base-code',    768,
        'Code / dev-work memory: symbols and framework names as concepts.', 'medium', false),
    ('*', 'journal',  'BAAI/bge-base-en-v1.5',                  768,
        'Reflective / emotional prose; slower decay.', 'slow', false),
    ('*', 'research', 'BAAI/bge-large-en-v1.5',                 1024,
        'Dense academic / technical text at deeper resolution.', 'medium', false)
ON CONFLICT (user_id, name) DO NOTHING;

-- ---------------------------------------------------------------------------
-- Per-dimension embedding columns on raw_embeddings. `embedding vector(384)`
-- stays (the general/MiniLM column); a record in a 768- or 1024-dim mode writes
-- the matching nullable column instead, and exactly one column is ever populated
-- per row. Partial IVFFlat cosine indexes so an empty column costs nothing.
-- ---------------------------------------------------------------------------
ALTER TABLE raw_embeddings ADD COLUMN IF NOT EXISTS embedding_768  vector(768);
ALTER TABLE raw_embeddings ADD COLUMN IF NOT EXISTS embedding_1024 vector(1024);

CREATE INDEX IF NOT EXISTS raw_embeddings_vec768_idx
    ON raw_embeddings USING ivfflat (embedding_768  vector_cosine_ops) WITH (lists = 100)
    WHERE embedding_768  IS NOT NULL;
CREATE INDEX IF NOT EXISTS raw_embeddings_vec1024_idx
    ON raw_embeddings USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists = 100)
    WHERE embedding_1024 IS NOT NULL;

-- The existing 384 index (005) is unconditional; make it partial too so the
-- three columns are symmetric and a null 384 slot doesn't sit in the index.
DROP INDEX IF EXISTS raw_embeddings_vec_idx;
CREATE INDEX IF NOT EXISTS raw_embeddings_vec_idx
    ON raw_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    WHERE embedding IS NOT NULL;

-- The original 384 column was NOT NULL (005); a mode that writes 768/1024 leaves
-- it empty, so it must become nullable.
ALTER TABLE raw_embeddings ALTER COLUMN embedding DROP NOT NULL;

-- ---------------------------------------------------------------------------
-- The same per-dimension columns + partial indexes on curated_embeddings, so a
-- curated node distilled inside a 768/1024 mode is stored in the matching column.
-- ---------------------------------------------------------------------------
ALTER TABLE curated_embeddings ADD COLUMN IF NOT EXISTS embedding_768  vector(768);
ALTER TABLE curated_embeddings ADD COLUMN IF NOT EXISTS embedding_1024 vector(1024);

CREATE INDEX IF NOT EXISTS curated_embeddings_vec768_idx
    ON curated_embeddings USING ivfflat (embedding_768  vector_cosine_ops) WITH (lists = 100)
    WHERE embedding_768  IS NOT NULL;
CREATE INDEX IF NOT EXISTS curated_embeddings_vec1024_idx
    ON curated_embeddings USING ivfflat (embedding_1024 vector_cosine_ops) WITH (lists = 100)
    WHERE embedding_1024 IS NOT NULL;

DROP INDEX IF EXISTS curated_embeddings_vec_idx;
CREATE INDEX IF NOT EXISTS curated_embeddings_vec_idx
    ON curated_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)
    WHERE embedding IS NOT NULL;

ALTER TABLE curated_embeddings ALTER COLUMN embedding DROP NOT NULL;
