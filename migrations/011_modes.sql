-- ---------------------------------------------------------------------------
-- Modes (cognitive registers): each pins an embedder + dimension so records
-- are only compared within the same geometry. Built-ins seed under the
-- reserved '*' template user and are cloned per-user on first use.
-- ---------------------------------------------------------------------------
CREATE TABLE modes (
    user_id           TEXT NOT NULL,
    name              TEXT NOT NULL,
    embedder          TEXT NOT NULL,
    embedding_dim     INT  NOT NULL,       -- 384 | 768 | 1024
    description       TEXT,
    default_decay     TEXT,                -- 'fast' | 'medium' | 'slow' | 'none'
    prompt_overrides  JSONB,
    is_default        BOOLEAN NOT NULL DEFAULT false,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, name)
);
CREATE UNIQUE INDEX modes_one_default_per_user_idx ON modes (user_id) WHERE is_default;

INSERT INTO modes (user_id, name, embedder, embedding_dim, description, default_decay, is_default)
VALUES
    ('*', 'general',  'sentence-transformers/all-MiniLM-L6-v2', 384,
        'Default English conversational memory.', 'medium', true),
    ('*', 'code',     'jinaai/jina-embeddings-v2-base-code',    768,
        'Code / dev-work memory: symbols and framework names as concepts.', 'medium', false),
    ('*', 'journal',  'BAAI/bge-base-en-v1.5',                  768,
        'Reflective / emotional prose; slower decay.', 'slow', false),
    ('*', 'research', 'BAAI/bge-large-en-v1.5',                 1024,
        'Dense academic / technical text at deeper resolution.', 'medium', false);
