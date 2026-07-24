-- Extensions the whole schema depends on.
--
--   vector    — pgvector, for the embedding columns on raw_embeddings /
--               curated_embeddings.
--   uuid-ossp — uuid_generate_v4(), the default for every primary key.
--
-- Everything downstream (raw_records → curated_* → catalog / proposals /
-- references / modes) builds on these two.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
