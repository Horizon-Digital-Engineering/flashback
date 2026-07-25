-- Extensions the whole schema depends on: pgvector (embedding columns)
-- and uuid-ossp (uuid_generate_v4 primary keys).

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
