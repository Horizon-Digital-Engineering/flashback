CREATE TABLE principals (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL CHECK (kind IN ('person','agent','service')),
    display_name TEXT,
    clearance    TEXT NOT NULL DEFAULT 'internal'
                 CHECK (clearance IN ('public','internal','restricted','confidential')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    disabled_at  TIMESTAMPTZ
);

CREATE TABLE grants (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    principal_id TEXT NOT NULL REFERENCES principals(id),
    owner_id     TEXT NOT NULL REFERENCES principals(id),
    target_type  TEXT NOT NULL
                 CHECK (target_type IN ('everything','topic','thread','record','node')),
    target_id    TEXT,
    access       TEXT NOT NULL DEFAULT 'read'
                 CHECK (access IN ('read','write','admin')),
    scope_mode   TEXT NOT NULL DEFAULT 'ongoing'
                 CHECK (scope_mode IN ('pinned','ongoing')),
    valid_from   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until  TIMESTAMPTZ,
    granted_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    granted_by   TEXT NOT NULL REFERENCES principals(id),
    revoked_at   TIMESTAMPTZ,
    revoked_by   TEXT REFERENCES principals(id),
    CHECK ((target_type = 'everything') = (target_id IS NULL))
);

CREATE INDEX grants_principal_idx ON grants (principal_id, target_type, target_id)
    WHERE revoked_at IS NULL;
CREATE INDEX grants_owner_idx ON grants (owner_id) WHERE revoked_at IS NULL;
