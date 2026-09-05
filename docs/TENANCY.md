# Tenancy & multi-user — exploratory design

*Status: **schema landed 2026-08-30, deliberately unenforced.** `principals`,
`grants` and `classifications` exist in `003_access.sql`; nothing reads them yet.
The split is intentional — the dimensions they record cannot be backfilled, so
they are captured from the first record onward, while enforcement waits until
there is a second person to enforce against. This document is the design frame;
what follows describes the model those tables were shaped for.*

---

## Today

Flashback is already multi-tenant at the **data layer**: every memory, every token, every consolidation run carries a `user_id`. Queries filter on it. The consolidation worker iterates per-user. The shape is there.

What it lacks is:

- **No tenant boundary.** All `user_id`s share one global namespace. No notion of "this group of users belongs together."
- **No enforcement.** `grants` can express "this principal may read this target
  until this date", and `classifications` can express how sensitive a record is,
  but no query path consults either yet. Access today is still "one `user_id`
  owns it".
- **Ownership and readership are still one field.** `raw_records.user_id` means
  creator *and* owner *and* the only reader. Splitting it is the next real step,
  and the honest shape is provenance immutable on raw, readership mutable in
  `grants` — which is why raw needs no change to get there.
- **No real auth flow.** Tokens are minted via CLI by the operator. No signup, no password reset, no per-user provisioning.

Two things the schema already decided, worth not re-litigating:

- **Agents are principals.** `principals.kind` is `('person','agent','service')`,
  so "the finance agent may read the tax agent's memory" is a grant, not a new
  mechanism. Sharing *files* between agents is a separate axis — filesystem
  permissions — and conflating the two puts access control in two places.
- **Grants are pinned or ongoing.** A grant on a single record is a frozen set; a
  grant on a topic or thread keeps including whatever arrives next. That is
  usually what you want and occasionally a leak, which is why grants carry a
  validity window rather than just an on/off.

What it *does* have is a role split, walled in both directions:

- **`service`** (the default) reaches REST + MCP only, and sees just its own `user_id`'s rows. This is what an integration holds.
- **`operator`** reaches the `/admin` UI only, and sees every user's rows. This is what you sign in with.

A service token is bounced from `/admin`; an operator token gets a 403 from the API. Mint with
`flashback token mint --user=<u> --name=<label> [--role=operator]`. `*` is reserved (operator
wildcard + the modes template) and must never be minted as a real user.

So the operator-vs-user boundary exists. What's still missing below is the *between-users* one.

The trait shape is right; the policy layer is missing.

---

## Two deployment shapes worth designing for

### 1. Self-hosted household / team

**Setup**: one Flashback install on a home server or office NAS. Multiple people — partners, kids, teammates — each have their own identity and memory space. Some memories are private; some are explicitly shared.

**Operator privacy is a non-issue**: the operator IS a household member. They control the box. They could read the database directly anyway. Privacy lives at the *between-users* boundary, not the operator-vs-everyone boundary.

**Driver use cases**:

- A family shares a grocery list, vacation plans, repair history, kids' school deadlines — but each adult also has private journaling, work notes, individual reflections.
- A small team shares project context, decisions, postmortems — but each engineer has private "thinking out loud" workspace and personal todo lists.
- A couple shares "us" memories (anniversaries, plans, friends) while each individually maintains their own.

**Shape that fits**: tenant + per-memory visibility, with shared visibility being explicit opt-in.

### 2. SaaS — operator is NOT the user

**Setup**: someone (HDE, a third party, an enterprise) hosts Flashback as a service. End users are strangers to the operator. The operator can technically see the database but should not.

**Operator privacy IS the problem**: the user-vs-operator boundary is now the critical one. Two honest paths:

- **Trust + audit** — operator *could* see content; admin UI shows only metadata; access logged; user agreement / law makes it explicit. What most SaaS products actually do.
- **Zero-knowledge** — content encrypted with user-held key; server stores ciphertext only; embedding + extraction happen client-side. What Signal / ProtonMail / Apple ADP do. Fundamentally restructures Flashback (LLM moves to the edge).

Different products at different stages of trust-need pick different points on this spectrum.

---

## Schema sketch (fresh-build, no migration assumed)

The household case is concrete enough to sketch. The SaaS case borrows the same primitives plus encryption.

```sql
-- A tenant is a household, team, org, "shared brain context". A user
-- belongs to exactly one tenant. (Future: cross-tenant federation is
-- a separate problem; out of scope here.)
CREATE TABLE tenants (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id           TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    is_operator  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Optional named groups within a tenant: "adults", "engineers", "the kids"
CREATE TABLE groups (
    id        TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name      TEXT NOT NULL,
    UNIQUE (tenant_id, name)
);

CREATE TABLE group_members (
    group_id TEXT NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    user_id  TEXT NOT NULL REFERENCES users(id)  ON DELETE CASCADE,
    PRIMARY KEY (group_id, user_id)
);

ALTER TABLE memories
    ADD COLUMN tenant_id  TEXT NOT NULL REFERENCES tenants(id),
    ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private',
    -- visibility values:
    --   'private'         → only owner (user_id) can read
    --   'tenant'          → any user in the tenant can read
    --   'group:<group_id>'→ any user in that group can read
    ADD CONSTRAINT visibility_valid CHECK (
        visibility IN ('private', 'tenant')
        OR visibility LIKE 'group:%'
    );
```

Apply the same `tenant_id` + `visibility` shape to `state_objects`. Tokens carry `user_id` (which transitively pins `tenant_id`); no need for token-level scope yet.

### Reads under this schema

Every memory query, today, is `WHERE user_id = $auth.user_id`. Becomes:

```sql
SELECT ... FROM memories
WHERE tenant_id = $auth.tenant_id
  AND (
        visibility = 'tenant'
        OR (visibility = 'private' AND user_id = $auth.user_id)
        OR (visibility LIKE 'group:%'
            AND substring(visibility from 7) IN (
                SELECT group_id FROM group_members WHERE user_id = $auth.user_id
            ))
      )
```

Verbose at the app layer. Postgres has [Row-Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html) that handles this at the DB level once you `SET app.current_user_id = '...'` per connection — the policy runs automatically on every query. Way less footgun than remembering the WHERE clause in 30 different handlers.

---

## Worked example — a family

Household tenant `home-001` with three users (parent-A, parent-B, 12yo kid). Groups: `adults`, `everyone`.

```
| memory                                       | owner    | visibility       |
|----------------------------------------------|----------|------------------|
| "rough night — didn't sleep well"            | parent-A | private          |
| "grocery list: eggs, milk, lunch items"      | parent-B | tenant           |
| "trip planning — initial budget thinking"    | parent-A | group:adults     |
| "trip is on, dates confirmed"                | parent-A | tenant           |
| "homework checklist for Tuesday"             | kid      | private          |
| "swim practice moved to Thursdays"           | parent-B | tenant           |
```

The kid searches "swim practice" → returns the practice-change memory. The kid searches "trip budget" → no results (not in `adults` group). parent-A searches anything → sees their own private + the tenant + the adults-group memories. parent-B searches → same as parent-A's view minus the journal.

Consolidation can run in two modes:
- **Per-user**: distills only that user's memories (private + visible). Personal patterns.
- **Per-tenant**: distills the tenant-visibility memories across all users in the tenant. Family-level patterns ("the family has talked about the trip for weeks"). Result is stored at tenant visibility.

Both run on the same worker; just different SQL filters.

---

## Operator-vs-tenant boundary (SaaS case)

The household model trusts the operator. SaaS must not. Three minimums:

1. **Admin endpoints must redact content**. Show counts, latency, model usage, error rates, billing-relevant numbers. Never raw memory text, never extraction entities, never search queries. Build it deliberately into the admin view shape.
2. **Operator access is logged**. Any time an operator views a user-scoped page or runs a maintenance query that touches user data, write it to an audit table the user can later inspect.
3. **No content in error reports**. Sentry / logs / panics scrub user text before egress. Otherwise the trust-model dies on the first prod incident.

Zero-knowledge (ciphertext at rest, embedding-on-client) is the gold standard but pushes the LLM call to the user's device. That changes Flashback's deployment story entirely — it becomes a sync service for client-encrypted notes, not a server-side memory system. Worth keeping in mind; not a near-term path.

---

## Admin role within a tenant

Separate from "SaaS operator." A tenant might want an internal admin role too — the parent who provisions kids' accounts, the team lead who manages group membership, the family-IT person who reads consolidation reports.

The `users.is_operator` flag in the sketch above is the bare minimum. A real role model would be:

- `member` — default. Sees own + tenant + group-shared memories.
- `tenant_admin` — can read tenant-level admin views (consolidation runs for the tenant), manage groups, invite users. Cannot read other users' *private* memories.
- `tenant_owner` — billing, deletion, owner of the tenant entity.

(SaaS operator is *above* all of these and lives outside the tenant; tenant_admin is internal.)

---

## Modes × Tenancy

[MODES.md](MODES.md) and this doc are **orthogonal axes**. A memory has:

- **Subject / visibility** (this doc): who owns it, who can see it
- **Mode** (MODES.md): which cognitive register it lives in (code / journal / family / etc.)

A `family` mode is a thing a user might declare — memories in that mode default to `tenant` visibility, get embedded with a conversational embedder, classified with a family-tuned extraction prompt. That's the convergence point: modes can declare default visibilities, but the visibility column is the authoritative axis.

Don't conflate them. Modes are about *how I think*; visibility is about *who gets to see what I think*.

---

## Open questions

1. **One tenant per user, or multi-tenant users?** A consultant might work with 3 teams. Today's sketch says one tenant per user. Cross-tenant federation (memories from team A visible to user-in-team-B with permission) is a much bigger problem; punt.
2. **Memory promotion across visibility levels.** "I journaled about the trip privately, eventually want to share with another user in the tenant." Should there be a copy-on-promote, or a re-tag-in-place? Copy preserves audit history; re-tag is cheaper. Lean copy.
3. **Consolidation scope.** Per-tenant distillation needs explicit user opt-in per memory, OR it inherits from the source memory's visibility (any tenant-visible memory is eligible; private memories never enter the family-level distillation). Lean inheritance.
4. **Embedding cost.** Two users in a family writing similar content end up with two embeddings of nearly the same vector. Tenant-level dedup is technically possible but the savings vs the complexity is probably not worth it until volume is large.
5. **Group operations under modes.** If `family` mode auto-tags `tenant` visibility, what happens when one family member sets a memory to mode=`code`? Should it auto-flip to private? Probably yes — mode declares the *intended audience* implicitly.

---

## What would kill the idea

- Self-hosted households turn out to be a vanishingly small audience and SaaS is the only real use case. Then we'd skip straight to operator-trust + per-user scoping; no group concept needed.
- Postgres Row-Level Security has surprising performance costs at scale (it does — millions of memories with complex policies can hurt). At which point you push the access check up to the app layer and live with the verbose WHERE clauses.
- Cross-user consolidation produces creepy or invasive distilled facts ("the family is fighting about money"). At which point per-user consolidation is the only safe mode and the shared-memory feature gets walked back.

---

## Status

**Schema shipped, enforcement deliberately cold.** The principals and grants tables exist so history starts now; nothing reads them until there is a second person. Captured so the next architecture move has a frame instead of a blank canvas.

The current `user_id` field is the seed of all this — already in the schema, already filtered on. Tenants and visibility are additive layers, not a rewrite.
