# Flashback

**Dynamic episodic memory for LLMs.**

[![CI](https://github.com/Horizon-Digital-Engineering/flashback/actions/workflows/ci.yml/badge.svg)](https://github.com/Horizon-Digital-Engineering/flashback/actions/workflows/ci.yml)
[![Security](https://github.com/Horizon-Digital-Engineering/flashback/actions/workflows/security.yml/badge.svg)](https://github.com/Horizon-Digital-Engineering/flashback/actions/workflows/security.yml)
[![CodeQL](https://github.com/Horizon-Digital-Engineering/flashback/actions/workflows/codeql.yml/badge.svg)](https://github.com/Horizon-Digital-Engineering/flashback/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/Horizon-Digital-Engineering/flashback/badge)](https://scorecard.dev/viewer/?uri=github.com/Horizon-Digital-Engineering/flashback)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Horizon-Digital-Engineering_flashback&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=Horizon-Digital-Engineering_flashback)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=Horizon-Digital-Engineering_flashback&metric=coverage)](https://sonarcloud.io/component_measures?id=Horizon-Digital-Engineering_flashback&metric=coverage)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=Horizon-Digital-Engineering_flashback&metric=security_rating)](https://sonarcloud.io/summary/new_code?id=Horizon-Digital-Engineering_flashback)
[![Release](https://img.shields.io/github/v/release/Horizon-Digital-Engineering/flashback?sort=semver&color=8A2BE2)](https://github.com/Horizon-Digital-Engineering/flashback/releases)
[![License: BUSL-1.1](https://img.shields.io/badge/license-BUSL--1.1-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-stable-orange.svg?logo=rust)](https://www.rust-lang.org/)

---

Flashback is a self-contained memory microservice that gives any LLM genuine episodic memory: dynamic retrieval that updates *within* a conversation, temporal history that preserves how beliefs evolved, and a four-tier memory hierarchy modeled on how human cognition actually works.

It speaks REST. It plugs into any stack. The complexity is inside.

---

## Why Flashback?

### Where conventional approaches fall short

Most current AI memory systems follow the same pattern: chunk some text, embed it, store the vector, return top-k at query time. It works for static knowledge bases. It misses something important for conversational memory.

Static RAG treats memory as a frozen index. The system retrieves the same way on your first message as on your tenth — without visibility into what you established in between. By turn five of a conversation, a human has already filed away your preferences, made inferences, updated their mental model. A static retriever is still looking at the same snapshot from before you said hello.

Consider how a real conversation evolves:

1. *"Help me debug this issue."*
2. *"I'm using the Acme framework — here's the error."*
3. *"Actually, this is the same pattern as last week's problem with the database layer."*
4. *"Right, and that was caused by the migration we ran in February."*

At message 4, a static RAG system is still retrieving from an index that was built before this conversation started. It doesn't know messages 2 and 3 exist as context. It can't connect "the migration" to the thread you just established — unless that context happened to be in a prior session that was already indexed.

A human with good memory would have been updating their mental model continuously. By message 3, they've already cross-referenced last week's problem. By message 4, they're reaching back further and surfacing the root cause.

> "The memory isn't dynamic or grows as you converse more — that's what we need to add to make every LLM technically a dynamic RAG... you start generic in one thread then as memories pop up you tailor"

### Dynamic RAG: the conversation as a living index

The fix is to close the loop *within* the conversation. After each exchange, new context is ingested immediately — so the next query runs against an index that already knows what was just established.

> "Human speech is dynamic RAG"

The flow isn't:
```
Query memory → Talk → Ingest after conversation ends
```

It's:
```
User says something
  → Query memory
  → LLM responds
  → Ingest that exchange immediately
  → Next message retrieves against an updated index
```

The context window is alive. Not a snapshot.

> "We actively modify our context based upon what we are currently working on... with AI there's multiple parts that have to work together"

### Memory is a hierarchy, not a flat index

Human cognition and computer architecture solved the same problem independently. Both landed on the same answer: a tiered hierarchy with distinct latency, capacity, and persistence tradeoffs at each level.

> "There's long term memory... there's short term memory... there's 'subconscious' memory... even computers — short term (working memory), long term (hard drive) and fast access (instinct, cache) and subconscious (ROMs... built-in things that are just known)"

Flashback maps this hierarchy directly onto its architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1 — ROM / Subconscious                                    │
│  The LLM's training weights. Language, world knowledge, reason- │
│  ing. Baked in. We don't store this — we build on top of it.   │
├─────────────────────────────────────────────────────────────────┤
│  TIER 2 — Cache / Instinct                                      │
│  Core memory. Always injected. User prefs, active project,      │
│  behavioral rules. Zero retrieval cost. Always in context.      │
├─────────────────────────────────────────────────────────────────┤
│  TIER 3 — RAM / Working Memory                                  │
│  Active context with TTL. In-progress tasks, recent turns,      │
│  things mentioned this session. High relevance, fast decay.     │
├─────────────────────────────────────────────────────────────────┤
│  TIER 4 — Disk / Long-Term Memory                               │
│  Episodic + semantic store. Facts, decisions, project history,  │
│  supersede chains. Searched via hybrid retrieval on demand.     │
└─────────────────────────────────────────────────────────────────┘
```

Most memory systems today collapse this entire hierarchy into a single tier: the disk. There's no cache. There's no RAM. There's no concept of what's immediately relevant versus what might be dug up later. Everything is equally distant.

### Memory isn't a database of current facts — it's a web of episodes

The subtler problem: most memory systems treat storage as a database of *current* truths. Old facts get deleted when new ones supersede them.

But human memory isn't a current-facts database. It's a web of episodes. You don't just know that you use a particular tool — you remember the moment you chose it, the alternatives you considered, the pivot you made six months in. Dead ends matter. Pivots matter. The decision trail is part of the memory.

> "The evolution of an idea is not simply a replace — there's a temporal component"

Flashback never deletes. Old records are marked superseded; new records carry a pointer back. A query for "what's current" returns the latest node. A query for "how did this evolve" walks the chain. The narrative is preserved.

---

## Features

- **Raw and curated** — an append-only raw layer of three record types (`conversation`, `document`, `state_object`) and a rebuildable layer above it. Raw holds only what a writer sent; anything we worked out by reading it — the embedding register, the resolved order, tags, scores — lives above and a rebuild is free to disagree. See [docs/REFERENCES.md](docs/REFERENCES.md).
- **Supersede-not-delete** — full temporal history of how beliefs evolved, queryable at any point in time
- **Ordered, tamper-evident** — each record carries the writer's claim about what preceded it, so conversations keep their true order and their forks without trusting clocks; separately, a sha256 chain over arrival order makes an edited record invalidate every record stored after it
- **Hybrid retrieval** — weighted combination of semantic similarity, BM25 keyword match, recency, and entity overlap, over HNSW vector indexes
- **Two retrieval modes** — *answer mode* for questions, *manager mode* for reconstructing "what's going on" at session start
- **Decay model** — retrieval scores degrade by half-life (not deletion); pinned memories never decay
- **MCP server included** — wrap Flashback as a remote MCP server (Streamable HTTP) so Claude Desktop, Claude Code, Cursor, or anything MCP-aware can share memory across clients
- **Bearer-token auth** — sha256-hashed at rest, scoped to a `user_id`; mint per-client tokens via CLI
- **Self-contained** — Docker Compose brings up the entire stack; no external API keys required

---

## Architecture

```
                Claude Desktop / Cursor / Claude Code / OpenAI Agents
                                       │
                                       ▼  Streamable HTTP MCP (JSON-RPC)
                  ┌────────────────────────────────────────┐
                  │  flashback-mcp        (port 8082)      │  Rust / rmcp
                  │  Wraps REST as typed tools.            │
                  │  Forwards Bearer to upstream.          │
                  └────────────┬───────────────────────────┘
                               │  HTTP, Bearer auth
                               ▼
                  ┌────────────────────────────────────────┐
                  │  flashback (server)   (port 8080)      │  Rust / Axum
                  │  REST API — memory + state CRUD,       │
                  │  hybrid retrieval, context assembly,   │
                  │  supersede chains, token management.   │
                  └────────────┬───────────────────────────┘
                               │
                      ┌────────┴────────┐
                      ▼                 ▼
        ┌─────────────────────┐  ┌─────────────────────────┐
        │  crates/nlp         │  │  Postgres + pgvector    │
        │  (in-process)       │  │  raw_records, derived_*,│
        │  Embeddings (MiniLM)│  │  curated_*              │
        │  Entity extraction  │  └─────────────────────────┘
        └─────────────────────┘
```

**Stack:** Rust workspace (`crates/server` Axum + SQLx + Tokio, `crates/mcp` rmcp + Streamable-HTTP, `crates/nlp` fastembed-rs embeddings + rule-based entity extraction, all in-process) · PostgreSQL 16 + pgvector · Docker Compose

---

## Quick Start

```bash
git clone https://github.com/Horizon-Digital-Engineering/flashback
cd flashback

# No external API keys required.
docker compose up --build
```

Four services come up:

| Service       | Port | What it does                                          |
|---------------|------|-------------------------------------------------------|
| `server`      | 8080 | REST API + admin UI (bearer auth)                     |
| `mcp`         | 8082 | MCP server (Streamable HTTP) for AI clients           |
| `db`          | 5432 | Postgres 16 + pgvector                                |
| `ollama`      | 11434 | Local models for extraction/distillation (loopback)  |

`pgweb`, a raw data browser on 8081, is opt-in: `docker compose --profile tools up`.

Migrations run automatically on first boot (`AUTO_MIGRATE=1` in the included `docker-compose.yml`). Verify:

```bash
curl http://localhost:8080/health
curl http://localhost:8082/health
```

### Mint a token

Tokens carry a **role**, and the two surfaces are walled off from each other:

| Role | Reaches | Sees |
|---|---|---|
| `service` (default) | REST API + MCP | its own user's records |
| `operator` | the `/admin` UI | the whole estate, every user |

A service token cannot open the admin UI; an operator token cannot call the
API. Mint what the client actually needs:

```bash
# integration credential (ritsu, Claude Code, Cursor …)
docker compose exec server ./flashback token mint --user=alice --name=claude-code

# your own admin login
docker compose exec server ./flashback token mint --user=alice --name=admin --role=operator
#   Token minted for user=alice role=operator
#   ID:     <uuid>
#   TOKEN:  fb_<32 base32-ish chars>     ← shown ONCE; save it now

docker compose exec server ./flashback token list
docker compose exec server ./flashback token revoke <id>
```

The plaintext token is shown exactly once at mint; only its sha256 is stored. Mint one token per client (Claude Desktop, Cursor, Claude Code, your own app) — all scoped to the same `user_id` means they all see the same memory.

### Wire up an MCP client

Point any MCP-aware client at `http://<your-host>:8082/mcp` (Streamable HTTP) with the bearer token. Examples:

**Claude Desktop / Claude Code (`~/.config/claude-desktop/...` or via Settings → MCP):**

```json
{
  "mcpServers": {
    "flashback": {
      "url": "https://flashback.yourdomain.com/mcp",
      "headers": { "Authorization": "Bearer fb_YOUR_TOKEN_HERE" }
    }
  }
}
```

**Cursor (`.cursor/mcp.json` or Settings → MCP):**

```json
{
  "mcpServers": {
    "flashback": {
      "url": "https://flashback.yourdomain.com/mcp",
      "headers": { "Authorization": "Bearer fb_YOUR_TOKEN_HERE" }
    }
  }
}
```

The model will then have 11 tools: `flashback_remember`, `flashback_search`, `flashback_assemble_context`, `flashback_supersede`, `flashback_lineage`, `flashback_core_add`, `flashback_core_list`, `flashback_state_create`, `flashback_state_get`, `flashback_state_patch`, `flashback_state_history`.

### Optional: smarter extraction via a hosted LLM

By default the extraction step that powers supersede detection and the new structured `extraction` field is a **rule-based** pass in pure Rust — fast, free, no external calls. For richer extraction (better topic resolution, intent classification, operation detection, coreference) point the system at a hosted LLM:

```bash
# .env
PROVIDER=remote
PROVIDER_REMOTE_PROVIDER=openrouter           # or anthropic | openai
PROVIDER_REMOTE_MODEL=anthropic/claude-haiku-4-5
OPENROUTER_API_KEY=sk-or-...                  # or ANTHROPIC_API_KEY / OPENAI_API_KEY
PROVIDER_FALLBACK=fail                        # `heuristic` for silent fallback on outage
```

OpenRouter speaks OpenAI's API shape and routes to ~all hosted models, so one config supports many providers. Direct Anthropic (`PROVIDER_REMOTE_PROVIDER=anthropic`) uses the Messages API with prompt-cache enabled for ~90% discount on system-prompt tokens.

**Privacy note:** with `PROVIDER=remote`, each ingested memory is sent to your chosen LLM provider for extraction. If you're already using this MCP server with Claude Desktop or similar, the LLM provider already sees your conversation — choosing `anthropic` for both routes the data to one party instead of two.

**The 90% case is the remote provider — whether the model is in the cloud or on a box next to you.** A "sidecar Ollama container, an M-series Mac on your LAN, a DGX Spark over Tailscale, Claude over the public internet" — they're all the same code path from Flashback's perspective. Just a different URL. Set `PROVIDER=remote` plus the right `PROVIDER_REMOTE_API_BASE` and you're done.

An *embedded* LLM provider (model running in-process inside the Flashback binary itself via mistral.rs — no HTTP at all) exists behind the `embedded-llm` build feature, off by default because the dependency is heavy. That path is narrower: it matters when Flashback IS the only service on a dedicated AI box, or in air-gapped deployments where no network egress is acceptable. For everyone else, the remote provider is the right pattern. **Running Ollama as a sidecar container today is already exactly this**:

```bash
# Host: install Ollama and pull a small model
ollama pull qwen2.5:3b
ollama serve     # listens on localhost:11434

# Flashback .env
PROVIDER=remote
PROVIDER_REMOTE_PROVIDER=openai
PROVIDER_REMOTE_MODEL=qwen2.5:3b
PROVIDER_REMOTE_API_BASE=http://host.docker.internal:11434/v1   # macOS/Windows
# PROVIDER_REMOTE_API_BASE=http://172.17.0.1:11434/v1            # Linux
# No API key needed: with an explicit PROVIDER_REMOTE_API_BASE the endpoint is
# self-chosen, and self-hosted model servers don't check one.
```

All of these environment values are the **bootstrap seed**. Once the server is up, the admin UI's **Settings** page (`/admin/settings`) configures the provider at runtime — pick the model from what the endpoint actually serves, test one real extraction, save, and the pipeline swaps over live with no restart. Saved settings are stored in the database and win over the environment from then on. The API key is the one exception: it stays in the environment and is never stored.

Hardware roadmap: DGX Spark, M-series 128 GB, Strix Halo will all run 70B+ comfortably. Two ways to use them:

- **Remote provider pointing at the box** (recommended) — Ollama / vLLM / whatever runs there, Flashback elsewhere points at it via URL. Same `PROVIDER=remote`, just a different `PROVIDER_REMOTE_API_BASE`.
- **Embedded provider on the box itself** (`--features embedded-llm`) — Flashback runs ON the AI box, owns the GPU directly via mistral.rs, ships as one binary. No HTTP boundary at all. Right answer when Flashback is the only service the box runs.

### Web admin UI

For every deployment there's a built-in admin UI at `/admin`. Login with any minted bearer token, then browse:

- **Dashboard** — counts, provider info, recent memories
- **Memories** — filterable table with delete actions
- **Memory detail** — full content, structured extraction, lineage (supersede chain) walked both directions
- **State** — current value of every state object you maintain
- **Settings** — the live extraction/distillation provider: model picked from what the endpoint actually serves, one-click test extraction, applies without a restart
- **Playground** — a seedable, sandboxed test bench for the whole loop: retrieval, the exact prompt, per-turn extraction, the streamed reply, and one-click distillation of the conversation into facts. Writes stay in a sandbox scope; reads can opt into the real store
- **Tokens** — list + revoke per-client tokens
- **Mind map** — force-directed network graph of your memories, edges for supersede / entity-overlap / same-thread. Server-rendered HTML + inline vanilla JS, no framework, no CDN, no external assets.

Visit `http://<your-host>:8080/admin` after `docker compose up`.

### Hosting

Two paths depending on what you're optimizing for.

#### One-click — DigitalOcean App Platform (~$45/mo)

[![Deploy to DO](https://www.deploytodo.com/do-btn-blue.svg)](https://cloud.digitalocean.com/apps/new?repo=https://github.com/Horizon-Digital-Engineering/flashback/tree/main)

App Platform reads [`.do/app.yaml`](.do/app.yaml) and provisions: a managed Postgres cluster, the REST server (at `/`), and the MCP server (at `/mcp`). HTTPS is automatic. After deploy, mint your first bearer:

```bash
doctl apps list                                # find your app id
doctl apps exec <app-id> server -- \
    ./flashback token mint --user=admin --name=initial --role=operator
```

Your MCP URL is `https://<app>.ondigitalocean.app/mcp`. Paste it + the bearer into your client config.

#### Cheap — one-line droplet install (~$24/mo)

On any fresh Ubuntu 22.04+ box:

```bash
curl -sSL https://raw.githubusercontent.com/Horizon-Digital-Engineering/flashback/main/deploy/install.sh | bash
```

The script installs docker, clones the repo into `/opt/flashback`, generates a strong `POSTGRES_PASSWORD`, brings the stack up, and mints an initial bearer token at `/root/FLASHBACK_TOKEN.txt`. Add Caddy + a domain for HTTPS.

Full guide (sizing, TLS via Caddy, backups, generic VPS): [deploy/README.md](deploy/README.md).

---

## Integration

There are two integration paths. **MCP** is the high-leverage path — your LLM client speaks MCP, no app code required. **Direct REST** is for custom apps that own the chat surface.

Every request requires `Authorization: Bearer <token>`. `user_id` is derived from the token — you never pass it in request bodies.

### Direct REST — the two-call pattern

**Before the LLM call** — retrieve context:

```bash
curl -H "Authorization: Bearer $TOKEN" -X POST http://localhost:8080/records/context \
  -d '{
    "thread_id": "chat_abc123",
    "query":     "what'"'"'s the current deploy target?",
    "limit":     20
  }'
```

Optional: `topic_id` (where the thread is filed), `mode` or `modes` (which
cognitive register to search), and `exclude_thread_id` — usually the live
conversation, since memory means *other* threads.

The response is bounded by a character budget, so it cannot blow your context
window.

**After the LLM call** — record each turn:

```bash
curl -H "Authorization: Bearer $TOKEN" -X POST http://localhost:8080/records \
  -d '{
    "type":       "conversation",
    "content":    "The current deploy target is production.",
    "source":     "myapp:assistant",
    "source_ref": "myapp:chat_abc123:42",
    "thread_id":  "chat_abc123",
    "prev_source_ref": "myapp:chat_abc123:41",
    "payload":    { "model": "claude-opus-5", "usage": { "output_tokens": 91 } }
  }'
```

One record per turn — a user turn and an assistant turn are two records, not one.

Three fields are worth getting right:

- **`source_ref`** is your own stable id for the message. It makes ingest
  idempotent: re-sending the same record updates nothing instead of duplicating.
- **`prev_source_ref`** is the `source_ref` of the turn this one followed. You
  never handle a flashback id — it resolves yours. This builds the chain that
  gives conversations real order and makes forks representable, so send turns
  for one thread **in order**.
- **`payload`** is whatever metadata you already have — model, token usage,
  latency, attachments. Stored verbatim, never interpreted. Some of it cannot be
  recovered later, so record it now even if nothing reads it yet.

### State objects — the reference half of memory

Things you *maintain* rather than *log* (a running todo list, an evolving plan)
are records of `type: "state_object"`, and they get a projected API of their own
under `/records/state`. The terminal record for a `(kind, key)` is the current
value; the supersede chain is its whole history.

```bash
# set (or revise) a value — each POST appends, nothing is mutated
curl -H "Authorization: Bearer $TOKEN" -X POST \
  http://localhost:8080/records/state/todo_list/deploy_checklist \
  -d '{ "data": { "items": [{ "text": "run tests" }, { "text": "build image" }] } }'

# current value
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/records/state/todo_list/deploy_checklist

# every revision, oldest first
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/records/state/todo_list/deploy_checklist/history

# everything of this kind
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/records/state/todo_list
```

See [docs/REFERENCES.md](docs/REFERENCES.md).

### Structured reads

```bash
# filter by scope and type
curl -H "Authorization: Bearer $TOKEN" -X POST http://localhost:8080/records/query \
  -d '{ "thread_id": "chat_abc123", "type": "conversation", "limit": 50 }'

# bulk load a corpus (idempotent on source_ref)
curl -H "Authorization: Bearer $TOKEN" -X POST http://localhost:8080/records/import \
  -d '{ "records": [ ... ] }'

# one record, its supersede chain, and what curation derived from it
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/records/{id}
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/records/{id}/lineage
curl -H "Authorization: Bearer $TOKEN" http://localhost:8080/records/{id}/derivations
```

---

## Memory Types

The raw layer accepts three record types — what a writer sends, stored verbatim:

| Raw type | What it is |
|----------|------------|
| `conversation` | One turn of an exchange, user or assistant side |
| `document` | Ingested reference material |
| `state_object` | A write to a named mutable cell (`todo_list`, `plan`, …) — the **reference** half; see [docs/REFERENCES.md](docs/REFERENCES.md) |

Everything cognitive is derived above raw by curation, and a rebuild is free to
disagree with all of it:

| Curated kind | Analogy | What it holds |
|--------------|---------|---------------|
| `episodic` | Short-term recall | Per-conversation episodes formed from raw turns |
| `semantic` | Long-term facts | Distilled beliefs, dated by their newest evidence |
| `summary` | The table of contents | Rolled-up view over episodes and facts |
| `entity` | The index | Noun phrases linking records that mention the same thing |

---

## The Name

Mid-conversation, something triggers a memory. The system surfaces it — not because you queried for it explicitly, but because the current context activated it. A flashback.

That's the experience the system is trying to produce. Not "here are the top 5 relevant documents." Not "here's what you told me to remember." But: the right memory, at the right moment, because the context demanded it.

Human conversations aren't transactions. They're journeys. Flashback is memory that travels with you.

---

## Docs

- [docs/VISION.md](docs/VISION.md) — the dynamic RAG thesis; why current memory is broken and what fixing it looks like
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full system design: memory hierarchy, memory types, decay model, consolidation pipeline, hybrid retrieval, supersede chains, 3D visualization, database schema
- [docs/REFERENCES.md](docs/REFERENCES.md) — the heap hypothesis: records vs references as the missing first-class split. Why a todo list is not a sequence of facts.
- [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) — embedding model choice by use case. Code-heavy conversation, multilingual, dense technical text — different recommendations. "More dimensions = better" is a myth; pick deliberately.
- [docs/MODES.md](docs/MODES.md) — **shipped.** First-class cognitive registers (code / general / journal / research), each with their own embedder and retrieval geometry. The brain-mode metaphor — humans don't run parallel brains, they switch register. Registers are per-user rows with a `/modes` CRUD API; the register a record lands in is derived, so a rebuild can reclassify.
- [docs/MODEL-TIERING.md](docs/MODEL-TIERING.md) — **exploratory, shipped.** Split the remote LLM model by role: a small fast model for `extract()` (write path, sub-2s) and a larger reasoning model for `distill_facts()` (background, minutes-OK). The trait was always designed for this; the config never followed.
- [docs/TENANCY.md](docs/TENANCY.md) — **schema shipped, enforcement deliberately cold.** Multi-user + tenant + visibility design. Self-hosted households (operator IS a user) vs SaaS (operator must NOT see user content) are different problems. The principals/grants tables exist; nothing enforces them until there is a second person.

---

## Status

**Alpha, end-to-end functional, releases signed.** The dynamic-RAG loop, the
append-only raw layer with its tamper-evident hash chain, curation
(episodes, distilled facts, summaries), hybrid retrieval, state objects,
cognitive registers, the REST API, the MCP server, and the admin UI all work.
Rust-native NLP runs in-process — embeddings via fastembed-rs, rule-based
entity extraction, and a pluggable LLM provider (remote / heuristic) for
extraction and distillation. See [CHANGELOG.md](CHANGELOG.md) for what each
release changed; 0.2.0 is breaking, and the changelog leads with why.

Still open, tracked and deliberate:

- Retrieval quality: two-stage ANN retrieval, decay that rewards being
  remembered instead of punishing it, and a synthesis-first recall contract
- Decay-based archival (decay only affects ranking; nothing is GC'd)
- Document chunking for large ingested files
- OAuth2 / OIDC (bearer tokens only today)
- Multi-user enforcement (schema shipped, cold until there is a second person)

If the thesis resonates, watch the repo. Feedback welcome — open an issue.

---

## License

Business Source License 1.1 — see [LICENSE](LICENSE). Auto-converts to Apache 2.0 on 2030-05-23 (or 4 years after each version's first publication, whichever is earlier). Source-available; non-production use is freely permitted. Production use is permitted except for offering Flashback as a hosted/managed service that competes with one offered by the Licensor.
