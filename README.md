# Flashback

**Dynamic episodic memory for LLMs. Not a search index — a brain.**

---

> "Every memory solution right now is just an MCP server — a thin wrapper around vector search. Store some chunks, cosine similarity, return top-k. That's a search index pretending to be a brain."

Flashback is a self-contained memory microservice that gives any LLM genuine episodic memory: dynamic retrieval that updates *within* a conversation, temporal history that preserves how beliefs evolved, and a four-tier memory hierarchy modeled on how human cognition actually works.

It speaks REST. It plugs into any stack. The complexity is inside.

---

## Why Flashback?

### The problem with every existing solution

Current AI memory systems are all variations on the same pattern: chunk some text, embed it, store the vector, return top-k at query time. It works. It's also the wrong model.

Static RAG treats memory as a frozen index. The system retrieves the same way on your first message as on your tenth — oblivious to everything you established in between. By turn five of a conversation, a human has already filed away your preferences, made inferences, updated their mental model. A static retriever is still looking at the same snapshot from before you said hello.

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

- **Episodic memory** — conversation snapshots with entities, timestamps, and session context, not just extracted facts
- **Supersede-not-delete** — full temporal history of how beliefs evolved, queryable at any point in time
- **Five memory types** — episodic, semantic, working, document, and procedural, each with distinct decay behavior
- **Hybrid retrieval** — weighted combination of semantic similarity, BM25 keyword match, recency, importance, project context, and entity overlap
- **Two retrieval modes** — *answer mode* for questions, *manager mode* for reconstructing "what's going on" at session start
- **Consolidation pipeline** — scheduled promotion from working memory to long-term; episodic records distilled into semantic facts over time
- **Decay model** — retrieval scores degrade by half-life (not deletion); pinned memories never decay
- **Automatic task extraction** — candidate tasks identified from conversation text without explicit commands
- **3D memory visualization** — UMAP-projected interactive graph with time scrubbing, type filtering, and supersede chain highlighting
- **Self-contained** — REST API, no external dependencies beyond PostgreSQL and the Python sidecar

---

## Architecture

```
┌──────────────────────────────────────┐
│  Rust / Axum  (port 8080)            │
│  Core API — memory CRUD, hybrid      │
│  retrieval, context assembly,        │
│  consolidation, visualization API    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  Python Sidecar  (port 8081)         │
│  Embedding, NER, LLM summarization   │
│  for consolidation + extraction      │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│  PostgreSQL + pgvector               │
│  Primary store — vector similarity,  │
│  BM25 full-text, temporal graph via  │
│  adjacency table + recursive CTEs    │
└──────────────────────────────────────┘
```

**Stack:** Rust (Axum, SQLx, Tokio) · Python (FastAPI) · PostgreSQL 16 + pgvector · Docker Compose

---

## Quick Start

```bash
git clone https://github.com/your-org/flashback
cd flashback

# Set your embedding API key
export OPENAI_API_KEY=sk-...

# Start everything
docker compose up
```

The API is live at `http://localhost:8080`. PostgreSQL at `5432`, Python sidecar at `8081`.

Run migrations on first boot:

```bash
docker compose exec server flashback migrate
```

---

## Integration

Flashback is designed around a two-call pattern that wraps your existing LLM calls.

### Before the LLM call — retrieve context

```bash
POST /context/assemble
{
  "session_id": "sess_abc123",
  "user_id":    "user_xyz",
  "project_id": "proj_flashback",
  "query":      "what's the current deploy target?"
}
```

Returns fully assembled layered context — procedural patterns, project state, retrieved memories, relevant document chunks, recent conversation — ready to inject into your system prompt. Not top-k chunks. A structured prompt layer.

### After the LLM call — ingest the exchange

```bash
POST /memory/ingest
{
  "session_id":     "sess_abc123",
  "user_id":        "user_xyz",
  "project_id":     "proj_flashback",
  "user_turn":      "what's the current deploy target?",
  "assistant_turn": "The current deploy target is production..."
}
```

That's it. The exchange is chunked, embedded, entity-extracted, and written to working memory. The next `/context/assemble` call retrieves against an index that already includes this exchange. The loop is closed.

### Core memory (always-on context)

```bash
# Pin a fact that appears in every prompt — no retrieval required
POST /memory/core
{
  "content": "Always use TypeScript. Never suggest full rewrites.",
  "user_id": "user_xyz"
}
```

### Semantic search

```bash
POST /memory/search
{
  "query":       "auth middleware changes",
  "mode":        "answer",
  "project_id":  "proj_flashback",
  "top_k":       10
}
```

### Temporal lineage

```bash
# Walk the supersede chain for any entity — see how a belief evolved
GET /lineage/auth_middleware
```

---

## Memory Types

| Type | Analogy | Decay | Description |
|------|---------|-------|-------------|
| `working` | RAM | Fast (48h TTL) | Active session context; promotes to episodic on close |
| `episodic` | Short-term recall | Medium (14d) | Conversation snapshots; raw material for consolidation |
| `semantic` | Long-term facts | Slow (90d) | Distilled beliefs extracted from multiple episodes |
| `document` | Reference shelf | Slow / versioned | Ingested files, chunked and re-indexed on change |
| `procedural` | Muscle memory | Slow (90d) | Learned workflows extracted from repeated patterns |

---

## The Name

Mid-conversation, something triggers a memory. The system surfaces it — not because you queried for it explicitly, but because the current context activated it. A flashback.

That's the experience the system is trying to produce. Not "here are the top 5 relevant documents." Not "here's what you told me to remember." But: the right memory, at the right moment, because the context demanded it.

Human conversations aren't transactions. They're journeys. Flashback is memory that travels with you.

---

## Docs

- [docs/VISION.md](docs/VISION.md) — the dynamic RAG thesis; why current memory is broken and what fixing it looks like
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — full system design: memory hierarchy, five types, decay model, consolidation pipeline, hybrid retrieval, supersede chains, 3D visualization, database schema

---

## Status

**Early development / alpha.** The architecture is fully specified; core infrastructure is being built out. Not production-ready. API surfaces may change.

If the thesis resonates, watch the repo. Contributions and feedback welcome — open an issue.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
