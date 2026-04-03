# Episodic Memory System — Architecture

## The Central Metaphor: Memory Is Memory

Human cognition and computer architecture solved the same problem independently: how do you give a system fast access to the things it uses most, while still being able to recall things it hasn't touched in years?

Both landed on the same answer — a memory hierarchy with tiered cost, speed, and capacity tradeoffs.

This system maps that hierarchy onto a persistent memory layer for AI assistants. The tiers are not metaphorical decoration; they drive concrete architectural decisions about where data lives, how it is retrieved, and when it expires.

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1 — ROM / Implicit / Subconscious                        │
│  The LLM's training weights. Language, reasoning, world         │
│  knowledge baked in at training time. We don't store this.      │
│  It is the foundation everything else runs on.                  │
├─────────────────────────────────────────────────────────────────┤
│  TIER 2 — Cache / Instinct / Fast-Access                       │
│  Core memory. Always injected into every prompt.                │
│  User preferences, active project state, communication style.   │
│  Zero retrieval cost. Like knowing your coffee order.           │
├─────────────────────────────────────────────────────────────────┤
│  TIER 3 — RAM / Short-Term / Working                           │
│  Active context with TTL. Current conversation, in-progress     │
│  tasks, things mentioned recently. High relevance, fast decay.  │
│  Consolidates into long-term or fades.                          │
├─────────────────────────────────────────────────────────────────┤
│  TIER 4 — Disk / Long-Term / Explicit                          │
│  The big store. Facts, episodes, project history, decision      │
│  trails, supersede chains. Searchable via semantic search,      │
│  graph traversal, and temporal queries.                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Memory Hierarchy](#memory-hierarchy)
2. [Five Memory Types](#five-memory-types)
3. [Decay Model](#decay-model)
4. [Consolidation Pipeline](#consolidation-pipeline)
5. [Data Models](#data-models)
6. [Ingestion Pipeline](#ingestion-pipeline)
7. [Hybrid Retrieval](#hybrid-retrieval)
8. [Retrieval Modes](#retrieval-modes)
9. [Layered Prompt Assembly](#layered-prompt-assembly)
10. [Automatic Task Extraction](#automatic-task-extraction)
11. [Supersede Model (Not Delete)](#supersede-model-not-delete)
12. [API Design](#api-design)
13. [3D Memory Visualization](#3d-memory-visualization)
14. [Deployment](#deployment)
15. [Prior Art: Mem0, Zep, Letta](#prior-art-mem0-zep-letta)

---

## Memory Hierarchy

### Tier 1 — ROM / Implicit / Subconscious

The LLM's training data is the substrate. It contains language, reasoning ability, broad world knowledge, and implicit norms. We do not store anything at this tier — it is the foundation on which all other tiers operate. A well-designed memory system amplifies what the model already knows; it does not fight the weights.

### Tier 2 — Cache / Instinct / Fast-Access

**Implementation: Core Memory (always-on context)**

Core memory is a small, curated set of facts always injected into the system prompt. No retrieval step, no latency, no threshold to meet. The assistant knows these things the same way you know your own name.

Contents:
- User identity, role, and communication preferences
- Active project names and current state
- Persistent behavioral rules ("always use TypeScript", "never suggest rewrites without being asked")
- Pinned memories that must survive indefinitely

Size constraint: Core memory is deliberately small — typically 500–1500 tokens. Everything here competes with the working context window. Each entry must earn its place.

### Tier 3 — RAM / Short-Term / Working

**Implementation: Working Memory (TTL-bounded, conversation-scoped)**

Working memory holds the active conversation context plus items that were recently relevant but are not important enough to pin. Items have an explicit TTL. When a session ends, the consolidation pipeline decides what to promote to long-term and what to discard.

Characteristics:
- High relevance to the immediate task
- Fast decay — hours to days depending on access frequency
- Automatically extracted candidate tasks (see [Automatic Task Extraction](#automatic-task-extraction))
- Consolidates into episodic or semantic memory on session close

### Tier 4 — Disk / Long-Term / Explicit

**Implementation: Vector store + temporal graph**

The long-term store holds everything that has been consolidated or explicitly saved. It is large, cheap to write, and expensive to query — which is why we invest in retrieval quality rather than raw size.

Sub-components:
- **Vector store** — dense embeddings for semantic search (pgvector or Qdrant)
- **Temporal graph** — time-ordered relationships between memories, entities, supersede chains (Neo4j or Postgres + recursive CTEs)
- **BM25 index** — keyword search over memory text (Elasticsearch or Postgres full-text)
- **Document store** — raw content for ingested files, chunked and indexed

---

## Five Memory Types

### 1. Episodic Memory

Conversation snapshots and event records. The "what happened" layer.

```json
{
  "type": "episodic",
  "timestamp": "2025-11-14T09:23:00Z",
  "session_id": "sess_abc123",
  "summary": "Refactored auth middleware to remove session token storage; discussed compliance requirements.",
  "entities": ["auth_middleware", "session_tokens", "compliance"],
  "embedding": [...],
  "decay_class": "medium"
}
```

Episodic memories are the raw material for consolidation. After enough time passes without re-access, the system may replace a dense episodic record with a shorter semantic distillation, preserving the key fact while shedding the conversational detail.

### 2. Semantic Memory

Distilled facts about the world, the project, and the user. The "what is true" layer.

```json
{
  "type": "semantic",
  "subject": "auth_middleware",
  "predicate": "was_rewritten_because",
  "object": "legal compliance — session token storage requirement",
  "confidence": 0.95,
  "source_episodes": ["ep_4f2a", "ep_7c1b"],
  "decay_class": "slow"
}
```

Semantic memories are generated by the consolidation pipeline, not written directly. They represent the system's "beliefs" — facts extracted from multiple corroborating episodes.

### 3. Working Memory

Active context with explicit TTL. Things the system needs right now.

```json
{
  "type": "working",
  "content": "User is mid-way through refactoring the ingestion pipeline; waiting on schema review.",
  "ttl_hours": 48,
  "expires_at": "2025-11-16T09:23:00Z",
  "priority": "high",
  "session_id": "sess_abc123",
  "decay_class": "fast"
}
```

Working memory items are evaluated at session close. High-priority items with recent access are promoted to episodic. Everything else expires.

### 4. Document Memory

Ingested file content, chunked and indexed. The "reference material" layer.

```json
{
  "type": "document",
  "source_path": "docs/ARCHITECTURE.md",
  "chunk_index": 3,
  "chunk_text": "...",
  "embedding": [...],
  "ingested_at": "2025-11-10T14:00:00Z",
  "content_hash": "sha256:abc...",
  "decay_class": "slow"
}
```

Document memories are re-ingested when the source file changes (content hash mismatch). They do not decay on their own — they are superseded by newer versions of the document.

### 5. Procedural Memory

Learned workflows and behavioral patterns. The "how to" layer.

```json
{
  "type": "procedural",
  "name": "deploy_sequence",
  "trigger": "when user asks to deploy",
  "steps": [
    "run tests",
    "build Docker image",
    "push to registry",
    "apply Helm chart"
  ],
  "learned_from": ["ep_9d3c", "ep_2a7f"],
  "decay_class": "slow"
}
```

Procedural memories are extracted when the system observes repeated patterns across episodes. They feed back into the prompt as behavioral context — the model learns the team's rituals without being re-instructed every session.

---

## Decay Model

Not all memories age at the same rate. The decay model assigns each memory a `decay_class` that determines how quickly its retrieval score degrades over time.

| Decay Class | Half-Life   | Example Types                          |
|-------------|-------------|----------------------------------------|
| `none`      | Never       | Pinned core memories                  |
| `slow`      | ~90 days    | Semantic facts, procedural, documents |
| `medium`    | ~14 days    | Episodic memories                     |
| `fast`      | ~48 hours   | Working memory, in-session context    |

Decay affects retrieval scoring, not physical deletion. A decayed memory is not removed — it simply scores lower in retrieval until it either gets re-accessed (resetting its decay clock) or is explicitly superseded.

Decay function used in scoring:

```
decay_factor = exp(-λ * days_since_access)

where λ = ln(2) / half_life_days
```

Pinned items have `λ = 0` — they never decay.

---

## Consolidation Pipeline

Consolidation is the process by which active context and episodic memories are distilled into longer-lived semantic and procedural knowledge. It runs on a scheduled basis.

### Schedule

| Cadence   | Scope                 | Action                                                           |
|-----------|-----------------------|------------------------------------------------------------------|
| Daily     | Working memory        | Promote high-priority items to episodic; expire the rest        |
| Weekly    | Episodic (7–30 days)  | Extract semantic facts; merge near-duplicate episodes            |
| Monthly   | Project-level         | Generate strategic summaries; consolidate procedural patterns   |

### Working Memory Consolidation (Daily)

```
for each expired working_memory item:
  if item.priority >= 'medium' AND item.access_count > 0:
    create episodic record from item
  else:
    mark as expired (soft delete)
```

### Episodic Consolidation (Weekly)

```
for each episodic memory older than 7 days:
  cluster similar episodes by entity overlap + embedding similarity
  for each cluster:
    if cluster.size >= 2:
      extract semantic facts via LLM summarization
      create/update semantic records with source_episodes references
      reduce episodic records to compressed form (keep timestamp, entities, short summary)
```

### Strategic Consolidation (Monthly)

```
for each project:
  collect semantic memories from past 30 days
  generate project-level summary: decisions made, patterns observed, open questions
  create procedural memories for any repeated multi-step workflows
  flag semantic memories with low confidence for human review
```

---

## Data Models

### Memory Record (canonical schema)

```typescript
interface MemoryRecord {
  id: string;                    // UUID
  type: MemoryType;              // episodic | semantic | working | document | procedural
  content: string;               // human-readable text
  embedding: number[];           // dense vector (1536-dim or model-specific)

  // Temporal
  created_at: Date;
  updated_at: Date;
  last_accessed_at: Date;
  expires_at?: Date;             // working memory only

  // Scoring
  importance: number;            // 0.0–1.0, set at write time + updated on access
  access_count: number;
  decay_class: DecayClass;       // none | slow | medium | fast

  // Context
  project_id?: string;
  session_id?: string;
  user_id: string;
  entities: string[];            // named entities extracted at write time

  // Supersede chain
  superseded_by?: string;        // ID of the memory that replaces this one
  supersedes?: string;           // ID of the memory this one replaced

  // Document-specific
  source_path?: string;
  chunk_index?: number;
  content_hash?: string;
}

type MemoryType = 'episodic' | 'semantic' | 'working' | 'document' | 'procedural';
type DecayClass = 'none' | 'slow' | 'medium' | 'fast';
```

### Entity Node (graph schema)

```typescript
interface EntityNode {
  id: string;
  name: string;
  type: string;                  // person | project | file | concept | decision
  first_seen: Date;
  last_seen: Date;
  memory_ids: string[];          // memories that reference this entity
}

interface EntityRelationship {
  source_id: string;
  target_id: string;
  relation: string;              // depends_on | replaced_by | authored_by | etc.
  since: Date;
  until?: Date;                  // null = still active
}
```

---

## Ingestion Pipeline

Files and external content enter the system through a structured ingestion pipeline that handles chunking, deduplication, and entity extraction.

```
Raw input (file, URL, paste)
        │
        ▼
┌───────────────┐
│  Hash check   │  ── if unchanged: skip
└───────────────┘
        │
        ▼
┌───────────────┐
│  Chunking     │  ── semantic chunking (split at paragraph/section boundaries)
└───────────────┘     target: 256–512 tokens per chunk with 10% overlap
        │
        ▼
┌───────────────┐
│  Embedding    │  ── embed each chunk
└───────────────┘
        │
        ▼
┌───────────────┐
│  NER + link   │  ── extract entities; link to existing entity nodes
└───────────────┘
        │
        ▼
┌───────────────┐
│  Supersede    │  ── if prior version exists: supersede old chunks, write new
└───────────────┘
        │
        ▼
┌───────────────┐
│  BM25 index   │  ── write to keyword search index
└───────────────┘
```

### Chunking Strategy

Semantic chunking is preferred over fixed-size chunking. The pipeline respects document structure: headings, code blocks, and list groups are not split mid-element. A 10% overlap window is used between adjacent chunks to prevent retrieval from missing context at boundaries.

### Deduplication

Content hash comparison runs before chunking. If the hash matches an existing document memory, ingestion is skipped. If the hash differs, the new version supersedes the old: existing chunks get `superseded_by` set to the new chunk IDs, and new chunks are written fresh.

---

## Hybrid Retrieval

Retrieval uses a weighted combination of signals rather than pure vector similarity. This prevents "semantic drift" where relevant but un-embedded keywords are missed, and ensures recency and project context boost the right memories.

### Scoring Formula

```
final_score(m) =
    w_sem  * semantic_similarity(query_embedding, m.embedding)    // cosine similarity
  + w_kw   * bm25_score(query_text, m.content)                   // keyword match
  + w_rec  * recency_score(m.last_accessed_at)                   // exp decay
  + w_imp  * m.importance                                         // explicit importance
  + w_proj * project_match(active_project, m.project_id)         // 1.0 or 0.0
  + w_ent  * entity_overlap(query_entities, m.entities)          // Jaccard similarity
  + w_task * active_task_bonus(m, active_tasks)                  // 0.5 if referenced

Default weights:
  w_sem  = 0.35
  w_kw   = 0.20
  w_rec  = 0.15
  w_imp  = 0.10
  w_proj = 0.10
  w_ent  = 0.05
  w_task = 0.05
```

Weights are configurable and can be tuned per deployment or per retrieval mode. The active task bonus rewards memories that are directly referenced in ongoing tasks — ensuring continuity across sessions.

### Pre-filtering

Before scoring, a pre-filter step reduces the candidate set:

1. **Type filter** — exclude memory types not relevant to the current retrieval mode
2. **Decay filter** — exclude memories with `decay_factor < 0.05` and `access_count == 0`
3. **Project filter** — optionally restrict to `project_id == active_project`
4. **ANN pre-retrieval** — fetch top-200 candidates by approximate nearest neighbor before applying full scoring

### Re-ranking

After scoring, results are re-ranked with a diversity penalty: if two memories have high entity overlap with each other (Jaccard > 0.8), the lower-scoring one is pushed down. This prevents redundant memories from dominating the top-k results.

---

## Retrieval Modes

The system has two distinct retrieval modes that differ in what they optimize for.

### Answer Mode (Classic RAG)

Used when the user asks a question or requests information. Optimizes for relevance to the query.

```
Input:  user query
Goal:   retrieve memories most likely to contain the answer
Output: top-k memories injected into context as supporting evidence

Emphasis:
  - semantic_similarity weight boosted (0.45)
  - bm25_score weight boosted (0.25)
  - recency weight reduced (0.10)
  - active_task_bonus disabled
```

Answer mode treats the memory store as a read-only knowledge base. Retrieved memories are injected as factual context; the model synthesizes the answer.

### Manager Mode (State Engine)

Used at session start, after long gaps, or when the system needs to reconstruct "what is going on." Optimizes for situational awareness.

```
Input:  active project + user identity + recent session IDs
Goal:   reconstruct the current state of work
Output: working memory snapshot + pending tasks + recent decisions

Emphasis:
  - active_task_bonus enabled and boosted (0.15)
  - project_match weight boosted (0.20)
  - entity_overlap weight boosted (0.15)
  - recency weight boosted (0.25)
  - semantic_similarity weight reduced (0.15)
```

Manager mode treats the memory store as a state machine. It asks: "What are the open threads?" and "What has changed since we last talked?" rather than "What is true about X?"

---

## Layered Prompt Assembly

At inference time, context is assembled in five layers, injected in order from most stable to most ephemeral.

```
┌────────────────────────────────────────────┐  ← injected first (most stable)
│  Layer 1: Procedural                       │
│  Learned behavioral patterns, team         │
│  rituals, deployment sequences.            │
│  Source: procedural memories               │
├────────────────────────────────────────────┤
│  Layer 2: Active Project Context           │
│  Current project state, active goals,      │
│  pinned core memory items.                 │
│  Source: core memory + project semantic    │
├────────────────────────────────────────────┤
│  Layer 3: Top Retrieved Memories           │
│  High-scoring episodic + semantic items    │
│  from hybrid retrieval (answer or          │
│  manager mode depending on query).         │
│  Source: long-term store (Tier 4)          │
├────────────────────────────────────────────┤
│  Layer 4: Top Document Chunks              │
│  Relevant sections from ingested files.    │
│  Injected after memories so document       │
│  content grounds the retrieved facts.      │
│  Source: document memories                 │
├────────────────────────────────────────────┤
│  Layer 5: Recent Conversation              │
│  Working memory items + last N turns of    │
│  the current session.                      │
│  Source: working memory (Tier 3)           │
└────────────────────────────────────────────┘  ← injected last (most ephemeral)
         │
         ▼
    [User message]
```

Each layer has a token budget. If any layer exceeds its budget, items are ranked by their retrieval score and truncated from the bottom up. Layer 5 is protected — recent conversation is never truncated below 3 turns.

| Layer | Default Budget |
|-------|---------------|
| 1 — Procedural            | 300 tokens  |
| 2 — Active Project        | 600 tokens  |
| 3 — Top Memories          | 1200 tokens |
| 4 — Document Chunks       | 800 tokens  |
| 5 — Recent Conversation   | 1500 tokens |
| **Total context overhead**| **4400 tokens** |

---

## Automatic Task Extraction

The system identifies candidate tasks from conversation text without requiring explicit task-creation commands.

### Extraction Triggers

The extractor runs on each assistant turn and scans the conversation for task-indicating patterns:

```
Patterns (with confidence score):
  "I need to ..."             → 0.80
  "I should ..."              → 0.65
  "We need to ..."            → 0.75
  "TODO: ..."                 → 0.90
  "Don't forget to ..."       → 0.85
  "Next step is ..."          → 0.70
  "Follow up on ..."          → 0.72
  "Before we can X, we need Y"→ 0.78 (extracts Y as blocking task)
```

### Confidence Scoring

Each candidate task is scored along three dimensions:

1. **Signal strength** — how strong is the linguistic indicator? (0.0–1.0)
2. **Entity density** — does the task reference known project entities? (boosted if yes)
3. **Recency** — was this mentioned more than once in the last N turns? (boosted if yes)

```
task_confidence = signal_strength * 0.6 + entity_density * 0.25 + recency_boost * 0.15
```

Tasks above `confidence >= 0.70` are automatically written to working memory as candidate tasks with `priority = 'medium'`. Tasks above `0.85` are written with `priority = 'high'`.

### Task Lifecycle

```
Candidate (confidence >= 0.70)
        │
        ├─── confirmed by user → promoted to tracked task
        │
        ├─── session ends without confirmation → consolidation decides
        │         if referenced again → episodic memory
        │         else → expires with working memory TTL
        │
        └─── user explicitly dismisses → soft-deleted
```

---

## Supersede Model (Not Delete)

Memory records are never hard-deleted. Instead, they are superseded: the old record remains in the store with its `superseded_by` field set, and the new record carries the `supersedes` pointer.

This design has three properties:

1. **Auditability** — you can always reconstruct the history of a belief. "The deployment target was staging, then changed to production, then reverted" is a chain, not an overwrite.

2. **Temporal queries** — queries scoped to a past date can ignore superseding records and retrieve what was true at that time.

3. **Conflict detection** — if two memories assert contradictory facts about the same entity, the system can detect the conflict rather than silently overwriting one.

### Supersede Chain Example

```
[ep_001] "Deploy target: staging"
    └─ superseded_by: ep_002

[ep_002] "Deploy target: production (changed for release)"
    ├─ supersedes: ep_001
    └─ superseded_by: ep_003

[ep_003] "Deploy target: staging (reverted after incident)"
    └─ supersedes: ep_002
```

A query for "current deploy target" returns `ep_003` (the terminal node). A query for "deploy target on [date between ep_001 and ep_002]" walks the chain and returns `ep_001`.

### Graph Traversal for Lineage

The temporal graph stores supersede chains as directed edges. Lineage queries use recursive graph traversal:

```cypher
MATCH path = (m:Memory {id: $id})-[:SUPERSEDED_BY*]->(latest:Memory)
WHERE NOT (latest)-[:SUPERSEDED_BY]->()
RETURN latest
```

---

## API Design

### REST Endpoints

```
POST   /memory                    Create a memory record
GET    /memory/:id                Retrieve by ID
PUT    /memory/:id/supersede      Supersede with new content
DELETE /memory/:id                Soft-delete (sets expired_at)

POST   /memory/search             Hybrid retrieval query
POST   /memory/ingest             Ingest a document or URL
POST   /memory/consolidate        Trigger manual consolidation

GET    /memory/core               Get core memory (always-on context)
PUT    /memory/core/:id           Update a core memory item
POST   /memory/core               Pin a new core memory item

GET    /tasks                     List active candidate tasks
PUT    /tasks/:id/confirm         Promote candidate to tracked task
DELETE /tasks/:id                 Dismiss a candidate task

GET    /lineage/:entity_id        Walk supersede chain for an entity
GET    /graph/entities            List entity nodes
GET    /graph/relationships       List entity relationships
```

### Search Request Schema

```typescript
interface SearchRequest {
  query: string;
  mode: 'answer' | 'manager';
  top_k?: number;                // default: 10
  project_id?: string;
  memory_types?: MemoryType[];   // filter by type
  since?: Date;                  // temporal lower bound
  until?: Date;                  // temporal upper bound
  weight_overrides?: Partial<RetrievalWeights>;
}

interface SearchResult {
  memories: MemoryRecord[];
  scores: number[];
  retrieval_metadata: {
    semantic_candidates: number;
    bm25_candidates: number;
    post_filter_count: number;
    total_latency_ms: number;
  };
}
```

### Context Assembly Endpoint

```
POST /context/assemble
```

Returns the fully assembled layered prompt context for a given session, broken down by layer. Used by the assistant integration layer to build the system prompt.

```typescript
interface ContextAssemblyRequest {
  session_id: string;
  user_id: string;
  project_id?: string;
  query?: string;               // if present, uses answer mode for Layer 3
  token_budget?: number;        // override total budget
}

interface ContextAssemblyResponse {
  layers: {
    procedural: string;
    project_context: string;
    memories: string;
    document_chunks: string;
    recent_conversation: string;
  };
  token_counts: Record<string, number>;
  total_tokens: number;
}
```

---

## 3D Memory Visualization

The system includes an optional visualization frontend that renders the memory store as an interactive 3D graph. This serves both as a debugging tool and as an intuitive way to explore memory structure.

### Concept: Memory Map

Each memory is a point in 3D space, positioned by dimensionality reduction of its embedding (UMAP from 1536D to 3D). Points are colored by memory type and connected by edges representing relationships and supersede chains.

```
Color coding:
  episodic    → blue
  semantic    → green
  working     → yellow (pulsing, to indicate transience)
  document    → grey
  procedural  → purple

Edge types:
  supersedes     → red arrow
  entity_overlap → thin grey line (opacity = Jaccard similarity)
  same_session   → thin blue line
  temporal_seq   → thin green line (chronological order)
```

### Implementation

Built with Three.js on the frontend, served as a static SPA from the API server.

**Force-directed layout**: The 3D positions from UMAP are used as initial positions. A force simulation (based on Three.js + d3-force-3d) adds mild spring forces along edges so related memories cluster together dynamically.

**Interaction**:
- Click a node → expand detail panel showing full memory content, scores, decay status
- Click an edge → show relationship type and strength
- Hover → show memory summary tooltip
- Time scrubber → animate the graph through time, showing memories appearing, decaying, and being superseded
- Filter panel → toggle memory types, projects, decay classes

**Data API for visualization**:

```
GET /viz/graph?project_id=...&since=...&until=...
```

Returns nodes + edges in a format ready for Three.js scene construction:

```typescript
interface VizGraph {
  nodes: Array<{
    id: string;
    type: MemoryType;
    position: [number, number, number];  // UMAP coordinates
    importance: number;                  // controls node size
    decay_factor: number;                // controls opacity
    label: string;                       // short summary
  }>;
  edges: Array<{
    source: string;
    target: string;
    relation: 'supersedes' | 'entity_overlap' | 'same_session' | 'temporal_seq';
    weight: number;
  }>;
}
```

**Performance**: The UMAP projection is pre-computed and stored alongside each memory's embedding. It is recomputed nightly for all memories (or incrementally as new memories are added). The frontend requests only the visible subgraph (based on current filters and viewport), not the full store.

---

## Deployment

### Components

```
┌─────────────────────────────────────────────────────┐
│  API Server (FastAPI or Express)                    │
│  - Memory CRUD                                      │
│  - Hybrid retrieval                                 │
│  - Context assembly                                 │
│  - Ingestion pipeline                               │
│  - Visualization API                                │
├─────────────────────────────────────────────────────┤
│  PostgreSQL + pgvector                              │
│  - Primary memory store                            │
│  - Vector similarity search                         │
│  - BM25 via pg_trgm or tsvector                    │
│  - Graph via adjacency table + recursive CTEs       │
├─────────────────────────────────────────────────────┤
│  Redis                                              │
│  - Working memory TTL store                         │
│  - Session cache                                    │
│  - Core memory hot cache                            │
├─────────────────────────────────────────────────────┤
│  Scheduler (APScheduler / cron)                     │
│  - Daily working memory consolidation               │
│  - Weekly episodic consolidation                    │
│  - Monthly strategic summaries                      │
│  - Nightly UMAP recompute                           │
├─────────────────────────────────────────────────────┤
│  Embedding Service                                  │
│  - Batch embedding for ingestion                    │
│  - On-demand embedding for queries                  │
│  - Model: configurable (OpenAI, local, etc.)        │
└─────────────────────────────────────────────────────┘
```

### Database Schema (PostgreSQL)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE memories (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  type          TEXT NOT NULL,
  content       TEXT NOT NULL,
  embedding     vector(1536),
  importance    FLOAT DEFAULT 0.5,
  access_count  INT DEFAULT 0,
  decay_class   TEXT DEFAULT 'medium',
  project_id    UUID,
  session_id    UUID,
  user_id       UUID NOT NULL,
  entities      TEXT[],
  superseded_by UUID REFERENCES memories(id),
  supersedes    UUID REFERENCES memories(id),
  source_path   TEXT,
  chunk_index   INT,
  content_hash  TEXT,
  expires_at    TIMESTAMPTZ,
  created_at    TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now(),
  last_accessed_at TIMESTAMPTZ DEFAULT now(),
  viz_position  FLOAT[3]   -- UMAP coordinates for visualization
);

CREATE INDEX ON memories USING ivfflat (embedding vector_cosine_ops)
  WITH (lists = 100);

CREATE INDEX ON memories USING GIN (to_tsvector('english', content));
CREATE INDEX ON memories (user_id, project_id, type);
CREATE INDEX ON memories (expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX ON memories (superseded_by) WHERE superseded_by IS NOT NULL;
```

### Docker Compose (Development)

```yaml
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      DATABASE_URL: postgresql://memory:memory@db:5432/memory
      REDIS_URL: redis://redis:6379
      EMBEDDING_MODEL: text-embedding-3-small
    depends_on: [db, redis]

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: memory
      POSTGRES_USER: memory
      POSTGRES_PASSWORD: memory
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redisdata:/data

  scheduler:
    build: .
    command: python -m scheduler
    environment:
      DATABASE_URL: postgresql://memory:memory@db:5432/memory
    depends_on: [db, redis]

volumes:
  pgdata:
  redisdata:
```

### Scaling Considerations

- **Embedding throughput**: batch embedding jobs run asynchronously via a task queue (Celery + Redis). Ingestion does not block the request path.
- **Vector index**: pgvector's IVFFlat index requires a `VACUUM ANALYZE` after bulk inserts. For large stores (>1M vectors), consider migrating to Qdrant as a dedicated vector backend.
- **Read replicas**: the retrieval path is read-heavy. Route `POST /memory/search` and `POST /context/assemble` to read replicas.
- **Consolidation jobs**: run on a separate worker process to avoid starving the API under heavy consolidation load.

---

## Prior Art: Mem0, Zep, Letta

Understanding what existing tools do — and what they don't — clarifies the design space this system occupies.

### Mem0

Mem0 (mem0.ai) provides a managed memory layer as a hosted API. It extracts facts from conversation and stores them as key-value pairs, returning relevant facts at query time via semantic search.

**What it does well**: low integration friction, managed infrastructure, reasonable retrieval for straightforward factual memory.

**What it lacks**:
- No temporal graph — no way to express "this was true until X, then changed to Y"
- No decay model — all memories are treated as equally current
- No supersede chain — updates overwrite
- No consolidation pipeline — no promotion from working to long-term
- No retrieval modes — no distinction between answering a question and reconstructing state
- Hosted-only — cannot self-host with full control over the data

### Zep

Zep focuses on long-term memory for conversational AI, with a strong emphasis on entity extraction and graph-based storage. It extracts entities and relationships from conversations and maintains a knowledge graph.

**What it does well**: entity and relationship extraction, graph storage, session management, self-hostable.

**What it lacks**:
- Decay model is limited — no TTL-aware working memory tier
- No five-type memory taxonomy (episodic/semantic/working/document/procedural)
- No layered prompt assembly specification
- No consolidation pipeline with scheduled rollup
- Visualization is minimal
- Scoring formula is not exposed — retrieval weighting is opaque

### Letta (formerly MemGPT)

Letta takes a different approach: it gives the LLM explicit tools to read and write its own memory, treating memory management as an agentic capability rather than a retrieval problem.

**What it does well**: fine-grained control, the model can explicitly decide what to remember and how, recursive self-editing.

**What it lacks**:
- Latency — every memory operation is a tool call, adding round-trips
- Cost — memory management consumes model tokens
- Fragility — memory quality depends on the model's in-context decisions, which can drift
- No scheduled consolidation — no autonomous maintenance
- No hybrid retrieval — if the model doesn't decide to look something up, it doesn't get looked up

### This System's Position

This system sits between Zep and Letta on the autonomy spectrum. It does not ask the model to manage its own memory (Letta's approach), nor does it treat memory as a pure retrieval index (Mem0). Instead:

- Memory management is automated but structured (consolidation pipeline, decay, supersede chain)
- Retrieval is transparent (explicit weighted formula, configurable)
- The model is a consumer of assembled context, not a manager of the raw store
- The system is self-hostable and schema-open

The goal is a memory layer that feels invisible to the model — like the model simply "knows" things — while remaining fully auditable and controllable to the operator.
