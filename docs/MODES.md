# Modes — exploratory design

*Status: **exploratory**. Not built. Not committed to. A design note capturing where the architecture is leaning.*

*Audience: people deciding whether to commit. If this turns out to be the right call, this doc moves to `ARCHITECTURE.md` as built. If it's wrong, this becomes the postmortem.*

---

## Thesis

**Humans don't run parallel brains. They switch register.**

When you're debugging Rust at 2pm and your partner asks how your day was at 7pm, you don't run two memory queries against two databases. You mode-switch. The same fact — "I solved the borrow-checker thing" — gets retrieved in two completely different framings: the technical reconstruction for a code session, the casual "yeah, productive day" for the evening conversation. Same memory, different *cognitive register*.

The current Flashback design has one register. It picks one embedding model at install time, one extraction style, one retrieval geometry. Works fine for "all my conversations are about one thing" — breaks down the moment you want both code-heavy technical memory AND casual conversational memory in the same system.

The exploratory proposal: **modes as a first-class axis of the memory model.** Every memory has a mode. Modes have their own embedder, their own retrieval geometry, their own extraction prompt. One memory system, multiple cognitive registers.

---

## Why not just run two Flashback installs?

That was the original lazy suggestion. It's wrong because:

1. **The memory belongs to one human** with one continuous life. Splitting it across systems forces the human to remember *where* a memory lives before they can find it. That's worse than no memory at all.
2. **Modes overlap.** "I had a long debug session today and now I'm tired" is one sentence with two modes' worth of content. Two installs can't hold a memory that spans both registers.
3. **Consolidation can't cross installs.** "Over the past month, every time I work late I get short with the kids" — that semantic distillation requires both modes visible to the same worker.

Modes inside one install handle all three.

---

## What a mode is

A named *register* for memory, declared by the user, scoped to that user. Each mode pins:

- **An embedder** — which fastembed model to vectorize memories with (and therefore which vector dimension this mode lives in)
- **Optional extraction prompt overrides** — a code mode might tell the AiProvider "treat type signatures as semantic units"; a journal mode might say "preserve emotional vocabulary"
- **Optional default decay class** — a mode might want all its memories to decay faster or slower than the global default
- **A description** — for the user's own reference; also given to the AiProvider when classifying which mode a turn belongs to

Examples a person might declare:

| Mode | Embedder | Notes |
|---|---|---|
| `code` | `jina-embeddings-v2-base-code` (768d) | symbol-heavy, framework names as concepts |
| `general` | `all-MiniLM-L6-v2` (384d) | default English conversation |
| `journal` | `BAAI/bge-base-en-v1.5` (768d) | emotional / reflective text, slow decay |
| `research` | `BAAI/bge-large-en-v1.5` (1024d) | dense academic text, deeper resolution |
| `family` | `intfloat/multilingual-e5-base` (768d) | multilingual mix |

The user picks the modes they need. Five is typical; one is fine; twenty is probably overcooking it.

---

## Schema (sketch — no migration plumbing, fresh-build assumption)

```sql
CREATE TABLE modes (
    user_id           TEXT NOT NULL,
    name              TEXT NOT NULL,
    embedder          TEXT NOT NULL,          -- e.g. "jina-embeddings-v2-base-code"
    embedding_dim     INT  NOT NULL,
    description       TEXT,
    default_decay     TEXT,                   -- 'fast' | 'medium' | 'slow' | 'none'
    prompt_overrides  JSONB,                  -- per-mode hints for the extraction call
    is_default        BOOLEAN NOT NULL DEFAULT false,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, name)
);

ALTER TABLE memories ADD COLUMN mode TEXT NOT NULL DEFAULT 'general';

-- One nullable column per dimension. A memory writes to exactly one.
ALTER TABLE memories ADD COLUMN embedding_384  vector(384);
ALTER TABLE memories ADD COLUMN embedding_768  vector(768);
ALTER TABLE memories ADD COLUMN embedding_1024 vector(1024);
-- (etc. for any dim we want to support)

-- Per-column indexes, partial so empty columns don't bloat.
CREATE INDEX memories_emb384_idx  ON memories USING ivfflat (embedding_384  vector_cosine_ops) WHERE embedding_384  IS NOT NULL;
CREATE INDEX memories_emb768_idx  ON memories USING ivfflat (embedding_768  vector_cosine_ops) WHERE embedding_768  IS NOT NULL;
CREATE INDEX memories_emb1024_idx ON memories USING ivfflat (embedding_1024 vector_cosine_ops) WHERE embedding_1024 IS NOT NULL;
```

Multiple nullable columns rather than one polymorphic "embedding" column because pgvector wants a typed dimension to build an index. A polymorphic JSONB-of-floats column works but loses the index. Parallel typed columns are uglier in the schema but right for the access pattern.

(Migration is not a concern. There are no users.)

---

## Mode precedence — three layers

When a memory comes in, the system has to decide which mode it belongs to. Three layers, in order:

### 1. Project default (declared, persistent)

When a project is created, it declares a default mode. "Flashback project: default mode = `code`." Anything ingested in that project, with no other signal, lands in `code` mode. Embedded with jina-code, written to `embedding_768`, retrieved against other `code`-mode memories.

This is the 80% case. Most of what a person ingests in a "code project" really is code-mode content.

### 2. Caller override (explicit, per-turn)

A single ingest call can override the project default. The MCP `flashback_remember` tool takes an optional `mode` parameter; if the client passes it, that wins.

Use case: mid-code-session, you tell Flashback "remind me to pick up coffee tomorrow." Caller marks it `mode=general` and the memory lands in the general space alongside other casual memories — not buried with deploy-target chatter.

### 3. AiProvider auto-classification (LLM-derived, fallback)

When the project default exists but the caller didn't override, the AiProvider gets a chance to override it during extraction. The extraction schema gains one field:

```rust
pub struct Extraction {
    pub topic: Option<String>,
    pub intent: Intent,
    pub operation: Option<Operation>,
    pub mode: Option<String>,    // ← new
    pub entities: Vec<String>,
    // ...
}
```

The system prompt grows by one line: "Classify the dominant register from this list: [code | general | journal | …]. If unclear, return null."

Heuristic provider returns None (no auto-classification) → project default wins.
Remote / embedded providers actually read the text and pick. Their answer wins over the project default but loses to an explicit caller override.

This is the escape hatch. Without it, you'd never get the "pick up coffee tomorrow" message into general mode unless the caller explicitly marked it. With it, the LLM notices "this isn't code-talk" and routes correctly.

---

## Retrieval — two paths

### Single-mode (default, fast)

Every search and assemble call carries an implicit or explicit mode. The query is embedded with that mode's embedder; the SQL filters `WHERE mode = $1` and uses `embedding_<dim>` for cosine. Cosine works because both query vector and candidate vectors are in the same space. Same hybrid scoring (cosine + BM25 + recency + project + entity) as today, just mode-scoped.

This is the right behavior for 95% of queries. When you're in code mode, you don't want the deploy-target memory pulled because of a fuzzy match against "I am exhausted from deploying my own emotional stability." Different mode, different vocabulary, different geometry — don't compare.

### Cross-mode (explicit, degraded)

When the caller explicitly asks `modes=[code, general]` or `modes=all`, retrieval can't use cosine across different-dim vectors. Falls back to:

- **BM25 keyword search** (mode-agnostic — runs against `content_tsv`)
- **Entity overlap** (mode-agnostic — runs against `entities[]`)
- **Recency + importance** (mode-agnostic — pure metadata)
- **Topic-string match** if extracted topics exist

You lose semantic-similarity recall but keep keyword-level recall. That's the right trade: cross-mode queries are rare, the user is explicitly bridging, and the loss is visible (a warning in the response).

---

## Consolidation under modes

Weekly distillation clusters episodes by topic. Modes add a hard cluster boundary: **distillation never crosses mode boundaries.** Two episodes about "the deploy target" in code mode get distilled together; an episode in code mode and an episode in journal mode about the same word never do.

Mechanically:
- Cluster within `(user_id, mode)`.
- Distilled `semantic` memory inherits the mode of its sources, embedded with the same model.
- The supersede chain still works (each source episode's `superseded_by` points at the new semantic memory).

The reason this matters: an LLM asked to distill a code-mode and journal-mode pair would write nonsense ("the user feels emotionally invested in their deploy target" — true in a loose sense, useless as a fact). Mode scoping prevents the worst failure mode of cross-domain distillation.

---

## Per-mode embedder choice

(This subsumes the old EMBEDDINGS.md model-choice table.)

When a user declares a mode, they pick its embedder. The defaults below are good starting points; users can override with any fastembed-supported model.

| Mode register | Recommended embedder | Dim | Why |
|---|---|---|---|
| `general` | `all-MiniLM-L6-v2` | 384 | Fast, free, adequate for short English. |
| `code` | `jina-embeddings-v2-base-code` | 768 | Trained on code + docs; treats symbols and framework names as concepts. |
| `journal` / `reflective` | `BAAI/bge-base-en-v1.5` | 768 | Best general-purpose 768d for English prose; good with emotional vocabulary. |
| `research` / `dense` | `BAAI/bge-large-en-v1.5` | 1024 | Earns the storage for dense academic / legal / medical text. |
| `multilingual` | `intfloat/multilingual-e5-base` | 768 | 100+ languages, code-switching tolerant. |
| `code-rust` (specialized) | `jina-embeddings-v2-base-code` | 768 | Same as `code`. A user might declare narrower modes if they want stricter scoping. |
| `code-frontend` (specialized) | same | 768 | (Same — useful for the scoping, not different model.) |
| Air-gapped tiny | `all-MiniLM-L6-v2` (Q4) | 384 | Quantized; same model, smaller binary. |

Two notes:
- A mode can declare a model **purely for scoping purposes** even when it uses the same embedder as another mode. `code-rust` and `code-frontend` can both use `jina-code`; they live in the same vector space but retrieval scopes by mode name. Same vectors are searchable, but `WHERE mode = 'code-rust'` keeps results focused.
- We don't need a column per "specialized" mode — we need a column per *dimension*. Two modes that share an embedder share a column; the `mode` field discriminates.

---

## Admin UI under modes

- The dashboard surfaces "modes" alongside memories / state / tokens.
- A mode-picker filter on the memories list and the map.
- The map view becomes mode-scoped: by default shows one mode at a time (because vector spaces don't combine). A multi-mode view is possible but visually it's "two separate point clouds with no edges between them" — that's the honest visualization of "two spaces."
- A new `/admin/modes` page to create, rename, set defaults, delete modes.

---

## Open questions

Things we'd want to think harder about before committing:

### Q1. Are modes per-user, per-project, or per-account?

Sketched above as per-user. But: should a project be able to declare a mode that the user hasn't created globally? Probably yes — let projects be self-contained.

### Q2. What's the smallest possible "modes" feature?

Could ship Phase 1 of this as: hardcoded list `[code, general]`, no `modes` table, no UI for managing modes. Just two embedding columns, two parallel indexes, project-level mode default. That's maybe 600 LOC instead of 1600. The full user-defined-modes story comes later.

### Q3. What if a user reads from one mode and writes to another?

Search returns code-mode memories, then the user types a follow-up that the LLM extracts as journal-mode. The follow-up gets stored in journal mode, even though it's a direct response to a code-mode memory. Is that wrong? Maybe — there's an argument for "thread coherence" overriding mode classification.

(Probably the right answer is: the AiProvider's classification needs context from recent turns. Pass the recent-turn modes in as part of `ExtractContext`. The LLM is more likely to keep mode-coherence with strong prior context.)

### Q4. Does this collapse with "type"?

Memory already has a `type` (working/episodic/semantic/document/procedural/state_object). Modes are orthogonal — type is "how long it lives + how it decays"; mode is "what cognitive register it belongs to." A working memory in code mode is different from a working memory in journal mode. Both are working. So two axes.

Could be tempting to collapse them — "make `state_object` a mode" — but that loses semantic clarity. Keep separate.

### Q5. What about modes for the embedder ALONE vs modes that change extraction style too?

The proposal above suggests modes can override the extraction prompt. That's powerful but adds complexity. A simpler version: modes only control which embedder gets used. Extraction is one prompt regardless. Worth deciding which scope is right.

### Q6. Auto-mode-discovery?

Long-term, could the system *learn* new modes from clustering patterns? "I notice you have a cluster of memories around X that doesn't fit any existing mode well — want to declare a mode for them?" That's Phase 6+ research. Mentioning it for completeness.

---

## What would kill this idea

Conditions under which we'd abandon the proposal and stay with the single-register design:

1. **Users only ever have one mode.** If empirically nobody declares more than `general`, the complexity isn't earning its keep — collapse back to one column.
2. **Auto-classification keeps misclassifying.** If the LLM puts memories in the wrong mode often enough that users have to manually correct, the cognitive load is worse than just having one bucket.
3. **Cross-mode queries become the common case.** If users mostly want unified retrieval, the mode boundary becomes friction, not feature.
4. **Embedding-model quality keeps converging.** If a future general-purpose model handles code AND journal AND research equally well, the entire reason for per-mode embedders evaporates.

(1) and (2) are the most likely killers. (3) and (4) are slower drift.

---

## Implementation order if we commit

Not a roadmap, just an order that minimizes "did we break it" risk:

1. Add the `mode` column to `memories` (defaulting to `general`). No new behavior yet — just record it.
2. Add the parallel embedding columns (`embedding_384`, `embedding_768`, etc.).
3. Add a hardcoded `modes` map in config (`code` → jina-code, `general` → MiniLM, etc.). Skip the user-defined modes table for now.
4. Route ingest: pick embedder by mode, write to correct column.
5. Route retrieval: scope by mode, pick correct column.
6. Add `mode` to `Extraction` schema; AiProvider extracts it; project-default fallback when null.
7. Admin UI: mode filter on memories list, mode label on detail page, per-mode map view.
8. Consolidation: cluster within mode.
9. (Phase 2 of this work): user-defined modes via the full `modes` table + admin UI for management.

Each step is independently shippable + reversible. The full thing lands incrementally rather than as a Big Bang.

---

## Status

**Not built. Leaning yes.** This doc exists so the design is captured before any code is written. If we commit, this file becomes the spec. If we don't, it stays as an exploration of an idea we considered.

The brain-mode metaphor is the conviction. Everything else is implementation detail.
