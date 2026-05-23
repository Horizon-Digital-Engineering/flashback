# Embedding model choice

*A use-case-driven guide to picking the right embedding model for Flashback. Read this before you reach for "more dimensions = better."*

---

## TL;DR

| Your use case | Recommended model | Dim |
|---|---|---|
| General conversational memory, English | `sentence-transformers/all-MiniLM-L6-v2` (default) | 384 |
| **Conversation about code / dev work** | **`jinaai/jina-embeddings-v2-base-code`** | **768** |
| Mixed prose + symbol-heavy technical text | `BAAI/bge-base-en-v1.5` | 768 |
| Multilingual or non-English | `intfloat/multilingual-e5-base` | 768 |
| Maximum quality, willing to pay storage + compute | `BAAI/bge-large-en-v1.5` or `Qwen/Qwen3-Embedding-0.6B` | 1024 |
| Air-gapped, smallest possible footprint | `sentence-transformers/all-MiniLM-L6-v2` (quantized) | 384 |

Default is 384/MiniLM because it's the most universally "fine" — but it's the wrong default for code-heavy conversation. **Pick deliberately.**

---

## Two ML systems, easy to confuse

Flashback has **two completely separate ML pieces**, and people often conflate them when the conversation turns to "what model are you using":

### 1. The embedder

- Turns text into a numeric vector (a 384-dim or 768-dim or 1024-dim float array)
- Used on every memory write AND every search query
- Always runs **locally, in-process, no API call** — even when `PROVIDER=remote`
- This is what determines the *dimensions* of your `vector(N)` pgvector column
- Implementation: [`fastembed-rs`](https://crates.io/crates/fastembed), which ships ONNX-converted versions of every model in the table above
- Not an LLM. It's a sentence-transformer.

### 2. The AiProvider

- Pulls **structure** out of text: topic, intent, operation, entities; distills clusters of episodes into semantic facts
- Three implementations: `heuristic` (regex), `remote` (Claude/GPT/OpenRouter), `embedded` (mistral.rs in-process)
- The provider is the "LLM" part of Flashback

These two systems are independent. Switching your AiProvider from heuristic to Claude does **not** change your embedding dimensions. Switching your embedder from MiniLM to BGE does **not** change anything about which LLM Flashback talks to.

The decision on this page is about the embedder. The AiProvider decision is in [deploy/README.md](../deploy/README.md#embedded-llm-runbook).

---

## What dimensions actually buy you

A higher-dimensional embedding can theoretically separate fine-grained semantic differences better — more "axes of meaning" to spread concepts along. In practice it's a story of diminishing returns with real costs:

| Dim  | Bytes/memory | Cosine compare | IVFFlat index @ 100k rows |
|------|--------------|----------------|----------------------------|
| 384  | 1.5 KB       | ~25 µs         | ~600 MB                    |
| 768  | 3.0 KB       | ~50 µs         | ~1.2 GB                    |
| 1024 | 4.0 KB       | ~70 µs         | ~1.6 GB                    |
| 1536 | 6.0 KB       | ~100 µs        | ~2.4 GB                    |

At 1536, every search does ~4x the math of 384 and every memory takes ~4x the disk. **That cost is real and continuous.** The benefit (better separation) is conditional on (a) your text being dense enough to need it and (b) the model being trained well enough to use those extra dimensions meaningfully.

### What dimensions do NOT buy you (the myth)

The "OpenAI 1536 is best because it's biggest" mental model is wrong. The [MTEB benchmark](https://huggingface.co/spaces/mteb/leaderboard) consistently shows:

- 384-dim MiniLM-L6 is within 1-3% of OpenAI's 1536 `text-embedding-3-small` on general short-text retrieval
- 768-dim BGE-base **beats** OpenAI's 1536 model on several retrieval benchmarks despite being 4x smaller
- 384-dim BGE-small often outperforms much larger competitors

**The model architecture and training data matter way more than the output dimension.** OpenAI ships 1536 partly because that's where their model lands, partly because "1536" is a marketable big number.

---

## Use case → model

### Default conversational memory (English, general)

```bash
# This is what ships out of the box.
# fastembed::EmbeddingModel::AllMiniLML6V2 (384 dim)
```

Good for: general "what did we say about X" memory, personal assistants, chat agents that mostly handle natural English questions.

Adequate for: lightly technical conversation, where most of the text is prose with occasional technical nouns ("we switched the deploy target to production").

**Not great for**: dense code symbols, type signatures, framework names treated as semantic units, error traces, mixed code/prose where the code is the meaning.

### Code-heavy technical conversation (the "this is going to be used with code" case)

```toml
# .env or build override
PROVIDER_EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-code
```

```rust
// crates/nlp/src/embed.rs default
model: EmbeddingModel::JinaEmbeddingsV2BaseCode,
```

```sql
-- migrations/005_embedding_768.sql (or whatever the bump is)
ALTER TABLE memories ALTER COLUMN embedding TYPE vector(768);
```

Why: `jina-embeddings-v2-base-code` is trained on Stack Overflow / GitHub / docs in addition to general text. It treats `Arc<Mutex<HashMap<K,V>>>` as a meaningful symbol cluster, not a bag of characters. It handles `useState` and `fastembed` and `pgvector` as concepts. Same training corpus has actual English explanations of code, so "talk about code" works as well as "code about code."

Trade-off: 2x storage, 2x compare time. For ~10,000 memories on a small VPS, total embedding index goes from ~60 MB to ~120 MB. Trivially affordable.

### Mixed prose + symbol-heavy (architecture discussions, design docs)

```rust
model: EmbeddingModel::BGEBaseENV15,  // 768 dim
```

`BAAI/bge-base-en-v1.5` is the "safe choice" general-purpose 768-dim embedder. Tops MTEB for English short-text retrieval, well-tested in production by lots of people, no surprises. Pick this if you don't know whether your data is "code-heavy" or "prose-heavy" — it handles both better than MiniLM.

### Multilingual / non-English

```rust
model: EmbeddingModel::MultilingualE5Base,  // 768 dim, 100+ languages
```

If your conversations include code-switching, non-English sessions, or you're building memory for a multilingual user.

### Maximum quality, willing to pay

```rust
model: EmbeddingModel::BGELargeENV15,  // 1024 dim, slower
// or
model: EmbeddingModel::Qwen3Embedding06B,  // 1024 dim, newer, needs `qwen3` feature
```

These earn the storage + compute cost on dense, jargon-heavy, fine-grained data: legal contracts, medical records, scientific papers, dense code refactor discussions. For most conversational use, this is overkill — BGE-base captures most of the gains.

### Air-gapped / smallest possible footprint

```rust
model: EmbeddingModel::AllMiniLML6V2Q,  // quantized — same accuracy, smaller binary
```

Same 384-dim MiniLM, INT8-quantized. Slightly faster on CPU, ~3x smaller on disk. No quality loss worth caring about.

### When in doubt — OpenAI 1536 for compatibility

```rust
// Requires PROVIDER=remote pointing at OpenAI's /v1/embeddings endpoint;
// not implemented as a default fastembed model.
```

The honest case for 1536-dim OpenAI: you have an existing vector DB seeded with OpenAI embeddings and don't want to re-embed everything. Otherwise, the local fastembed options are usually faster, free, and competitive on quality.

---

## How to switch

Embedding model is a deploy-time choice. Three files to touch:

### 1. Pick the model in `crates/nlp/src/embed.rs`

```rust
// in EmbedderConfig::default()
model: EmbeddingModel::JinaEmbeddingsV2BaseCode,  // or whatever you picked
```

(A future env-var-driven version is on the roadmap — `PROVIDER_EMBEDDING_MODEL=jina-code` — but the source-level swap is one line today.)

### 2. Update the pgvector column dimension

```sql
-- migrations/005_embedding_768.sql (or 1024, depending on your pick)
ALTER TABLE memories ALTER COLUMN embedding TYPE vector(768);
DROP INDEX IF EXISTS memories_embedding_idx;
CREATE INDEX memories_embedding_idx
    ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```

### 3. Backfill existing rows

Either:

```sql
-- Nuke and re-ingest (clean slate)
DELETE FROM memories;
```

Or write a one-shot job (~80 LOC) that re-embeds each existing memory's content. The new vector goes back into the same row. Old retrieval semantics roughly survive because the underlying text is the same.

### 4. Restart the server. First boot will pull the new ONNX model (~80-200 MB depending on choice) into the `FLASHBACK_FASTEMBED_CACHE` directory. Subsequent boots reuse it.

---

## What this guide doesn't cover

- **Reranking**: a small cross-encoder model that re-scores the top-K results from vector search. fastembed-rs supports rerankers; we don't use them yet. Phase 6-ish material.
- **Hybrid sparse + dense**: combining BM25 with vector search at the embedding level (not just the score level). [`prithivida/Splade_PP_en_v1`](https://huggingface.co/prithivida/Splade_PP_en_v1) is a sparse model. Same crate. Same code path.
- **Tuning chunking** for embedding (Phase 5 — currently we embed each memory whole; long memories may need splitting).

---

## The honest meta-take

This is a knob you tune *as you use the system and learn what your data looks like*, not something you over-engineer upfront. Default is 384/MiniLM because it's the most universally fine. If retrieval quality lags on your real data, the first thing to try is bumping to BGE-base (768) — almost certainly an improvement, low cost. If even that doesn't capture your symbol density, jina-code or Qwen3 is the next step.

Don't chase 1536 because OpenAI does. Don't reach for it without a measured reason. **The right answer is "the smallest embedder that gives you acceptable retrieval on YOUR data."**

---

## The bigger frame: this is one knob in a "modes" architecture

This page treats "pick an embedder" as a one-time install-level decision. That's where Flashback ships today.

The exploratory direction we're leaning toward is **modes** — first-class cognitive registers (code / general / journal / research) where each mode declares its own embedder and lives in its own retrieval geometry. "What embedder should I pick" becomes "what embedder should each mode pick," and the table on this page becomes the per-mode picking guide.

If you're reading this doc as "I'm deploying Flashback today, what model" — the table above is your answer.

If you're reading it as "how should the architecture handle multiple use cases long-term" — [docs/MODES.md](MODES.md) is the conversation worth having. The brain-mode metaphor (humans switch register, they don't run parallel brains) is the spine.
