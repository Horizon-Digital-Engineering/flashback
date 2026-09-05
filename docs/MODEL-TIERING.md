# Model tiering — extraction vs distillation

*Status: **exploratory, about to ship**. The trait was always designed for this; the config never followed. This doc captures the why before the wiring lands.*

---

## Thesis

**One model does two jobs of very different shapes. Stop pretending they're the same job.**

Every memory write triggers `extract()` — parse a paragraph into ~6 JSON fields. It runs on the user's critical path and must finish in human-interactive time.

Every consolidation cycle triggers `distill_facts()` — cluster 10-20 episodic memories and synthesize 1-3 semantic facts. It runs as a background worker and gets minutes if it needs them.

Today both calls hit the same model behind the same config. That forces a single uncomfortable compromise: pick a big model and feel the latency on every write, or pick a small model and lose the reasoning depth distillation actually needs.

Tier the models by role.

---

## The trait already knows

```rust
// crates/nlp/src/provider/mod.rs
async fn extract(&self, text: &str, ctx: &ExtractContext) -> Result<Extraction>;
async fn distill_facts(&self, episodes: &[EpisodeRef]) -> Result<DistillResponse>;
```

Two methods. Two task shapes. The interface was split at design time. Only the *implementation* collapsed them by reading one `PROVIDER_REMOTE_MODEL` env var and routing both calls to it.

This isn't a redesign — it's finishing what the trait started.

---

## The two roles

### Extraction — small, fast, deterministic

- **Latency budget**: < 2s. Lives on the write path.
- **Input**: one paragraph + a small JSON schema.
- **Output**: ~150-250 tokens. Strict JSON.
- **Reasoning needed**: minimal. Classify intent from 7 options, lowercase a few entities, pick a topic phrase.
- **Failure mode if undersized**: occasional wrong classification (e.g. `operation: remove` when the speaker meant `add`). Cost is real but bounded.
- **Right size**: 3B-7B. Quantized aggressively (Q4 is fine).

### Distillation — large, slow, judgmental

- **Latency budget**: minutes. Lives in the consolidation worker, runs daily/weekly.
- **Input**: 10-20 episodic memories totaling maybe 5-15 KB of text.
- **Output**: 1-3 distilled facts (~200-500 tokens), each citing source episode IDs.
- **Reasoning needed**: significant. Identify shared claims, resolve contradictions, write declarative present-tense facts, judge which subset of episodes supports each fact.
- **Failure mode if undersized**: facts get sloppy — repeated, contradictory, or citing wrong episodes. Pollutes semantic memory permanently (until manually superseded).
- **Right size**: 14B-32B. Lower quantization tolerated (Q5/Q6 worth the modest speed cost).

These are different jobs. They want different models.

---

## Config shape

Three env vars where there's currently one. All optional. All fall back to `PROVIDER_REMOTE_MODEL` if unset, so single-model setups keep working unchanged.

```sh
# Existing — still works as the default for both roles.
PROVIDER_REMOTE_MODEL=qwen2.5:14b-instruct-q4_K_M

# New — override per role when you want the split.
PROVIDER_REMOTE_EXTRACT_MODEL=qwen2.5:3b-instruct
PROVIDER_REMOTE_DISTILL_MODEL=qwen2.5:14b-instruct-q5_K_M
```

The same pattern repeats for `max_tokens` and `timeout_ms`, since the two roles want different limits:

```sh
PROVIDER_REMOTE_EXTRACT_MAX_TOKENS=256
PROVIDER_REMOTE_EXTRACT_TIMEOUT_MS=10000

PROVIDER_REMOTE_DISTILL_MAX_TOKENS=1024
PROVIDER_REMOTE_DISTILL_TIMEOUT_MS=120000
```

The base URL and API key stay shared — both roles hit the same backend, just with different model strings in the request body.

---

## Worked example on a 16 GB GPU

This is what motivates the doc — concrete numbers from the box we're running on.

**RX 7800 XT, 16 GB VRAM, ROCm:**

| Model | VRAM | Tok/s | Resident? |
|---|---|---|---|
| `qwen2.5:3b-instruct-q4_K_M` | ~2.0 GB | ~150-200 | Yes, warm-loaded |
| `qwen2.5:14b-instruct-q5_K_M` | ~10.5 GB | ~30-40 | Yes, warm-loaded |
| **Total** | **~12.5 GB** | | **3.5 GB headroom** |

Ollama keeps both warm if requests arrive within `OLLAMA_KEEP_ALIVE` (default 5 min). A write that triggers extraction hits the 3B for ~1s. A nightly consolidation pass hits the 14B for ~30s per cluster.

The extraction path becomes interactive. The distillation path keeps its reasoning depth. Same hardware, no compromise.

---

## Interplay with other settings

- **PROVIDER_FALLBACK**: applies per-call, per-role. If extract fails, fall back to heuristic for extract; if distill fails, fall back per the same rule. The two roles don't share fate.
- **Embedded LLM provider** (`PROVIDER=embedded`): tier within mistral.rs the same way — `PROVIDER_EMBEDDED_EXTRACT_MODEL` / `PROVIDER_EMBEDDED_DISTILL_MODEL`. Same shape, different runtime.
- **Heuristic provider**: trivially fast for both; tiering is meaningless. The split lives in `RemoteLlmProvider` and `EmbeddedLlmProvider` only.

---

## Open questions

1. **Should `distill_facts` accept a `mode` parameter** so different modes (per [MODES.md](MODES.md)) can declare different distillation models? Per-mode distillation models is plausible but currently overkill — `mode` isn't built yet.
2. **Should extraction model selection be runtime-overridable per-call** (the MCP tool accepts an optional `model` field), or strictly config-driven? Lean config-driven for now; runtime override is feature creep.
3. **Ollama-specific**: with two models resident, `OLLAMA_NUM_PARALLEL` should probably be `2` so distill and extract calls don't queue behind each other. Worth documenting as the recommended Ollama sidecar config.

---

## What would kill the idea

- Distillation turns out to be just as latency-sensitive as extraction (e.g. user-triggered "consolidate now" becomes a common UX). At that point distill is *also* on the critical path and the split doesn't buy what we thought.
- The "right size" for both jobs converges (a future 5B model that's as good at distillation as today's 14B). At which point you'd pick one model anyway.
- Multi-model resident VRAM becomes a problem (cheaper hardware than RDNA3 ships with). For ≤8 GB cards the user can't warm-keep both; ollama swap-in/swap-out latency erodes the speed win.

None of these feel imminent.

---

## Status

**Built.** `PROVIDER_REMOTE_EXTRACT_MODEL` / `PROVIDER_REMOTE_DISTILL_MODEL` (with per-role max-tokens and timeouts) wire exactly this split; unset, both roles fall back to the single-model config.
