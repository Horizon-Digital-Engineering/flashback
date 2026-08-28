//! Playground — watch the dynamic-RAG loop happen, one turn at a time.
//!
//! A host like ritsu is a black box from here: turns go in, context comes back,
//! and whether the memory was any GOOD is invisible. This page stands in for a
//! host and shows the whole round trip: what the query retrieved, the exact
//! prompt that would be sent, what the model said, and what got written back.
//!
//! It deliberately calls the SAME seams a real host does — `assemble_inner` for
//! retrieval and `ingest_record` for the write — because a playground with its
//! own private path proves nothing about the real pipeline.
//!
//! The model is optional — it is only the probe; the memory system is what is
//! under test. Retrieval and prompt assembly work with nothing configured, and
//! any OpenAI-compatible endpoint (LM Studio, Ollama, LiteLLM, OpenRouter)
//! provides the completion.
//!
//! The turn endpoint streams SSE in three phases, matching where the time
//! actually goes: `trace` lands ~instantly (retrieval is milliseconds), `delta`
//! events trickle for as long as a local model takes to talk, `done` carries
//! the stats. The diagnostics are readable before the model finishes — which is
//! the point, since they're the product here.

use std::convert::Infallible;
use std::time::Duration;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};
use uuid::Uuid;

use crate::{
    error::{AppError, AppResult},
    routes::records::{
        assemble_inner, ingest_record, AssembleRequest, IngestRecordRequest, RawRecordRow,
    },
    settings::validate_base_url,
    AppState,
};

use crate::auth::AuthUser;
use sqlx::PgPool;

/// Source tags for playground writes. They are ordinary `conversation` records
/// running the ordinary pipeline — but scoped to the sandbox project, so test
/// chatter promotes, clusters and distills in its own bucket and never
/// infects the real store. Same code path, separate data space.
const SOURCE_USER: &str = "playground:user";
const SOURCE_ASSISTANT: &str = "playground:assistant";

const DEFAULT_LIMIT: i64 = 12;
const LLM_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Deserialize)]
pub struct LlmSettings {
    /// OpenAI-compatible base, e.g. `http://127.0.0.1:11434/v1`.
    pub base_url: String,
    pub model: String,
    #[serde(default)]
    pub api_key: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct TurnRequest {
    pub message: String,
    /// The conversation this turn belongs to. Episodes form per container, so
    /// keeping one id across turns is what makes them cohere into an episode.
    pub container_id: String,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub limit: Option<i64>,
    /// Credential for the configured endpoint, held browser-side (see the 011
    /// migration for why it isn't stored). Local model servers ignore it.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Also retrieve from the REAL store. Off by default: the sandbox is a
    /// self-contained bench whose contents you control; checking the box on
    /// the page widens reads to real memories. Writes stay sandboxed always.
    #[serde(default)]
    pub include_real: bool,
}

/// The retrieval scope a turn runs with: sandbox-only by default, sandbox plus
/// the real store when the operator asks. `(project_id, include_sandbox)` in
/// `AssembleRequest` terms.
fn turn_scope(include_real: bool) -> (Option<String>, bool) {
    if include_real {
        (None, true)
    } else {
        (
            Some(crate::routes::records::SANDBOX_PROJECT.to_string()),
            false,
        )
    }
}

/// The default framing. Deliberately plain: it tells the model the memories are
/// available without insisting they're relevant, so a bad retrieval shows up as
/// the model ignoring them rather than being forced to use them.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant with persistent memory. \
Answer in a few sentences unless asked for detail — most of the wait a user \
experiences is output length.";

#[derive(Debug, Serialize)]
pub struct RetrievedItem {
    pub id: Uuid,
    pub r#type: String,
    pub source: String,
    pub content: String,
    pub event_time: String,
    pub container_id: Option<String>,
    /// True when the record lives in the sandbox scope — the diagnostics tag
    /// each memory so a mixed retrieval says which store it came from.
    pub sandbox: bool,
}

impl From<&RawRecordRow> for RetrievedItem {
    fn from(r: &RawRecordRow) -> Self {
        Self {
            id: r.id,
            r#type: r.r#type.clone(),
            source: r.source.clone(),
            content: r.content.clone(),
            event_time: r.event_time.to_rfc3339(),
            container_id: r.container_id.clone(),
            sandbox: r.project_id.as_deref() == Some(crate::routes::records::SANDBOX_PROJECT),
        }
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct PromptMessage {
    pub role: String,
    pub content: String,
}

/// What the model reported about its own call. Absent when no model ran.
#[derive(Debug, Serialize, Default)]
pub struct LlmStats {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    pub latency_ms: u64,
}

/// Phase 1 of the SSE stream: everything known before the model runs. Sent the
/// moment retrieval finishes, so the memory diagnostics are on screen while a
/// slow local model is still talking.
#[derive(Debug, Serialize)]
pub struct TraceEvent {
    /// What retrieval returned for this query, in rank order.
    pub retrieved: Vec<RetrievedItem>,
    /// The exact messages a host would send. This IS the context going in.
    pub prompt: Vec<PromptMessage>,
    /// The user turn's raw record id — written before the model is called, so
    /// a dead model never loses what you typed.
    pub written: Vec<Uuid>,
    pub degraded: bool,
    pub warning: Option<String>,
    /// The model that will answer, and whether it was inherited from the
    /// system provider rather than set in the sandbox.
    pub model: Option<String>,
    pub model_inherited: bool,
    /// Set when no model will run (not configured); the stream ends after this.
    pub llm_error: Option<String>,
}

/// The model a playground turn runs with: the sandbox override when both of
/// its fields are set, otherwise the SYSTEM provider — the playground's job is
/// to exercise the configured pipeline, so the configured distill-role model
/// is the default probe. Returns `(settings, inherited)`.
pub(crate) async fn resolve_llm_settings(
    pool: &PgPool,
    env: &crate::config::ProviderConfig,
    own: &Settings,
    browser_key: Option<String>,
) -> (Option<LlmSettings>, bool) {
    if let (Some(base), Some(model)) = (own.base_url.as_deref(), own.model.as_deref()) {
        return (
            Some(LlmSettings {
                base_url: base.to_string(),
                model: model.to_string(),
                api_key: browser_key,
            }),
            false,
        );
    }
    let sys = crate::settings::resolve_from_db(pool, env).await;
    if sys.kind == crate::config::ProviderKind::Remote {
        if let Some(base) = sys.remote.api_base.clone() {
            // The server-held key is the same one the pipeline itself sends to
            // this endpoint; a browser-supplied key still wins for overrides.
            let env_key = (!sys.remote.api_key.is_empty()).then(|| sys.remote.api_key.clone());
            return (
                Some(LlmSettings {
                    base_url: base,
                    model: sys.remote.distill_model.clone(),
                    api_key: browser_key.or(env_key),
                }),
                true,
            );
        }
    }
    (None, false)
}

/// Phase 3: the wrap-up after the model finishes (or fails).
#[derive(Debug, Serialize)]
pub struct DoneEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<LlmStats>,
    pub llm_error: Option<String>,
    /// All raw ids for the turn (user, then assistant when one was written).
    pub written: Vec<Uuid>,
}

/// Render retrieved records as the context block a host injects. Kept plain and
/// legible: what you read here is byte-for-byte what the model gets.
fn render_context_block(items: &[RetrievedItem]) -> String {
    let mut s = String::from(
        "Relevant memories retrieved for this turn. Use them only if they bear \
         on what was asked; say so plainly if they don't.\n\n",
    );
    for (i, it) in items.iter().enumerate() {
        s.push_str(&format!(
            "[{}] ({}, {}, {})\n{}\n\n",
            i + 1,
            it.r#type,
            it.source,
            it.event_time,
            it.content.trim()
        ));
    }
    s
}

/// Stream a chat completion from any OpenAI-compatible endpoint, forwarding
/// content deltas as they arrive. Returns the full accumulated text plus stats.
/// Usage numbers come from the final chunk when the server sends them
/// (`stream_options.include_usage`); absent is fine — they're display-only.
async fn call_llm_stream(
    cfg: &LlmSettings,
    messages: &[PromptMessage],
    tx: &tokio::sync::mpsc::Sender<Event>,
) -> Result<(String, LlmStats), String> {
    // Re-checked here, not just on save: a row written before this guard
    // existed (or by any future path) must not become an outbound request.
    validate_base_url(&cfg.base_url)?;
    let started = std::time::Instant::now();
    let url = format!("{}/chat/completions", cfg.base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(LLM_TIMEOUT)
        .build()
        .map_err(|e| e.to_string())?;

    let mut req = client.post(&url).json(&json!({
        "model": cfg.model,
        "messages": messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect::<Vec<_>>(),
        "stream": true,
        "stream_options": {"include_usage": true},
    }));
    if let Some(key) = cfg.api_key.as_deref().filter(|k| !k.trim().is_empty()) {
        req = req.bearer_auth(key);
    }

    let mut res = req.send().await.map_err(|e| e.to_string())?;
    if !res.status().is_success() {
        let status = res.status();
        let body = res.text().await.unwrap_or_default();
        return Err(format!(
            "{status}: {}",
            body.chars().take(400).collect::<String>()
        ));
    }

    let mut text = String::new();
    let mut stats = LlmStats::default();
    let mut buf = String::new();
    // SSE from the model server: `data: {json}\n\n` frames, `data: [DONE]` last.
    while let Some(chunk) = res.chunk().await.map_err(|e| e.to_string())? {
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(pos) = buf.find("\n\n") {
            let frame = buf[..pos].to_string();
            buf.drain(..pos + 2);
            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data: ") else {
                    continue;
                };
                if data.trim() == "[DONE]" {
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else {
                    continue;
                };
                if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
                    if !delta.is_empty() {
                        text.push_str(delta);
                        // Forward as JSON so newlines and quotes survive SSE framing.
                        let _ = tx
                            .send(Event::default().event("delta").data(
                                serde_json::to_string(&json!({ "t": delta })).unwrap_or_default(),
                            ))
                            .await;
                    }
                }
                if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                    stats.prompt_tokens = u["prompt_tokens"].as_u64();
                    stats.completion_tokens = u["completion_tokens"].as_u64();
                }
            }
        }
    }
    stats.latency_ms = started.elapsed().as_millis() as u64;
    if text.is_empty() {
        return Err("stream ended with no content".into());
    }
    Ok((text, stats))
}

/// Run one turn as an SSE stream: `trace` (retrieval + prompt + user write,
/// ~instant) → `delta`* (model tokens) → `done` (stats + assistant write).
/// Emits `error` and ends if the turn can't start at all.
pub async fn turn(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<TurnRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(32);
    tokio::spawn(run_turn(state, user, req, tx));
    Sse::new(ReceiverStream::new(rx).map(Ok)).keep_alive(KeepAlive::default())
}

async fn run_turn(
    state: AppState,
    user: AuthUser,
    req: TurnRequest,
    tx: tokio::sync::mpsc::Sender<Event>,
) {
    let send = |ev: &str, data: String| {
        let tx = tx.clone();
        let ev = ev.to_string();
        async move {
            let _ = tx.send(Event::default().event(ev).data(data)).await;
        }
    };
    let fail = |msg: String| send("error", msg);

    if req.message.trim().is_empty() {
        fail("message must not be empty".into()).await;
        return;
    }
    // The operator wildcard can read every user's rows, but a write has to land
    // somewhere real — refuse rather than invent an owner.
    let user_id = user.user_id.clone();
    if user_id == crate::auth::ALL_USERS {
        fail("playground needs a concrete user_id; sign in as a non-wildcard operator".into())
            .await;
        return;
    }

    let settings = load_settings(&state.pool, &user_id)
        .await
        .unwrap_or_default();
    let (scope_project, scope_include_sandbox) = turn_scope(req.include_real);

    // 1) Retrieval — the same call a host makes.
    let assembled = match assemble_inner(
        &state.pool,
        &*state.nlp,
        &user_id,
        AssembleRequest {
            project_id: scope_project.clone(),
            include_sandbox: scope_include_sandbox,
            container_id: None,
            mode: req.mode.clone(),
            modes: None,
            // The live thread is already on screen; memory means OTHER conversations.
            exclude_container_id: Some(req.container_id.clone()),
            query: Some(req.message.clone()),
            limit: Some(
                req.limit
                    .or(settings.context_limit.map(i64::from))
                    .unwrap_or(DEFAULT_LIMIT),
            ),
        },
    )
    .await
    {
        Ok(a) => a,
        Err(e) => {
            fail(format!("retrieval failed: {e}")).await;
            return;
        }
    };
    let retrieved: Vec<RetrievedItem> = assembled.records.iter().map(RetrievedItem::from).collect();

    // 2) The prompt a host would send.
    let mut prompt = vec![PromptMessage {
        role: "system".into(),
        content: settings
            .system_prompt
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(DEFAULT_SYSTEM_PROMPT)
            .to_string(),
    }];
    if !retrieved.is_empty() {
        prompt.push(PromptMessage {
            role: "system".into(),
            content: render_context_block(&retrieved),
        });
    }
    prompt.push(PromptMessage {
        role: "user".into(),
        content: req.message.clone(),
    });

    // 3) Record the user turn BEFORE the model runs — a dead model must never
    // lose what was said. No importance, no tier claim: the store decides what
    // any of this was worth.
    let mut written = Vec::new();
    match ingest_record(
        &state.pool,
        &*state.nlp,
        &user_id,
        IngestRecordRequest {
            r#type: "conversation".into(),
            content: req.message.clone(),
            event_time: None,
            source: SOURCE_USER.into(),
            source_ref: None,
            project_id: Some(crate::routes::records::SANDBOX_PROJECT.to_string()),
            container_id: Some(req.container_id.clone()),
            mode: req.mode.clone(),
            importance: None,
            supersedes: None,
            payload: Some(json!({ "origin": "playground" })),
        },
    )
    .await
    {
        Ok(r) => written.push(r.id),
        Err(e) => {
            fail(format!("write failed: {e}")).await;
            return;
        }
    }

    // The endpoint comes from saved settings, never from the request — so a
    // half-filled form can't silently degrade the turn to retrieval-only.
    let (llm_cfg, inherited) = resolve_llm_settings(
        &state.pool,
        &state.cfg.provider,
        &settings,
        req.api_key.clone(),
    )
    .await;

    // The extractor's reading of this turn, concurrently with the reply — the
    // half of the pipeline this page exists to make visible. Same model, same
    // input as the ingest path's own extraction.
    {
        let nlp = state.nlp.clone();
        let msg = req.message.clone();
        let tx = tx.clone();
        tokio::spawn(async move {
            let payload = match nlp.extract_full(&msg).await {
                Ok(x) => serde_json::to_string(&x).unwrap_or_default(),
                Err(e) => json!({ "error": e.to_string() }).to_string(),
            };
            let _ = tx
                .send(Event::default().event("extraction").data(payload))
                .await;
        });
    }

    let trace = TraceEvent {
        retrieved,
        prompt: prompt.clone(),
        written: written.clone(),
        degraded: assembled.degraded,
        warning: assembled.warning,
        model: llm_cfg.as_ref().map(|c| c.model.clone()),
        model_inherited: inherited,
        llm_error: llm_cfg.is_none().then(|| {
            "No model configured — set a base URL and model in the playground \
             settings, or configure the system provider on the Settings page \
             (the playground inherits it). Retrieval still ran; the trace is \
             complete up to the point a model would have been called."
                .to_string()
        }),
    };
    send("trace", serde_json::to_string(&trace).unwrap_or_default()).await;

    let Some(cfg) = llm_cfg else { return };

    // 4) Stream the completion, then record the assistant side.
    let (response, stats, llm_error) = match call_llm_stream(&cfg, &prompt, &tx).await {
        Ok((text, stats)) => (Some(text), Some(stats), None),
        Err(e) => (None, None, Some(e)),
    };

    if let Some(text) = response.as_deref() {
        ingest_record(
            &state.pool,
            &*state.nlp,
            &user_id,
            IngestRecordRequest {
                r#type: "conversation".into(),
                content: text.to_string(),
                event_time: None,
                source: SOURCE_ASSISTANT.into(),
                source_ref: None,
                project_id: Some(crate::routes::records::SANDBOX_PROJECT.to_string()),
                container_id: Some(req.container_id.clone()),
                mode: req.mode.clone(),
                importance: None,
                supersedes: None,
                payload: Some(json!({
                    "origin": "playground",
                    "model": cfg.model,
                })),
            },
        )
        .await
        .map(|r| written.push(r.id))
        .unwrap_or_else(|e| {
            tracing::warn!("playground: assistant write failed: {e}");
        });
    }

    let done = DoneEvent {
        stats,
        llm_error,
        written,
    };
    send("done", serde_json::to_string(&done).unwrap_or_default()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_block_is_verbatim_and_numbered() {
        let items = vec![RetrievedItem {
            id: Uuid::nil(),
            r#type: "conversation".into(),
            source: "host:helper:user".into(),
            content: "  the quarterly report is due friday  ".into(),
            event_time: "2026-07-26T12:00:00+00:00".into(),
            container_id: Some("conv-9".into()),
            sandbox: false,
        }];
        let block = render_context_block(&items);
        assert!(block.contains("[1] (conversation, host:helper:user, 2026-07-26T12:00:00+00:00)"));
        // Trimmed but not reworded — what you read is what the model gets.
        assert!(block.contains("the quarterly report is due friday"));
    }

    #[test]
    fn empty_retrieval_still_renders_a_header() {
        assert!(render_context_block(&[]).starts_with("Relevant memories"));
    }

    #[test]
    fn turn_scope_is_sandbox_only_unless_real_is_asked_for() {
        let (project, include_sandbox) = turn_scope(false);
        assert_eq!(project.as_deref(), Some("playground"));
        assert!(!include_sandbox);
        let (project, include_sandbox) = turn_scope(true);
        assert!(project.is_none());
        assert!(include_sandbox);
    }

    #[test]
    fn seed_lines_parse_the_optional_date_prefix() {
        let (t, c) = parse_seed_line("2025-11-02 | switched the backup drive");
        assert_eq!(c, "switched the backup drive");
        assert_eq!(t.unwrap().to_rfc3339(), "2025-11-02T12:00:00+00:00");
        let (t, c) = parse_seed_line("prefers coffee at 93C");
        assert!(t.is_none());
        assert_eq!(c, "prefers coffee at 93C");
        // A pipe without a date stays content, whole.
        let (t, c) = parse_seed_line("a | b");
        assert!(t.is_none());
        assert_eq!(c, "a | b");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn seeded_memories_are_sandbox_documents_with_their_dates(pool: PgPool) {
        use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};
        struct SeedStub;
        #[async_trait::async_trait]
        impl crate::nlp::NlpService for SeedStub {
            fn provider_name(&self) -> &'static str {
                "stub"
            }
            fn provider_can_distill(&self) -> bool {
                false
            }
            fn embedder_model_name(&self) -> &str {
                "stub"
            }
            fn embedder_dimension(&self) -> usize {
                384
            }
            async fn embed_one(&self, _t: &str) -> Result<Vec<f32>, AppError> {
                Ok(vec![0.1; 384])
            }
            async fn embed_batch(&self, t: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
                Ok(t.iter().map(|_| vec![0.1; 384]).collect())
            }
            fn extract_entities(&self, _t: &str) -> Vec<String> {
                Vec::new()
            }
            async fn extract_full(&self, _t: &str) -> Result<Extraction, AppError> {
                Ok(Extraction::empty())
            }
            async fn distill_facts(
                &self,
                _e: &[EpisodeRef],
            ) -> Result<Vec<DistilledFact>, ProviderError> {
                Err(ProviderError::NotConfigured("stub".into()))
            }
        }

        let n = seed_lines(
            &pool,
            &SeedStub,
            "alice",
            "2025-11-02 | switched the backup drive\n\nprefers coffee at 93C\n",
        )
        .await
        .unwrap();
        assert_eq!(n, 2, "blank lines are skipped, not seeded");

        let rows: Vec<(String, Option<String>, chrono::DateTime<chrono::Utc>)> = sqlx::query_as(
            "SELECT content, project_id, event_time FROM raw_records \
             WHERE user_id = 'alice' ORDER BY content",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter().all(|r| r.1.as_deref() == Some("playground")),
            "every seed lands in the sandbox scope"
        );
        let dated = rows.iter().find(|r| r.0.contains("backup")).unwrap();
        assert_eq!(dated.2.to_rfc3339(), "2025-11-02T12:00:00+00:00");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn llm_settings_inherit_the_system_provider_when_sandbox_is_blank(pool: PgPool) {
        let mut env = crate::config::ProviderConfig::from_env();
        env.kind = crate::config::ProviderKind::Heuristic;
        env.remote.api_base = None;

        // Nothing anywhere → no model, not inherited.
        let (cfg, inherited) = resolve_llm_settings(&pool, &env, &Settings::default(), None).await;
        assert!(cfg.is_none());
        assert!(!inherited);

        // A system settings row → the distill-role model, marked inherited.
        crate::settings::save(
            &pool,
            &crate::settings::SystemSettings {
                provider: Some("remote".into()),
                remote_backend: Some("openai".into()),
                api_base: Some("http://127.0.0.1:11434/v1".into()),
                extract_model: Some("small:3b".into()),
                distill_model: Some("gemma4:12b".into()),
                extract_timeout_ms: None,
                distill_timeout_ms: None,
            },
        )
        .await
        .unwrap();
        let (cfg, inherited) = resolve_llm_settings(&pool, &env, &Settings::default(), None).await;
        let cfg = cfg.expect("system provider must be inherited");
        assert!(inherited);
        assert_eq!(cfg.model, "gemma4:12b", "the distill role is the probe");
        assert_eq!(cfg.base_url, "http://127.0.0.1:11434/v1");

        // A sandbox override still wins.
        let own = Settings {
            base_url: Some("http://127.0.0.1:1234/v1".into()),
            model: Some("probe:7b".into()),
            ..Default::default()
        };
        let (cfg, inherited) = resolve_llm_settings(&pool, &env, &own, None).await;
        assert!(!inherited);
        assert_eq!(cfg.unwrap().model, "probe:7b");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn facts_for_container_scopes_to_the_conversation(pool: PgPool) {
        let raw_here = Uuid::new_v4();
        let raw_other = Uuid::new_v4();
        for (id, container) in [(raw_here, "here"), (raw_other, "elsewhere")] {
            sqlx::query(
                "INSERT INTO raw_records (id, type, content, event_time, source, user_id, container_id) \
                 VALUES ($1, 'conversation', 'x', NOW(), 'test', 'alice', $2)",
            )
            .bind(id)
            .bind(container)
            .execute(&pool)
            .await
            .unwrap();
        }
        for (fact, raw, content) in [
            (Uuid::new_v4(), raw_here, "learned from here"),
            (Uuid::new_v4(), raw_other, "learned elsewhere"),
        ] {
            sqlx::query(
                "INSERT INTO curated_nodes (id, kind, content, level, user_id) \
                 VALUES ($1, 'semantic', $2, 0, 'alice')",
            )
            .bind(fact)
            .bind(content)
            .execute(&pool)
            .await
            .unwrap();
            sqlx::query(
                "INSERT INTO curated_edges (from_id, to_id, kind) VALUES ($1, $2, 'derived_from')",
            )
            .bind(fact)
            .bind(raw)
            .execute(&pool)
            .await
            .unwrap();
        }

        let facts = facts_for_container(&pool, "alice", "here").await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "learned from here");
        assert!(facts_for_container(&pool, "bob", "here")
            .await
            .unwrap()
            .is_empty());
    }
}

// ---------------------------------------------------------------------------
// Seed — fill the sandbox with memories to play against.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SeedRequest {
    /// One memory per line. A line may start with `YYYY-MM-DD |` to backdate
    /// its event time — that is what makes recency ranking and the distill
    /// prompt's newest-wins rule testable against seeded history.
    pub text: String,
}

/// Ingest pasted lines as sandbox `document` records through the REAL ingest
/// path — embeddings, entity extraction, everything — so what you play with
/// was made exactly the way real memories are.
pub async fn seed(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<SeedRequest>,
) -> AppResult<Json<serde_json::Value>> {
    let user_id = user.user_id.clone();
    if user_id == crate::auth::ALL_USERS {
        return Err(AppError::bad_request(
            "seeding needs a concrete user_id; sign in as a non-wildcard operator",
        ));
    }
    let seeded = seed_lines(&state.pool, &*state.nlp, &user_id, &req.text).await?;
    Ok(Json(json!({ "seeded": seeded })))
}

/// Parse `YYYY-MM-DD | content` when the prefix is present.
fn parse_seed_line(line: &str) -> (Option<chrono::DateTime<chrono::Utc>>, &str) {
    if let Some((date, rest)) = line.split_once('|') {
        if let Ok(d) = chrono::NaiveDate::parse_from_str(date.trim(), "%Y-%m-%d") {
            let t = d
                .and_hms_opt(12, 0, 0)
                .map(|dt| chrono::DateTime::from_naive_utc_and_offset(dt, chrono::Utc));
            return (t, rest.trim());
        }
    }
    (None, line.trim())
}

pub(crate) async fn seed_lines(
    pool: &PgPool,
    nlp: &dyn crate::nlp::NlpService,
    user_id: &str,
    text: &str,
) -> AppResult<i64> {
    let mut seeded = 0_i64;
    for line in text.lines() {
        let (event_time, content) = parse_seed_line(line);
        if content.is_empty() {
            continue;
        }
        ingest_record(
            pool,
            nlp,
            user_id,
            IngestRecordRequest {
                r#type: "document".into(),
                content: content.to_string(),
                event_time,
                source: "playground:seed".into(),
                source_ref: None,
                project_id: Some(crate::routes::records::SANDBOX_PROJECT.to_string()),
                container_id: None,
                mode: None,
                importance: None,
                supersedes: None,
                payload: Some(json!({ "origin": "playground-seed" })),
            },
        )
        .await?;
        seeded += 1;
    }
    Ok(seeded)
}

// ---------------------------------------------------------------------------
// Distill now — watch the conversation become facts.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct DistillNowRequest {
    pub container_id: String,
}

/// Run one REAL incremental curation pass — the same one the scheduler runs,
/// same per-user lock — then report the semantic facts whose lineage reaches
/// this conversation. Talk, distill, see what it learned.
pub async fn distill_now(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<DistillNowRequest>,
) -> AppResult<Json<serde_json::Value>> {
    let user_id = user.user_id.clone();
    if user_id == crate::auth::ALL_USERS {
        return Err(AppError::bad_request(
            "distillation needs a concrete user_id; sign in as a non-wildcard operator",
        ));
    }
    let stats = crate::curation::curate(&state.pool, &*state.nlp, &user_id).await?;
    let facts = facts_for_container(&state.pool, &user_id, &req.container_id).await?;
    Ok(Json(json!({
        "locked_out": stats.locked_out,
        "promoted": stats.promoted,
        "refreshed": stats.refreshed,
        "distilled": stats.distilled,
        "skipped_distill": stats.skipped_distill,
        "provider": state.nlp.provider_name(),
        "facts": facts,
    })))
}

#[derive(Debug, Serialize, sqlx::FromRow)]
pub struct ContainerFact {
    pub id: Uuid,
    pub content: String,
    pub event_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// Semantic facts whose `derived_from` lineage includes any raw record of this
/// conversation — what the store has learned from it, newest evidence first.
pub(crate) async fn facts_for_container(
    pool: &PgPool,
    user_id: &str,
    container_id: &str,
) -> AppResult<Vec<ContainerFact>> {
    let rows = sqlx::query_as::<_, ContainerFact>(
        r#"
        SELECT DISTINCT n.id, n.content, n.event_time
        FROM curated_nodes n
        JOIN curated_edges e ON e.from_id = n.id AND e.kind = 'derived_from'
        JOIN raw_records r ON r.id = e.to_id
        WHERE n.kind = 'semantic'
          AND n.user_id = $1
          AND r.container_id = $2
        ORDER BY n.event_time DESC NULLS LAST, n.id
        "#,
    )
    .bind(user_id)
    .bind(container_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Settings — server-side, per operator.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Default, sqlx::FromRow)]
pub struct Settings {
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub context_limit: Option<i32>,
}

pub(crate) async fn load_settings(pool: &PgPool, user_id: &str) -> AppResult<Settings> {
    let row = sqlx::query_as::<_, Settings>(
        "SELECT base_url, model, system_prompt, context_limit \
         FROM playground_settings WHERE user_id = $1",
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.unwrap_or_default())
}

/// Save (upsert). Blank strings normalise to NULL so "cleared" and "never set"
/// are the same state — otherwise an empty base_url reads as configured and the
/// turn silently runs without a model, which is exactly the trap this replaces.
pub async fn save_settings(
    State(state): State<AppState>,
    user: AuthUser,
    Json(req): Json<Settings>,
) -> AppResult<Json<Settings>> {
    let norm = |v: Option<String>| v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let base_url = norm(req.base_url);
    if let Some(u) = base_url.as_deref() {
        validate_base_url(u).map_err(AppError::bad_request)?;
    }
    let model = norm(req.model);
    let system_prompt = norm(req.system_prompt);
    let context_limit = req.context_limit.filter(|n| *n > 0 && *n <= 200);

    sqlx::query(
        "INSERT INTO playground_settings (user_id, base_url, model, system_prompt, context_limit) \
         VALUES ($1, $2, $3, $4, $5) \
         ON CONFLICT (user_id) DO UPDATE SET \
           base_url = EXCLUDED.base_url, model = EXCLUDED.model, \
           system_prompt = EXCLUDED.system_prompt, context_limit = EXCLUDED.context_limit, \
           updated_at = NOW()",
    )
    .bind(&user.user_id)
    .bind(&base_url)
    .bind(&model)
    .bind(&system_prompt)
    .bind(context_limit)
    .execute(&state.pool)
    .await?;

    Ok(Json(Settings {
        base_url,
        model,
        system_prompt,
        context_limit,
    }))
}
