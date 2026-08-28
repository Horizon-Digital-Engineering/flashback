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
/// — the point is that curation can't tell them apart from a host's.
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
    /// Set when no model will run (not configured); the stream ends after this.
    pub llm_error: Option<String>,
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

    // 1) Retrieval — the same call a host makes.
    let assembled = match assemble_inner(
        &state.pool,
        &*state.nlp,
        &user_id,
        AssembleRequest {
            project_id: None,
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
            project_id: None,
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
    let llm_cfg = match (settings.base_url.as_deref(), settings.model.as_deref()) {
        (Some(base), Some(model)) => Some(LlmSettings {
            base_url: base.to_string(),
            model: model.to_string(),
            api_key: req.api_key.clone(),
        }),
        _ => None,
    };

    let trace = TraceEvent {
        retrieved,
        prompt: prompt.clone(),
        written: written.clone(),
        degraded: assembled.degraded,
        warning: assembled.warning,
        llm_error: llm_cfg.is_none().then(|| {
            "No model configured — set a base URL and model name in settings. \
             Retrieval still ran; the trace is complete up to the point a model \
             would have been called."
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
                project_id: None,
                container_id: Some(req.container_id.clone()),
                mode: req.mode.clone(),
                importance: None,
                supersedes: None,
                payload: Some(json!({
                    "origin": "playground",
                    "model": settings.model,
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
