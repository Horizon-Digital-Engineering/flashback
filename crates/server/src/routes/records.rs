//! The RAW-records door (medallion "bronze"). Universal typed records, immutable
//! and append-only. Raw itself holds NO embeddings — but on ingest we compute a
//! DERIVED embedding into `raw_embeddings` (best-effort; raw is the source of
//! truth, the embedding is rebuildable). `/records/context` does hybrid
//! vector+keyword retrieval; the curation pipeline (summaries/references/decay)
//! layers on later behind the same endpoints.
//!
//! Endpoints (nested under /records):
//!   POST /records          ingest a typed record (+ derived embedding)
//!   POST /records/query    structured reads (dashboards)
//!   POST /records/context  hybrid vector+keyword context (getContext)
//!   GET  /records/:id       one record
//!   GET  /records/:id/lineage   the supersede chain

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Duration, Utc};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{AssertSqlSafe, FromRow, PgPool};
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    nlp::NlpService,
    AppState,
};

/// Columns of raw_records in RawRecordRow field order. `content_tsv` is a
/// generated tsvector and is deliberately NOT selected (not on the struct).
/// `state_kind`/`state_key` are promoted state_object columns (010) — read via
/// the payload on the struct, so also not selected here.
pub(crate) const COLS: &str = "id, type, content, content_hash, event_time, ingest_time, \
    source, source_ref, user_id, project_id, container_id, mode, importance, \
    supersedes, payload";
const COLS_R: &str = "r.id, r.type, r.content, r.content_hash, r.event_time, r.ingest_time, \
    r.source, r.source_ref, r.user_id, r.project_id, r.container_id, r.mode, r.importance, \
    r.supersedes, r.payload";

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/", post(ingest))
        .route("/import", post(import_records))
        .route("/query", post(query_records))
        .route("/context", post(assemble))
        .route("/summaries", get(summaries))
        .route("/rebuild", post(rebuild_curation))
        // The raw-native reference surface (state_object records).
        .nest("/state", crate::references::router(state.clone()))
        .route("/{id}", get(get_record))
        .route("/{id}/lineage", get(lineage))
        .route("/{id}/derivations", get(derivations))
        .with_state(state)
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct RawRecordRow {
    pub id: Uuid,
    #[serde(rename = "type")]
    pub r#type: String,
    pub content: String,
    pub content_hash: String,
    pub event_time: DateTime<Utc>,
    pub ingest_time: DateTime<Utc>,
    pub source: String,
    pub source_ref: Option<String>,
    pub user_id: String,
    pub project_id: Option<String>,
    pub container_id: Option<String>,
    pub mode: Option<String>,
    pub importance: Option<f32>,
    pub supersedes: Option<Uuid>,
    pub payload: Option<Value>,
}

impl RawRecordRow {
    /// The reference kind for a state_object row, from the `{kind,...}` payload
    /// convention (the same value the 010 trigger promotes onto `state_kind`).
    pub(crate) fn state_kind_of(&self, payload: &Value) -> AppResult<String> {
        payload
            .get("kind")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state_object payload missing kind")))
    }

    /// The reference key for a state_object row, from the `{key,...}` payload.
    pub(crate) fn state_key_of(&self, payload: &Value) -> AppResult<String> {
        payload
            .get("key")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state_object payload missing key")))
    }
}

/// A raw type is HOW A RECORD MUST BE PROCESSED, not a memory tier and not what
/// it is about — a writer says what arrived, curation says what it became. Only
/// types with a real processing rule are accepted: an accepted-but-unread type
/// is a silent data hole. Kept in lockstep with the CHECK in
/// `003_raw_records.sql`; add a value here only alongside its extractor.
fn validate_type(t: &str) -> AppResult<()> {
    match t {
        "conversation" | "document" | "state_object" => Ok(()),
        other => Err(AppError::bad_request(format!("unknown type: {other}"))),
    }
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct IngestRecordRequest {
    pub r#type: String,
    pub content: String,
    /// When it actually happened (RFC3339). Defaults to ingest time.
    #[serde(default)]
    pub event_time: Option<DateTime<Utc>>,
    pub source: String,
    #[serde(default)]
    pub source_ref: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub container_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    /// id this record supersedes (forward pointer; the old row is never mutated).
    #[serde(default)]
    pub supersedes: Option<Uuid>,
    /// Metadata the SOURCE handed us, verbatim — the exporting model, a
    /// conversation title, the folder a file was dropped in. Never normalised on
    /// the way in: normalising is interpreting, and interpreting is deriving.
    #[serde(default)]
    pub payload: Option<Value>,
}

#[derive(Debug, Serialize)]
pub struct IngestRecordResponse {
    pub id: Uuid,
}

async fn ingest(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<IngestRecordRequest>,
) -> AppResult<Json<IngestRecordResponse>> {
    Ok(Json(
        ingest_record(&state.pool, &*state.nlp, &auth_user.user_id, req).await?,
    ))
}

pub(crate) async fn ingest_record(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    req: IngestRecordRequest,
) -> AppResult<IngestRecordResponse> {
    validate_type(&req.r#type)?;
    if req.content.trim().is_empty() {
        return Err(AppError::bad_request("content must not be empty"));
    }

    // A supersede target must exist AND belong to the caller.
    if let Some(sid) = req.supersedes {
        let owner: Option<String> =
            sqlx::query_scalar("SELECT user_id FROM raw_records WHERE id = $1")
                .bind(sid)
                .fetch_optional(pool)
                .await?;
        match owner {
            Some(u) if u == user_id => {}
            _ => {
                return Err(AppError::bad_request(format!(
                    "supersedes target {sid} not found"
                )))
            }
        }
    }

    let id = Uuid::new_v4();
    let event_time = req.event_time.unwrap_or_else(Utc::now);
    let importance = req.importance.map(|i| i.clamp(0.0, 1.0));

    // Resolve the mode by precedence: caller override → LLM auto-classify →
    // project default → general. The LLM classification only fires for an
    // LLM-capable provider (the heuristic returns `None`, so no extra work on
    // the no-LLM path); a resolve failure never blocks ingest.
    let mode = resolve_ingest_mode(pool, nlp, user_id, req.mode.as_deref(), &req.content).await;

    // content_hash + content_tsv + ingest_time are generated/defaulted by the DB.
    sqlx::query(
        r#"
        INSERT INTO raw_records
            (id, type, content, event_time, source, source_ref,
             user_id, project_id, container_id, mode, importance, supersedes, payload)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
        "#,
    )
    .bind(id)
    .bind(&req.r#type)
    .bind(&req.content)
    .bind(event_time)
    .bind(&req.source)
    .bind(&req.source_ref)
    .bind(user_id)
    .bind(&req.project_id)
    .bind(&req.container_id)
    .bind(mode.as_ref().map(|m| m.name.as_str()))
    .bind(importance)
    .bind(req.supersedes)
    .bind(&req.payload)
    .execute(pool)
    .await?;

    // Derived embedding — best-effort, in the resolved mode's geometry. Raw is
    // the source of truth; if the embedder is down a backfill job can fill the
    // gap later. Never fails ingest.
    let embedder_key = mode
        .as_ref()
        .map(|m| m.embedder.clone())
        .unwrap_or_else(|| nlp.embedder_model_name().to_string());
    if let Ok((dim, embedding)) = nlp.embed_for_mode(&embedder_key, &req.content).await {
        let _ = write_raw_embedding(pool, id, &embedder_key, dim, embedding).await;
    }

    Ok(IngestRecordResponse { id })
}

/// Resolve the mode a record lands in. Runs the AiProvider's classifier only
/// when the caller didn't pin a mode AND the provider is LLM-capable (the
/// heuristic returns `None`, so this stays a no-op on the no-LLM path). A
/// classification / resolution failure degrades gracefully to the project
/// default. Returns `None` only if the modes table itself is unreachable, in
/// which case the record is stored mode-less (the historical behavior).
async fn resolve_ingest_mode(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    caller_override: Option<&str>,
    content: &str,
) -> Option<crate::modes::Mode> {
    // Only reach for the LLM when there's no explicit override — the override
    // wins anyway, so we'd waste a call.
    let llm_classified = if caller_override.is_none() {
        match nlp.extract_full(content).await {
            Ok(e) => e.mode,
            Err(_) => None,
        }
    } else {
        None
    };
    crate::modes::resolve_mode(pool, user_id, caller_override, llm_classified.as_deref())
        .await
        .ok()
}

/// Batch-embed `texts` with a specific embedder key. When the key is the
/// default embedder's model, the fast batched path is used; otherwise each item
/// is embedded through the mode-aware single path (the extra embedders load
/// lazily and inference is cheap per item). Returns vectors in input order.
async fn embed_batch_with(
    nlp: &dyn NlpService,
    embedder_key: &str,
    texts: Vec<String>,
) -> Result<Vec<Vec<f32>>, AppError> {
    // The default 384-dim embedder has a real batched path — use it.
    let is_default = flashback_nlp::model_name_for_key(embedder_key)
        .map(|n| n == nlp.embedder_model_name())
        .unwrap_or(true);
    if is_default {
        return nlp.embed_batch(texts).await;
    }
    let mut out = Vec::with_capacity(texts.len());
    for t in texts {
        let (_dim, v) = nlp.embed_for_mode(embedder_key, &t).await?;
        out.push(v);
    }
    Ok(out)
}

/// Write a derived embedding into the `embedding_<dim>` column matching the
/// mode's embedder dimension. Exactly one column is populated per (record,
/// model) row; the other two stay NULL. Idempotent on the composite PK.
pub(crate) async fn write_raw_embedding(
    pool: &PgPool,
    record_id: Uuid,
    model: &str,
    dim: usize,
    embedding: Vec<f32>,
) -> Result<(), sqlx::Error> {
    let col = embedding_col_for_dim(dim);
    let sql = format!(
        "INSERT INTO raw_embeddings (record_id, model, {col}) VALUES ($1, $2, $3) \
         ON CONFLICT (record_id, model) DO NOTHING"
    );
    sqlx::query(AssertSqlSafe(sql))
        .bind(record_id)
        .bind(model)
        .bind(Vector::from(embedding))
        .execute(pool)
        .await?;
    Ok(())
}

/// The `raw_embeddings` / `curated_embeddings` column name for a vector
/// dimension. 384 is the original `embedding` column; 768/1024 are the
/// per-dimension columns added for modes. Unknown dims fall back to the 384
/// column (the default embedder's), which is the safe historical behavior.
pub(crate) fn embedding_col_for_dim(dim: usize) -> &'static str {
    match dim {
        768 => "embedding_768",
        1024 => "embedding_1024",
        _ => "embedding",
    }
}

// ---------------------------------------------------------------------------
// Bulk import (backdated, idempotent) — the door for a corpus of past
// conversations. Records with a stable `source_ref` dedup on re-import.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ImportRecord {
    pub r#type: String,
    pub content: String,
    #[serde(default)]
    pub event_time: Option<DateTime<Utc>>,
    pub source: String,
    #[serde(default)]
    pub source_ref: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub container_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    /// Metadata the source handed us, verbatim. `conversation_title` here is
    /// what episode formation prefers when naming a conversation. An export's
    /// own grouping (a ChatGPT "project") belongs here too, NOT in `project_id`
    /// — `project_id` partitions and would fragment the curated layer.
    #[serde(default)]
    pub payload: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ImportRequest {
    pub records: Vec<ImportRecord>,
}

#[derive(Debug, Serialize)]
pub struct ImportResponse {
    pub imported: usize,
    pub skipped: usize,
}

async fn import_records(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<ImportRequest>,
) -> AppResult<Json<ImportResponse>> {
    Ok(Json(
        import_records_inner(&state.pool, &*state.nlp, &auth_user.user_id, req).await?,
    ))
}

/// Bulk insert with per-row dedup on (user_id, source, source_ref). Invalid /
/// empty rows are skipped, not fatal. Embeddings are batch-computed after commit.
/// Clients chunk large corpora into batches (e.g. a few hundred records/call).
pub(crate) async fn import_records_inner(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    req: ImportRequest,
) -> AppResult<ImportResponse> {
    let total = req.records.len();
    // (record_id, content, embedder_key) for the post-commit embed pass. Each
    // row carries the embedder its resolved mode pins so the batch writes into
    // the matching per-dimension column.
    let mut inserted: Vec<(Uuid, String, String)> = Vec::new();

    let mut tx = pool.begin().await?;
    for r in req.records {
        if validate_type(&r.r#type).is_err() || r.content.trim().is_empty() {
            continue;
        }
        let id = Uuid::new_v4();
        let event_time = r.event_time.unwrap_or_else(Utc::now);
        let importance = r.importance.map(|i| i.clamp(0.0, 1.0));
        // Bulk import resolves mode by caller override → project default (no
        // per-record LLM classification — a corpus load stays cheap; the
        // background curation pass can re-derive later).
        let mode = crate::modes::resolve_mode(pool, user_id, r.mode.as_deref(), None)
            .await
            .ok();
        let embedder_key = mode
            .as_ref()
            .map(|m| m.embedder.clone())
            .unwrap_or_else(|| nlp.embedder_model_name().to_string());
        let new_id: Option<Uuid> = sqlx::query_scalar(
            r#"
            INSERT INTO raw_records
                (id, type, content, event_time, source, source_ref,
                 user_id, project_id, container_id, mode, importance, payload)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            ON CONFLICT (user_id, source, source_ref) WHERE source_ref IS NOT NULL
            DO NOTHING
            RETURNING id
            "#,
        )
        .bind(id)
        .bind(&r.r#type)
        .bind(&r.content)
        .bind(event_time)
        .bind(&r.source)
        .bind(&r.source_ref)
        .bind(user_id)
        .bind(&r.project_id)
        .bind(&r.container_id)
        .bind(mode.as_ref().map(|m| m.name.as_str()))
        .bind(importance)
        .bind(&r.payload)
        .fetch_optional(&mut *tx)
        .await?;
        if let Some(nid) = new_id {
            inserted.push((nid, r.content, embedder_key));
        }
    }
    tx.commit().await?;

    // Embed the freshly imported rows in their mode's geometry (best-effort; a
    // backfill can retry). Group by embedder so each distinct model batches once.
    if !inserted.is_empty() {
        let mut by_embedder: std::collections::HashMap<String, Vec<(Uuid, String)>> =
            std::collections::HashMap::new();
        for (rid, content, key) in &inserted {
            by_embedder
                .entry(key.clone())
                .or_default()
                .push((*rid, content.clone()));
        }
        for (embedder_key, rows) in by_embedder {
            let contents: Vec<String> = rows.iter().map(|(_, c)| c.clone()).collect();
            // Resolve the dim once for this embedder (unknown → default 384).
            let dim = flashback_nlp::model_for_key(&embedder_key)
                .map(|(_, d)| d)
                .unwrap_or(384);
            let embs = match embed_batch_with(nlp, &embedder_key, contents).await {
                Ok(e) if e.len() == rows.len() => e,
                _ => continue,
            };
            for ((rid, _), emb) in rows.iter().zip(embs) {
                let _ = write_raw_embedding(pool, *rid, &embedder_key, dim, emb).await;
            }
        }
    }

    Ok(ImportResponse {
        imported: inserted.len(),
        skipped: total - inserted.len(),
    })
}

// ---------------------------------------------------------------------------
// Query (structured reads)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct QueryRecordsRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub container_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub since: Option<DateTime<Utc>>,
    #[serde(default)]
    pub until: Option<DateTime<Utc>>,
    #[serde(default)]
    pub limit: Option<i64>,
}

async fn query_records(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<QueryRecordsRequest>,
) -> AppResult<Json<Vec<RawRecordRow>>> {
    Ok(Json(
        query_records_inner(&state.pool, &auth_user.user_id, req).await?,
    ))
}

/// Active = not superseded (no newer row points at it) AND not expired.
pub(crate) async fn query_records_inner(
    pool: &PgPool,
    user_id: &str,
    req: QueryRecordsRequest,
) -> AppResult<Vec<RawRecordRow>> {
    let limit = req.limit.unwrap_or(100).clamp(1, 1000);
    let sql = format!(
        r#"
        SELECT {COLS} FROM raw_records
        WHERE user_id = $1
          AND ($2::text IS NULL OR project_id = $2)
          AND ($3::text IS NULL OR container_id = $3)
          AND ($4::text IS NULL OR mode = $4)
          AND ($5::text IS NULL OR type = $5)
          AND ($6::timestamptz IS NULL OR event_time >= $6)
          AND ($7::timestamptz IS NULL OR event_time <= $7)
          AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $1)
        ORDER BY event_time DESC
        LIMIT $8
        "#
    );
    // sql is built only from trusted column constants (COLS) + $-params.
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(&req.project_id)
        .bind(&req.container_id)
        .bind(&req.mode)
        .bind(&req.r#type)
        .bind(req.since)
        .bind(req.until)
        .bind(limit)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Context assemble (getContext): summary-first, decay/importance-weighted
// retrieval when a query is present, recency when it isn't.
//
// The blended score for a query hit is
//     w_sem·cosine + w_kw·bm25 + w_rec·recency + w_imp·importance + w_decay·ref_weight
// applied over the CURATED layer first (summaries + level-0 nodes), which is
// the primary feed — "summarize first, drill to raw on demand". The curated
// hits are mapped back to their source raw records (the shape callers already
// consume), and direct raw retrieval backstops any slot the curated layer
// hasn't covered yet. Retrieval bumps each returned curated node's decay clock.
// ---------------------------------------------------------------------------

/// Blend weights. `w_sem` dominates (semantic recall is the point); keyword is
/// the lexical backstop; recency/importance/decay are the demotion signals that
/// keep a stale-but-similar node below a fresh one. They need not sum to 1 —
/// only their relative magnitude ranks.
const W_SEM: f64 = 0.55;
const W_KW: f64 = 0.20;
const W_REC: f64 = 0.10;
const W_IMP: f64 = 0.075;
const W_DECAY: f64 = 0.075;
/// Half-life (hours) of the recency term e^(-age/REC_S). ~30 days.
const REC_S_HOURS: f64 = 30.0 * 24.0;

#[derive(Debug, Clone, Deserialize)]
pub struct AssembleRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    /// Include the playground sandbox scope in an UNSCOPED request. Off by
    /// default so hosts never retrieve sandbox chatter as memory; the
    /// playground sets it so its own prior test conversations stay
    /// recallable beside the real store. An explicit `project_id` ignores it.
    #[serde(default)]
    pub include_sandbox: bool,
    #[serde(default)]
    pub container_id: Option<String>,
    /// The cognitive register to search. A single mode name uses that mode's
    /// vector geometry. `"all"` (or `modes` = `["all"]`) is a cross-mode
    /// request that can't compare cosine across mismatched dims, so it degrades
    /// to keyword + recency. Unset resolves to the user's default register.
    #[serde(default)]
    pub mode: Option<String>,
    /// Explicit multi-mode form. `["all"]` — or more than one distinct mode —
    /// is a cross-mode (degraded) request.
    #[serde(default)]
    pub modes: Option<Vec<String>>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub limit: Option<i64>,
    /// Leave out the conversation the caller is currently IN. The host already
    /// replays the live thread in its own window; injecting it again is an echo
    /// — it crowds out the cross-conversation memory this endpoint exists for,
    /// and a query's nearest neighbour is otherwise its own previous turn.
    #[serde(default)]
    pub exclude_container_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct AssembleResponse {
    pub records: Vec<RawRecordRow>,
    /// True when the request spanned modes and fell back to keyword/entity/
    /// recency retrieval (no cross-dimension vector search). The `warning`
    /// spells out why.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub degraded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

/// The project scope every playground write lands in. Same pipeline, separate
/// data space: sandbox records promote, cluster and distill inside their own
/// per-project curation bucket, and an unscoped context assembly excludes them
/// unless the caller opts in via `include_sandbox`.
pub(crate) const SANDBOX_PROJECT: &str = "playground";

async fn assemble(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<AssembleRequest>,
) -> AppResult<Json<AssembleResponse>> {
    Ok(Json(
        assemble_inner(&state.pool, &*state.nlp, &auth_user.user_id, req).await?,
    ))
}

/// Ceiling on the total content characters one assemble response carries.
/// Tokens are the real currency, but this server has no tokenizer; characters
/// are an honest, deterministic proxy (~4 per English token). The default is
/// sized so a full page never dominates a host's context window on its own.
fn assemble_char_budget() -> usize {
    std::env::var("FLASHBACK_ASSEMBLE_CHAR_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or(60_000)
}

/// Enforce the record limit and the character budget on a final record set,
/// in rank order. The first record always survives, so one long document can
/// still be served alone. Returns a warning sentence when the budget clipped.
fn clamp_assembled(records: &mut Vec<RawRecordRow>, limit: i64) -> Option<String> {
    records.truncate(limit as usize);
    let budget = assemble_char_budget();
    let before = records.len();
    let mut used = 0usize;
    let mut keep = before;
    for (i, r) in records.iter().enumerate() {
        used = used.saturating_add(r.content.chars().count());
        if used > budget && i > 0 {
            keep = i;
            break;
        }
    }
    if keep < before {
        records.truncate(keep);
        Some(format!(
            "content clipped to the {budget}-character budget: {keep} of {before} \
             ranked records returned (raise FLASHBACK_ASSEMBLE_CHAR_BUDGET to widen)"
        ))
    } else {
        None
    }
}

/// Merge an optional clip warning into an existing warning line.
fn merge_warning(existing: Option<String>, clip: Option<String>) -> Option<String> {
    match (existing, clip) {
        (Some(a), Some(b)) => Some(format!("{a} {b}")),
        (a, b) => a.or(b),
    }
}

pub(crate) async fn assemble_inner(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    req: AssembleRequest,
) -> AppResult<AssembleResponse> {
    let limit = req.limit.unwrap_or(50).clamp(1, 200);
    let query = req.query.clone().unwrap_or_default();
    // SQL literal for the sandbox opt-in — a compile-time-safe constant, not user input.
    let sandbox_gate = if req.include_sandbox { "TRUE" } else { "FALSE" };

    // A cross-mode request can't compare cosine across mismatched dims. When the
    // caller asks for `all` (or names more than one distinct mode) we degrade to
    // keyword + entity + recency over raw, and flag it. A single mode (or unset,
    // which resolves to the user's default) takes the normal vector path.
    let cross_mode = is_cross_mode(&req);

    // Resolve the single register to search when not cross-mode. `None` on a
    // cross-mode request; otherwise the mode's embedder + dim drive which vector
    // column and geometry we use.
    let resolved_mode = if cross_mode {
        None
    } else {
        let name = req.mode.as_deref().or_else(|| {
            req.modes
                .as_ref()
                .and_then(|m| m.first().map(|s| s.as_str()))
        });
        crate::modes::resolve_mode(pool, user_id, name, None)
            .await
            .ok()
    };
    // The mode name the query is scoped to, and the embedder + column that
    // matches its geometry. Cross-mode leaves these empty and never scopes/embeds.
    let scope_mode: Option<String> = resolved_mode.as_ref().map(|m| m.name.clone());
    let embedder_key = resolved_mode
        .as_ref()
        .map(|m| m.embedder.clone())
        .unwrap_or_else(|| nlp.embedder_model_name().to_string());

    if query.trim().is_empty() {
        // No query -> pure recency over raw (the backstop feed; nothing to score).
        let sql = format!(
            r#"
            SELECT {COLS} FROM raw_records
            WHERE user_id = $1
              AND ($2::text IS NULL OR project_id = $2)
              AND ($2::text IS NOT NULL OR {sandbox_gate} OR project_id IS DISTINCT FROM '{SANDBOX_PROJECT}')
              AND ($3::text IS NULL OR container_id = $3)
              AND ($4::text IS NULL OR mode = $4)
              AND ($6::text IS NULL OR container_id IS DISTINCT FROM $6)
              AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $1)
            ORDER BY event_time DESC
            LIMIT $5
            "#
        );
        let mut records = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
            .bind(user_id)
            .bind(&req.project_id)
            .bind(&req.container_id)
            .bind(&scope_mode)
            .bind(limit)
            .bind(&req.exclude_container_id)
            .fetch_all(pool)
            .await?;
        let clip = clamp_assembled(&mut records, limit);
        return Ok(AssembleResponse {
            records,
            degraded: false,
            warning: clip,
        });
    }

    // Cross-mode: no shared vector geometry — keyword(BM25) + entity + recency
    // over raw, across every mode. Visible `degraded` flag + a warning.
    if cross_mode {
        let mut records = cross_mode_degraded(pool, nlp, user_id, &req, &query, limit).await?;
        let clip = clamp_assembled(&mut records, limit);
        return Ok(AssembleResponse {
            records,
            degraded: true,
            warning: merge_warning(
                Some(
                    "cross-mode request: vector search skipped (modes live in different \
                     embedding spaces); ranked by keyword, entity overlap, and recency."
                        .to_string(),
                ),
                clip,
            ),
        });
    }

    // Single-mode vector path — embed the query in this mode's geometry and read
    // the matching per-dimension column.
    let (dim, qvec) = nlp.embed_for_mode(&embedder_key, &query).await?;
    let model = embedder_key.clone();
    let emb_col = embedding_col_for_dim(dim);

    // 1) Rank the CURATED layer with the full blended score. Summaries and
    // level-0 nodes compete on the same score; higher `level` breaks ties so a
    // summary that covers a cluster leads its own children ("summary first").
    // recency = e^(-age_hours/REC_S); ref_weight = e^(-idle_hours/half_life)
    // (the Ebbinghaus decay, computed inline from ref_weights; a node with no
    // ref_weights row scores 1.0 so absence never demotes it). The `mode` scope
    // is exact (never crosses a register), and the embedding column matches the
    // mode's dimension.
    let curated_scored: Vec<(Uuid, i32)> = {
        let sql = format!(
            r#"
            SELECT n.id, n.level
            FROM curated_nodes n
            LEFT JOIN curated_embeddings ce ON ce.node_id = n.id AND ce.model = $1
            LEFT JOIN ref_weights rw ON rw.ref_id = n.id
            WHERE n.user_id = $2
              AND ($3::text IS NULL OR n.project_id = $3)
              AND ($3::text IS NOT NULL OR {sandbox_gate} OR n.project_id IS DISTINCT FROM '{SANDBOX_PROJECT}')
              AND n.mode IS NOT DISTINCT FROM $4
              AND NOT EXISTS (
                  SELECT 1 FROM curated_edges s WHERE s.to_id = n.id AND s.kind = 'supersedes'
              )
              AND ($14::text IS NULL OR n.meta->>'container_id' IS DISTINCT FROM $14)
            ORDER BY (
                  $5 * COALESCE(NULLIF(1 - (ce.{emb_col} <=> $6), 'NaN'::float8), 0)
                + $7 * ts_rank(to_tsvector('english', n.content), plainto_tsquery('english', $8))
                + $9  * exp(- GREATEST(EXTRACT(EPOCH FROM (NOW() - COALESCE(n.event_time, n.created_at))) / 3600.0, 0) / $10)
                + $11 * COALESCE(n.importance, 0)
                + $12 * COALESCE(
                        exp(- GREATEST(EXTRACT(EPOCH FROM (NOW() - rw.last_access)) / 3600.0, 0)
                              / COALESCE(rw.s_hours,
                                  CASE rw.decay_class
                                      WHEN 'pinned' THEN 2160.0
                                      WHEN 'ephemeral' THEN 48.0
                                      ELSE 336.0
                                  END)),
                        1.0)
            ) DESC, n.level DESC, n.created_at DESC
            LIMIT $13
        "#
        );
        // emb_col is a trusted constant from embedding_col_for_dim; all values
        // are $-bound params.
        sqlx::query_as::<_, (Uuid, i32)>(AssertSqlSafe(sql))
            .bind(&model)
            .bind(user_id)
            .bind(&req.project_id)
            .bind(&scope_mode)
            .bind(W_SEM)
            .bind(Vector::from(qvec.clone()))
            .bind(W_KW)
            .bind(&query)
            .bind(W_REC)
            .bind(REC_S_HOURS)
            .bind(W_IMP)
            .bind(W_DECAY)
            .bind(limit)
            .bind(&req.exclude_container_id)
            .fetch_all(pool)
            .await?
    };

    // Bump the decay clock for every curated node we're serving (accessing a
    // node resets its retention to 1.0). Never deletes.
    let touched: Vec<Uuid> = curated_scored.iter().map(|(id, _)| *id).collect();
    crate::decay::touch_refs(pool, crate::decay::REF_KIND_CURATED, &touched).await?;

    // 2) Resolve curated hits -> source raw ids, in curated-rank order. A
    // summary resolves through its `summarizes` chain down to level-0 nodes,
    // then each level-0 node's `derived_from` edges to raw ids. Each node may
    // only contribute up to the REMAINING capacity, enforced inside the
    // expansion query — the limit binds during expansion, never after it.
    let mut ordered_raw_ids: Vec<Uuid> = Vec::new();
    let mut seen: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for (node_id, _level) in &curated_scored {
        let remaining = limit - ordered_raw_ids.len() as i64;
        if remaining <= 0 {
            break;
        }
        let raw_ids = raw_ids_under_node(pool, *node_id, remaining).await?;
        for rid in raw_ids {
            if seen.insert(rid) && (ordered_raw_ids.len() as i64) < limit {
                ordered_raw_ids.push(rid);
            }
        }
    }

    // 3) Fetch those raw records (still active + owned), preserving curated rank.
    let mut records = fetch_raw_in_order(
        pool,
        user_id,
        &ordered_raw_ids,
        req.exclude_container_id.as_deref(),
    )
    .await?;

    // 3b) Reference bias: references describe the PRESENT ("what am I currently
    // maintaining"); records describe the past. For present-tense / "current"
    // queries, give a matching reference (a state_object terminal row) a small
    // additive boost so it floats above episodic noise — the design's "small
    // bias term". State_objects never enter the curated feed (they aren't
    // `working`), so this is the surface that makes references first-class in
    // retrieval. Shape is unchanged: references are ordinary RawRecordRow rows.
    if is_present_tense(&query) {
        let ref_hits = reference_hits(
            pool,
            user_id,
            &req,
            &scope_mode,
            &qvec,
            &model,
            emb_col,
            &query,
            limit,
        )
        .await?;
        let seen_ids: std::collections::HashSet<Uuid> = records.iter().map(|r| r.id).collect();
        let mut biased: Vec<RawRecordRow> = ref_hits
            .into_iter()
            .filter(|r| !seen_ids.contains(&r.id))
            .collect();
        biased.extend(records);
        records = biased;
        records.truncate(limit as usize);
    }

    // 4) Backstop: if the curated layer under-fills the page (e.g. curation
    // hasn't run yet, or a fresh raw record isn't summarized), top up with a
    // direct raw hybrid search — the same 0.7·cosine + 0.3·keyword fallback.
    if (records.len() as i64) < limit {
        let need = limit - records.len() as i64;
        let exclude: Vec<Uuid> = records.iter().map(|r| r.id).collect();
        let backstop = raw_hybrid(
            pool,
            user_id,
            &req,
            &scope_mode,
            &qvec,
            &model,
            emb_col,
            &query,
            need,
            &exclude,
        )
        .await?;
        records.extend(backstop);
    }

    let clip = clamp_assembled(&mut records, limit);
    Ok(AssembleResponse {
        records,
        degraded: false,
        warning: clip,
    })
}

/// A request is cross-mode (degraded) when it explicitly asks for `all` or names
/// more than one distinct mode. A single named mode, or an unset mode, is
/// single-mode (the latter resolves to the user's default register).
fn is_cross_mode(req: &AssembleRequest) -> bool {
    let mut names: Vec<&str> = Vec::new();
    if let Some(m) = req.mode.as_deref() {
        names.push(m);
    }
    if let Some(ms) = &req.modes {
        names.extend(ms.iter().map(|s| s.as_str()));
    }
    if names.iter().any(|n| n.eq_ignore_ascii_case("all")) {
        return true;
    }
    let distinct: std::collections::HashSet<&str> = names.into_iter().collect();
    distinct.len() > 1
}

/// Cross-mode retrieval: no shared vector geometry, so rank raw records across
/// every mode by keyword (BM25) + entity overlap + recency. Deliberately no
/// cosine term — the whole point is that the vectors aren't comparable. The
/// entity overlap uses the query's own extracted entities against `entity_index`.
async fn cross_mode_degraded(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    req: &AssembleRequest,
    query: &str,
    limit: i64,
) -> AppResult<Vec<RawRecordRow>> {
    let entities = nlp.extract_entities(query);
    // SQL literal for the sandbox opt-in — a compile-time-safe constant, not user input.
    let sandbox_gate = if req.include_sandbox { "TRUE" } else { "FALSE" };
    let sql = format!(
        r#"
        SELECT {COLS_R} FROM raw_records r
        LEFT JOIN (
            SELECT record_id, COUNT(*) AS hits
            FROM entity_index
            WHERE user_id = $2 AND entity = ANY($5)
            GROUP BY record_id
        ) ent ON ent.record_id = r.id
        WHERE r.user_id = $2
          AND ($3::text IS NULL OR r.project_id = $3)
          AND ($3::text IS NOT NULL OR {sandbox_gate} OR r.project_id IS DISTINCT FROM '{SANDBOX_PROJECT}')
          AND ($4::text IS NULL OR r.container_id = $4)
          AND ($7::text IS NULL OR r.container_id IS DISTINCT FROM $7)
          AND r.id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $2)
        ORDER BY (
            0.5 * ts_rank(r.content_tsv, plainto_tsquery('english', $1))
          + 0.3 * LEAST(COALESCE(ent.hits, 0) / 3.0, 1.0)
          + 0.2 * exp(- GREATEST(EXTRACT(EPOCH FROM (NOW() - r.event_time)) / 3600.0, 0) / {REC_S_HOURS})
        ) DESC, r.event_time DESC
        LIMIT $6
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(query)
        .bind(user_id)
        .bind(&req.project_id)
        .bind(&req.container_id)
        .bind(&entities)
        .bind(limit)
        .bind(&req.exclude_container_id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

/// The additive reference bias applied to a matching reference on present-tense
/// queries — a modest bump, not an override: a strongly on-topic record can
/// still outrank a weakly-matching reference. Sized to lift a reference above
/// same-topic episodic noise it would otherwise tie or trail.
const W_REF_BIAS: f64 = 0.15;

/// Cheap heuristic for "is this query asking about the current/present state?".
/// References answer the present-tense half of memory, so these cue words gate
/// the reference bias. Deliberately conservative — past-tense/"what happened"
/// queries get the normal record ranking with no reference boost.
fn is_present_tense(query: &str) -> bool {
    const CUES: &[&str] = &[
        "current",
        "currently",
        "right now",
        "now",
        "working on",
        "what am i",
        "what's on",
        "whats on",
        "what is on",
        "todo",
        "to-do",
        "to do",
        "status",
        "state of",
        "active",
        "in progress",
        "latest",
    ];
    let q = query.to_lowercase();
    CUES.iter().any(|c| q.contains(c))
}

/// Terminal state_object rows ranked by the hybrid score plus the reference
/// bias, scoped + owned. Only terminal (not-superseded) references are eligible,
/// so a superseded value never surfaces. Ordered best-first.
#[allow(clippy::too_many_arguments)]
async fn reference_hits(
    pool: &PgPool,
    user_id: &str,
    req: &AssembleRequest,
    scope_mode: &Option<String>,
    qvec: &[f32],
    model: &str,
    emb_col: &str,
    query: &str,
    limit: i64,
) -> AppResult<Vec<RawRecordRow>> {
    // emb_col is a trusted constant from embedding_col_for_dim; all values bound.
    // SQL literal for the sandbox opt-in — a compile-time-safe constant, not user input.
    let sandbox_gate = if req.include_sandbox { "TRUE" } else { "FALSE" };
    let sql = format!(
        r#"
        SELECT {COLS_R} FROM raw_records r
        LEFT JOIN raw_embeddings e ON e.record_id = r.id AND e.model = $1
        WHERE r.user_id = $2
          AND r.type = 'state_object'
          AND ($3::text IS NULL OR r.project_id = $3)
          AND ($3::text IS NOT NULL OR {sandbox_gate} OR r.project_id IS DISTINCT FROM '{SANDBOX_PROJECT}')
          AND ($4::text IS NULL OR r.container_id = $4)
          AND ($10::text IS NULL OR r.container_id IS DISTINCT FROM $10)
          AND r.mode IS NOT DISTINCT FROM $5
          AND r.id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $2)
        ORDER BY (
            $9
            + 0.7 * COALESCE(NULLIF(1 - (e.{emb_col} <=> $6), 'NaN'::float8), 0)
            + 0.3 * ts_rank(r.content_tsv, plainto_tsquery('english', $7))
        ) DESC, r.event_time DESC
        LIMIT $8
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(model)
        .bind(user_id)
        .bind(&req.project_id)
        .bind(&req.container_id)
        .bind(scope_mode)
        .bind(Vector::from(qvec.to_vec()))
        .bind(query)
        .bind(limit)
        .bind(W_REF_BIAS)
        .bind(&req.exclude_container_id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

/// Raw ids reachable from a curated node: walk `summarizes` down to the
/// level-0 nodes, then their `derived_from` edges to raw record ids. A level-0
/// node (no `summarizes` children) resolves straight through its own
/// `derived_from` edges.
///
/// Capped and recency-ordered: a summary near the root of a large tree can
/// reach most of the corpus, and an uncapped expansion once returned all of it
/// from a single ranked hit. The cap keeps one node from monopolizing a page,
/// and ordering by the raw record's event time means a truncated expansion
/// keeps the newest evidence rather than an arbitrary subset.
async fn raw_ids_under_node(pool: &PgPool, node_id: Uuid, cap: i64) -> AppResult<Vec<Uuid>> {
    if cap <= 0 {
        return Ok(Vec::new());
    }
    let sql = r#"
        WITH RECURSIVE tree AS (
            SELECT $1::uuid AS id
            UNION
            SELECT e.to_id
            FROM curated_edges e
            JOIN tree t ON e.from_id = t.id AND e.kind = 'summarizes'
        )
        SELECT src.to_id
        FROM (
            SELECT DISTINCT df.to_id
            FROM tree t
            JOIN curated_edges df ON df.from_id = t.id AND df.kind = 'derived_from'
        ) src
        JOIN raw_records r ON r.id = src.to_id
        ORDER BY r.event_time DESC
        LIMIT $2
    "#;
    let ids: Vec<Uuid> = sqlx::query_scalar(sql)
        .bind(node_id)
        .bind(cap)
        .fetch_all(pool)
        .await?;
    Ok(ids)
}

/// Fetch raw records by id, filtered to the caller + still-active (not
/// superseded, not expired), returned in the order the ids were given.
async fn fetch_raw_in_order(
    pool: &PgPool,
    user_id: &str,
    ids: &[Uuid],
    exclude_container: Option<&str>,
) -> AppResult<Vec<RawRecordRow>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    // The container exclusion is enforced HERE, at the last fetch, not only at
    // node selection — a summary node spans containers and carries no container
    // of its own, so its raw ids can smuggle the live conversation back in.
    let sql = format!(
        r#"
        SELECT {COLS} FROM raw_records
        WHERE user_id = $1
          AND id = ANY($2)
          AND ($3::text IS NULL OR container_id IS DISTINCT FROM $3)
          AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $1)
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(ids)
        .bind(exclude_container)
        .fetch_all(pool)
        .await?;
    // Re-order to the curated rank (SQL doesn't preserve ANY() order).
    let pos: std::collections::HashMap<Uuid, usize> =
        ids.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let mut rows = rows;
    rows.sort_by_key(|r| pos.get(&r.id).copied().unwrap_or(usize::MAX));
    Ok(rows)
}

/// Direct raw hybrid retrieval (0.7·cosine + 0.3·keyword) used as the backstop
/// under the curated feed. Excludes ids already returned from the curated pass.
#[allow(clippy::too_many_arguments)]
async fn raw_hybrid(
    pool: &PgPool,
    user_id: &str,
    req: &AssembleRequest,
    scope_mode: &Option<String>,
    qvec: &[f32],
    model: &str,
    emb_col: &str,
    query: &str,
    limit: i64,
    exclude: &[Uuid],
) -> AppResult<Vec<RawRecordRow>> {
    // SQL literal for the sandbox opt-in — a compile-time-safe constant, not user input.
    let sandbox_gate = if req.include_sandbox { "TRUE" } else { "FALSE" };
    // emb_col is a trusted constant from embedding_col_for_dim; all values bound.
    let sql = format!(
        r#"
        SELECT {COLS_R} FROM raw_records r
        LEFT JOIN raw_embeddings e ON e.record_id = r.id AND e.model = $1
        WHERE r.user_id = $2
          AND ($3::text IS NULL OR r.project_id = $3)
          AND ($3::text IS NOT NULL OR {sandbox_gate} OR r.project_id IS DISTINCT FROM '{SANDBOX_PROJECT}')
          AND ($4::text IS NULL OR r.container_id = $4)
          AND ($10::text IS NULL OR r.container_id IS DISTINCT FROM $10)
          AND r.mode IS NOT DISTINCT FROM $5
          AND r.id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $2)
          AND NOT (r.id = ANY($9))
        ORDER BY (
            0.7 * COALESCE(NULLIF(1 - (e.{emb_col} <=> $6), 'NaN'::float8), 0)
            + 0.3 * ts_rank(r.content_tsv, plainto_tsquery('english', $7))
        ) DESC, r.event_time DESC
        LIMIT $8
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(model)
        .bind(user_id)
        .bind(&req.project_id)
        .bind(&req.container_id)
        .bind(scope_mode)
        .bind(Vector::from(qvec.to_vec()))
        .bind(query)
        .bind(limit)
        .bind(exclude)
        .bind(&req.exclude_container_id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Summaries (RAPTOR view): the summary nodes for a scope, for the graph/catalog
// viz. Read-only window into the curated layer, behind the same bearer auth.
//   GET /records/summaries?level=&project=&mode=
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SummariesQuery {
    #[serde(default)]
    pub level: Option<i32>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Serialize, FromRow)]
pub struct SummaryNodeRow {
    pub id: Uuid,
    pub kind: String,
    pub content: String,
    pub level: i32,
    pub user_id: String,
    pub project_id: Option<String>,
    pub mode: Option<String>,
    pub importance: Option<f32>,
    pub event_time: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    /// Ids of the child nodes this summary covers (`summarizes` edges).
    pub children: Vec<Uuid>,
}

async fn summaries(
    State(state): State<AppState>,
    auth_user: AuthUser,
    axum::extract::Query(q): axum::extract::Query<SummariesQuery>,
) -> AppResult<Json<Vec<SummaryNodeRow>>> {
    Ok(Json(
        summaries_inner(&state.pool, &auth_user.user_id, q).await?,
    ))
}

pub(crate) async fn summaries_inner(
    pool: &PgPool,
    user_id: &str,
    q: SummariesQuery,
) -> AppResult<Vec<SummaryNodeRow>> {
    // Summary nodes for the caller, optionally filtered by level/project/mode,
    // each with its `summarizes` children folded into an array. `level=0`
    // returns none (level 0 is episodic/semantic, not summaries).
    let sql = r#"
        SELECT n.id, n.kind, n.content, n.level, n.user_id, n.project_id, n.mode,
               n.importance, n.event_time, n.created_at,
               COALESCE(
                   (SELECT array_agg(e.to_id)
                    FROM curated_edges e
                    WHERE e.from_id = n.id AND e.kind = 'summarizes'),
                   '{}'
               ) AS children
        FROM curated_nodes n
        WHERE n.kind = 'summary'
          AND n.user_id = $1
          AND ($2::int  IS NULL OR n.level = $2)
          AND ($3::text IS NULL OR n.project_id = $3)
          AND ($4::text IS NULL OR n.mode = $4)
        ORDER BY n.level DESC, n.created_at ASC
    "#;
    let rows = sqlx::query_as::<_, SummaryNodeRow>(sql)
        .bind(user_id)
        .bind(q.level)
        .bind(&q.project)
        .bind(&q.mode)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Get by id
// ---------------------------------------------------------------------------

async fn get_record(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<RawRecordRow>> {
    Ok(Json(
        get_record_inner(&state.pool, &auth_user.user_id, id).await?,
    ))
}

pub(crate) async fn get_record_inner(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
) -> AppResult<RawRecordRow> {
    let sql = format!("SELECT {COLS} FROM raw_records WHERE id = $1");
    let row = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(id)
        .fetch_optional(pool)
        .await?
        .ok_or_else(|| AppError::not_found(format!("record {id}")))?;
    if row.user_id != user_id {
        return Err(AppError::not_found(format!("record {id}")));
    }
    Ok(row)
}

// ---------------------------------------------------------------------------
// Lineage (walk the supersede chain — supersedes only; superseded status derived)
// ---------------------------------------------------------------------------

async fn lineage(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<Vec<RawRecordRow>>> {
    Ok(Json(
        lineage_inner(&state.pool, &auth_user.user_id, id).await?,
    ))
}

pub(crate) async fn lineage_inner(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
) -> AppResult<Vec<RawRecordRow>> {
    // Authorize on the anchor record first.
    let _ = get_record_inner(pool, user_id, id).await?;

    let sql = format!(
        r#"
        WITH RECURSIVE
        back AS (
            SELECT {COLS} FROM raw_records WHERE id = $1
            UNION ALL
            SELECT {COLS_R} FROM raw_records r JOIN back b ON r.id = b.supersedes AND r.user_id = $2
        ),
        fwd AS (
            SELECT {COLS} FROM raw_records WHERE id = $1
            UNION ALL
            SELECT {COLS_R} FROM raw_records r JOIN fwd f ON r.supersedes = f.id AND r.user_id = $2
        )
        SELECT {COLS} FROM (
            SELECT {COLS} FROM back UNION SELECT {COLS} FROM fwd
        ) u
        ORDER BY ingest_time ASC
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(id)
        .bind(user_id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Derivations (glass-box): curated nodes derived from this raw id. Read-only
// window into the NEW curation layer — the raw door itself is unchanged.
// ---------------------------------------------------------------------------

async fn derivations(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<Vec<crate::curation::DerivationRow>>> {
    // Authorize on the raw record first (404s a foreign id), then list.
    let _ = get_record_inner(&state.pool, &auth_user.user_id, id).await?;
    Ok(Json(
        crate::curation::derivations_of(&state.pool, &auth_user.user_id, id).await?,
    ))
}

// ---------------------------------------------------------------------------
// Curation rebuild (admin/glass-box): wipe + re-derive the caller's curated
// layer from raw. Proves the "curation is rebuildable from raw" contract.
// Same bearer auth as every other door; scoped to the caller's user.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct RebuildResponse {
    pub promoted: i64,
    pub distilled: i64,
    pub clusters_seen: i64,
    pub distill_skipped: bool,
    pub summarized: i64,
    pub max_level: i32,
}

async fn rebuild_curation(
    State(state): State<AppState>,
    auth_user: AuthUser,
) -> AppResult<Json<RebuildResponse>> {
    let stats = crate::curation::rebuild(&state.pool, &*state.nlp, &auth_user.user_id).await?;
    Ok(Json(RebuildResponse {
        promoted: stats.promoted,
        distilled: stats.distilled,
        clusters_seen: stats.clusters_seen,
        distill_skipped: stats.skipped_distill,
        summarized: stats.summarized,
        max_level: stats.max_level,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};

    #[derive(Clone)]
    struct StubNlp;

    #[async_trait]
    impl NlpService for StubNlp {
        fn provider_name(&self) -> &'static str {
            "stub"
        }
        fn provider_can_distill(&self) -> bool {
            false
        }
        fn embedder_model_name(&self) -> &str {
            "stub-embedder"
        }
        fn embedder_dimension(&self) -> usize {
            384
        }
        async fn embed_one(&self, _text: &str) -> Result<Vec<f32>, AppError> {
            Ok(vec![0.1_f32; 384]) // non-zero to keep cosine defined
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| vec![0.1_f32; 384]).collect())
        }
        fn extract_entities(&self, _text: &str) -> Vec<String> {
            Vec::new()
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            _e: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("stub".into()))
        }
    }

    /// A stub whose `embed_for_mode` honors the embedder key's real dimension
    /// (via `model_for_key`) so the per-mode embedding column routing is
    /// exercised for real. Vectors are uniform (cosine is constant), so the mode
    /// SCOPE + column dim — not similarity — decide what a query retrieves.
    #[derive(Clone)]
    struct ModeAwareStub;

    #[async_trait]
    impl NlpService for ModeAwareStub {
        fn provider_name(&self) -> &'static str {
            "mode-aware-stub"
        }
        fn provider_can_distill(&self) -> bool {
            false
        }
        fn embedder_model_name(&self) -> &str {
            "sentence-transformers/all-MiniLM-L6-v2"
        }
        fn embedder_dimension(&self) -> usize {
            384
        }
        async fn embed_one(&self, _text: &str) -> Result<Vec<f32>, AppError> {
            Ok(vec![0.1_f32; 384])
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| vec![0.1_f32; 384]).collect())
        }
        async fn embed_for_mode(
            &self,
            embedder_key: &str,
            _text: &str,
        ) -> Result<(usize, Vec<f32>), AppError> {
            let dim = flashback_nlp::model_for_key(embedder_key)
                .map(|(_, d)| d)
                .unwrap_or(384);
            Ok((dim, vec![0.1_f32; dim]))
        }
        fn extract_entities(&self, text: &str) -> Vec<String> {
            flashback_nlp::extract_entities(text)
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            _e: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("stub".into()))
        }
    }

    fn req(content: &str) -> IngestRecordRequest {
        IngestRecordRequest {
            r#type: "document".into(),
            content: content.into(),
            event_time: None,
            source: "test".into(),
            source_ref: None,
            project_id: Some("health".into()),
            container_id: None,
            mode: None,
            importance: None,
            supersedes: None,
            payload: None,
        }
    }

    fn q() -> QueryRecordsRequest {
        QueryRecordsRequest {
            project_id: None,
            container_id: None,
            mode: None,
            r#type: None,
            since: None,
            until: None,
            limit: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn sandbox_scope_never_reaches_an_unscoped_assembly(pool: PgPool) {
        // One real record, one sandbox record, same topic; curate so the
        // exclusion is exercised through the curated layer too, not just raw.
        ingest_record(
            &pool,
            &ModeAwareStub,
            "leslie",
            IngestRecordRequest {
                project_id: None,
                ..req("the deploy target for the pgvector service is staging")
            },
        )
        .await
        .unwrap();
        ingest_record(
            &pool,
            &ModeAwareStub,
            "leslie",
            IngestRecordRequest {
                project_id: Some(SANDBOX_PROJECT.into()),
                ..req("sandbox chatter: the deploy target for the pgvector service is bananas")
            },
        )
        .await
        .unwrap();
        crate::curation::rebuild(&pool, &ModeAwareStub, "leslie")
            .await
            .unwrap();

        let base = AssembleRequest {
            include_sandbox: false,
            project_id: None,
            container_id: None,
            mode: None,
            modes: None,
            exclude_container_id: None,
            query: Some("deploy target".into()),
            limit: Some(10),
        };

        // Unscoped: the sandbox is invisible — a host never retrieves it.
        let out = assemble_inner(&pool, &ModeAwareStub, "leslie", base.clone())
            .await
            .unwrap();
        assert!(!out.records.is_empty());
        assert!(
            out.records
                .iter()
                .all(|r| r.project_id.as_deref() != Some(SANDBOX_PROJECT)),
            "sandbox record leaked into an unscoped assembly"
        );

        // Opt-in: the playground sees both spaces.
        let out = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            AssembleRequest {
                include_sandbox: true,
                ..base.clone()
            },
        )
        .await
        .unwrap();
        assert!(out
            .records
            .iter()
            .any(|r| r.project_id.as_deref() == Some(SANDBOX_PROJECT)));

        // Explicit sandbox scope: only the sandbox.
        let out = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            AssembleRequest {
                project_id: Some(SANDBOX_PROJECT.into()),
                ..base
            },
        )
        .await
        .unwrap();
        assert!(!out.records.is_empty());
        assert!(out
            .records
            .iter()
            .all(|r| r.project_id.as_deref() == Some(SANDBOX_PROJECT)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn assemble_limit_binds_even_through_a_summary_tree(pool: PgPool) {
        // Eight same-topic documents curate into episodic nodes plus a summary
        // covering all of them. A ranked summary hit used to expand to its
        // whole subtree AFTER the limit check, so one node returned the whole
        // corpus; the limit must bind during expansion.
        for i in 0..8 {
            ingest_record(
                &pool,
                &ModeAwareStub,
                "leslie",
                IngestRecordRequest {
                    project_id: None,
                    ..req(&format!(
                        "note {i}: the deploy target for the pgvector service is staging"
                    ))
                },
            )
            .await
            .unwrap();
        }
        crate::curation::rebuild(&pool, &ModeAwareStub, "leslie")
            .await
            .unwrap();
        let summaries: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes WHERE kind = 'summary' AND user_id = 'leslie'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(summaries > 0, "test needs a summary tree to be meaningful");

        let out = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: None,
                modes: None,
                exclude_container_id: None,
                query: Some("deploy target".into()),
                limit: Some(3),
            },
        )
        .await
        .unwrap();
        assert!(
            out.records.len() <= 3,
            "limit=3 must bind, got {} records",
            out.records.len()
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn assemble_clips_to_the_character_budget_with_a_warning(pool: PgPool) {
        // Three 25k-char documents against the default 60k budget: two fit,
        // the third is clipped and the response says so.
        for i in 0..3 {
            let body = format!("filler document {i} ").repeat(1_400); // ~25k chars
            ingest_record(&pool, &ModeAwareStub, "leslie", req(&body))
                .await
                .unwrap();
        }
        let out = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: Some("health".into()),
                container_id: None,
                mode: None,
                modes: None,
                exclude_container_id: None,
                query: Some("filler document".into()),
                limit: Some(10),
            },
        )
        .await
        .unwrap();
        assert_eq!(out.records.len(), 2, "third record must be clipped");
        let warning = out.warning.expect("clip must be announced");
        assert!(warning.contains("clipped"), "got: {warning}");
        let total: usize = out.records.iter().map(|r| r.content.chars().count()).sum();
        assert!(total <= 60_000, "budget must hold, got {total} chars");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_and_query_roundtrip(pool: PgPool) {
        let out = ingest_record(&pool, &StubNlp, "leslie", req("took 5mg lisinopril"))
            .await
            .unwrap();
        let rows = query_records_inner(&pool, "leslie", q()).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, out.id);
        assert_eq!(rows[0].content, "took 5mg lisinopril");
        assert_eq!(rows[0].content_hash.len(), 32); // md5 hex

        // The derived embedding landed in the separate table.
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM raw_embeddings WHERE record_id = $1")
            .bind(out.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn raw_records_are_immutable(pool: PgPool) {
        let out = ingest_record(&pool, &StubNlp, "leslie", req("original"))
            .await
            .unwrap();
        let upd = sqlx::query("UPDATE raw_records SET content = 'tampered' WHERE id = $1")
            .bind(out.id)
            .execute(&pool)
            .await;
        assert!(upd.is_err(), "UPDATE on raw_records should be blocked");
        let del = sqlx::query("DELETE FROM raw_records WHERE id = $1")
            .bind(out.id)
            .execute(&pool)
            .await;
        assert!(del.is_err(), "DELETE on raw_records should be blocked");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_forward_pointer_hides_old_keeps_it(pool: PgPool) {
        let v1 = ingest_record(&pool, &StubNlp, "leslie", req("weight 180"))
            .await
            .unwrap();
        let mut r2 = req("weight 178");
        r2.supersedes = Some(v1.id);
        let v2 = ingest_record(&pool, &StubNlp, "leslie", r2).await.unwrap();

        let active = query_records_inner(&pool, "leslie", q()).await.unwrap();
        assert_eq!(active.iter().map(|r| r.id).collect::<Vec<_>>(), vec![v2.id]);

        let old = get_record_inner(&pool, "leslie", v1.id).await.unwrap();
        assert_eq!(old.content, "weight 180");
        assert_eq!(old.supersedes, None); // never mutated

        let line = lineage_inner(&pool, "leslie", v2.id).await.unwrap();
        assert_eq!(
            line.iter().map(|r| r.id).collect::<Vec<_>>(),
            vec![v1.id, v2.id]
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn lineage_keeps_branched_siblings(pool: PgPool) {
        let v1 = ingest_record(&pool, &StubNlp, "leslie", req("weight 180"))
            .await
            .unwrap();
        let mut ra = req("weight 178");
        ra.supersedes = Some(v1.id);
        let a = ingest_record(&pool, &StubNlp, "leslie", ra).await.unwrap();
        let mut rb = req("weight 179");
        rb.supersedes = Some(v1.id);
        let b = ingest_record(&pool, &StubNlp, "leslie", rb).await.unwrap();
        let mut got: Vec<_> = lineage_inner(&pool, "leslie", v1.id)
            .await
            .unwrap()
            .iter()
            .map(|r| r.id)
            .collect();
        got.sort();
        let mut want = vec![v1.id, a.id, b.id];
        want.sort();
        assert_eq!(got, want); // all three; the branch sibling is not dropped
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn truncate_is_blocked(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("x"))
            .await
            .unwrap();
        let err = sqlx::query("TRUNCATE raw_records CASCADE")
            .execute(&pool)
            .await;
        assert!(err.is_err(), "TRUNCATE on raw_records must be blocked");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_hybrid_ranks_keyword_over_recency(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("weighed 180 lbs"))
            .await
            .unwrap();
        ingest_record(
            &pool,
            &StubNlp,
            "leslie",
            req("discussed lisinopril dosage"),
        )
        .await
        .unwrap();
        ingest_record(&pool, &StubNlp, "leslie", req("ate lunch"))
            .await
            .unwrap();
        // Stub embeddings are uniform, so the vector term is constant and the
        // keyword term decides — the lisinopril record must come first.
        let out = assemble_inner(
            &pool,
            &StubNlp,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: None,
                modes: None,
                query: Some("lisinopril".into()),
                limit: None,
                exclude_container_id: None,
            },
        )
        .await
        .unwrap();
        assert!(out.records[0].content.contains("lisinopril"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_empty_query_is_recency(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("first"))
            .await
            .unwrap();
        let latest = ingest_record(&pool, &StubNlp, "leslie", req("second"))
            .await
            .unwrap();
        let out = assemble_inner(
            &pool,
            &StubNlp,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: None,
                modes: None,
                query: None,
                limit: None,
                exclude_container_id: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(out.records[0].id, latest.id); // most recent first
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn scoped_and_isolated_by_user(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("mine"))
            .await
            .unwrap();
        ingest_record(&pool, &StubNlp, "bob", req("theirs"))
            .await
            .unwrap();
        assert_eq!(
            query_records_inner(&pool, "leslie", q())
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            query_records_inner(&pool, "bob", q()).await.unwrap().len(),
            1
        );
        let leslies = query_records_inner(&pool, "leslie", q()).await.unwrap()[0].id;
        assert!(get_record_inner(&pool, "bob", leslies).await.is_err());
    }

    fn imp(content: &str, source_ref: Option<&str>) -> ImportRecord {
        ImportRecord {
            r#type: "conversation".into(),
            content: content.into(),
            event_time: None,
            source: "chatgpt".into(),
            source_ref: source_ref.map(|s| s.into()),
            project_id: None,
            container_id: None,
            mode: None,
            importance: None,
            payload: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn import_bulk_then_dedup_on_reimport(pool: PgPool) {
        let batch = ImportRequest {
            records: vec![imp("chat one", Some("m1")), imp("chat two", Some("m2"))],
        };
        let out = import_records_inner(&pool, &StubNlp, "leslie", batch)
            .await
            .unwrap();
        assert_eq!((out.imported, out.skipped), (2, 0));

        // Re-import the same source_refs -> all deduped, nothing duplicated.
        let batch2 = ImportRequest {
            records: vec![
                imp("chat one", Some("m1")),
                imp("chat two edited", Some("m2")),
            ],
        };
        let out2 = import_records_inner(&pool, &StubNlp, "leslie", batch2)
            .await
            .unwrap();
        assert_eq!((out2.imported, out2.skipped), (0, 2));
        assert_eq!(
            query_records_inner(&pool, "leslie", q())
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn import_without_source_ref_always_inserts(pool: PgPool) {
        let b1 = ImportRequest {
            records: vec![imp("ok", None)],
        };
        let b2 = ImportRequest {
            records: vec![imp("ok", None)],
        };
        assert_eq!(
            import_records_inner(&pool, &StubNlp, "leslie", b1)
                .await
                .unwrap()
                .imported,
            1
        );
        assert_eq!(
            import_records_inner(&pool, &StubNlp, "leslie", b2)
                .await
                .unwrap()
                .imported,
            1
        );
        assert_eq!(
            query_records_inner(&pool, "leslie", q())
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn imported_records_are_embedded_and_searchable(pool: PgPool) {
        let batch = ImportRequest {
            records: vec![imp("discussed lisinopril", Some("x1"))],
        };
        assert_eq!(
            import_records_inner(&pool, &StubNlp, "leslie", batch)
                .await
                .unwrap()
                .imported,
            1
        );
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM raw_embeddings")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 1);
        let ctx = assemble_inner(
            &pool,
            &StubNlp,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: None,
                modes: None,
                query: Some("lisinopril".into()),
                limit: None,
                exclude_container_id: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(ctx.records.len(), 1);
    }

    // -- summary-first + decay-weighted retrieval --------------------------

    /// The default (general) register's embedder key — the model + column a
    /// mode-less/general record's derived embedding uses. Records ingested with
    /// no explicit mode resolve to `general`, so a curated node deriving from
    /// them shares this mode + model for retrieval to match.
    const GENERAL_MODEL: &str = "sentence-transformers/all-MiniLM-L6-v2";

    /// Insert a level-0 curated node deriving from `raw_id`, in the `general`
    /// register (matching a mode-less raw ingest), embed it (uniform via StubNlp
    /// so cosine is constant across nodes), and return the node id.
    async fn curated_node_from(pool: &PgPool, user_id: &str, raw_id: Uuid, content: &str) -> Uuid {
        let node_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO curated_nodes (id, kind, content, level, user_id, mode, importance, event_time) \
             VALUES ($1, 'episodic', $2, 0, $3, 'general', 0.5, NOW())",
        )
        .bind(node_id)
        .bind(content)
        .bind(user_id)
        .execute(pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO curated_edges (from_id, to_id, kind) VALUES ($1, $2, 'derived_from')",
        )
        .bind(node_id)
        .bind(raw_id)
        .execute(pool)
        .await
        .unwrap();
        let emb = StubNlp.embed_one(content).await.unwrap();
        sqlx::query(
            "INSERT INTO curated_embeddings (node_id, model, embedding) VALUES ($1, $2, $3)",
        )
        .bind(node_id)
        .bind(GENERAL_MODEL)
        .bind(Vector::from(emb))
        .execute(pool)
        .await
        .unwrap();
        node_id
    }

    /// The echo filter: a caller's own conversation must not come back as
    /// "memory" — the host already has the live thread in its window, and a
    /// query's nearest neighbour is otherwise its own previous turn.
    #[sqlx::test(migrations = "../../migrations")]
    async fn assemble_excludes_the_callers_own_container(pool: PgPool) {
        let nlp = StubNlp;
        for (container, text) in [
            ("conv-live", "what is on the release checklist right now"),
            ("conv-old", "the release needs the changelog written first"),
        ] {
            ingest_record(
                &pool,
                &nlp,
                "alice",
                IngestRecordRequest {
                    r#type: "conversation".into(),
                    content: text.into(),
                    event_time: None,
                    source: "test".into(),
                    source_ref: None,
                    project_id: None,
                    container_id: Some(container.into()),
                    mode: None,
                    importance: None,
                    supersedes: None,
                    payload: None,
                },
            )
            .await
            .unwrap();
        }

        // Once curated, a summary node spans BOTH conversations and carries no
        // container of its own — the exclusion must hold through that path too,
        // not just on direct raw hits.
        crate::curation::rebuild(&pool, &nlp, "alice")
            .await
            .unwrap();

        let mut req = ctx_req("release checklist");
        req.exclude_container_id = Some("conv-live".into());
        let out = assemble_inner(&pool, &nlp, "alice", req).await.unwrap();

        assert!(!out.records.is_empty(), "the other conversation should hit");
        assert!(
            out.records
                .iter()
                .all(|r| r.container_id.as_deref() != Some("conv-live")),
            "own-container rows must never come back as memory"
        );
    }

    fn ctx_req(query: &str) -> AssembleRequest {
        AssembleRequest {
            include_sandbox: false,
            project_id: None,
            container_id: None,
            mode: None,
            modes: None,
            query: Some(query.into()),
            limit: None,
            exclude_container_id: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_prefers_curated_summary_sources_first(pool: PgPool) {
        // One raw record has a curated node (the summarized/curated feed); another
        // raw record is only in the raw layer (backstop). Both match the query on
        // keyword; the curated-backed one must lead.
        let curated_raw = ingest_record(&pool, &StubNlp, "leslie", req("deploy target is staging"))
            .await
            .unwrap();
        let raw_only = ingest_record(&pool, &StubNlp, "leslie", req("deploy target notes"))
            .await
            .unwrap();
        curated_node_from(&pool, "leslie", curated_raw.id, "deploy target is staging").await;

        let out = assemble_inner(&pool, &StubNlp, "leslie", ctx_req("deploy target"))
            .await
            .unwrap();
        assert!(out.records.len() >= 2);
        // The curated-backed raw record leads; the raw-only one is the backstop.
        assert_eq!(out.records[0].id, curated_raw.id);
        assert!(out.records.iter().any(|r| r.id == raw_only.id));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_decays_stale_below_fresh_of_equal_similarity(pool: PgPool) {
        // Two curated nodes with identical embedding + event_time; only decay
        // (last_access) differs. The stale one must rank below the fresh one.
        let raw_old = ingest_record(&pool, &StubNlp, "leslie", req("deploy target one"))
            .await
            .unwrap();
        let raw_new = ingest_record(&pool, &StubNlp, "leslie", req("deploy target two"))
            .await
            .unwrap();
        let n_old = curated_node_from(&pool, "leslie", raw_old.id, "deploy target").await;
        let n_new = curated_node_from(&pool, "leslie", raw_new.id, "deploy target").await;

        // Pin identical event_time so the recency term is equal; only decay differs.
        sqlx::query("UPDATE curated_nodes SET event_time = NOW() WHERE id IN ($1, $2)")
            .bind(n_old)
            .bind(n_new)
            .execute(&pool)
            .await
            .unwrap();
        // Old: last accessed 60 days ago (deeply decayed). New: just now.
        sqlx::query(
            "INSERT INTO ref_weights (ref_id, ref_kind, weight, last_access, decay_class) \
             VALUES ($1, 'curated', 1.0, NOW() - INTERVAL '60 days', 'default')",
        )
        .bind(n_old)
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO ref_weights (ref_id, ref_kind, weight, last_access, decay_class) \
             VALUES ($1, 'curated', 1.0, NOW(), 'default')",
        )
        .bind(n_new)
        .execute(&pool)
        .await
        .unwrap();

        let out = assemble_inner(&pool, &StubNlp, "leslie", ctx_req("deploy target"))
            .await
            .unwrap();
        let pos_new = out.records.iter().position(|r| r.id == raw_new.id).unwrap();
        let pos_old = out.records.iter().position(|r| r.id == raw_old.id).unwrap();
        assert!(
            pos_new < pos_old,
            "fresh node must outrank the decayed one (new@{pos_new}, old@{pos_old})"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_access_resets_decay_clock(pool: PgPool) {
        let raw = ingest_record(&pool, &StubNlp, "leslie", req("deploy target reset"))
            .await
            .unwrap();
        let node = curated_node_from(&pool, "leslie", raw.id, "deploy target").await;
        // Seed a stale last_access.
        sqlx::query(
            "INSERT INTO ref_weights (ref_id, ref_kind, weight, last_access, decay_class) \
             VALUES ($1, 'curated', 1.0, NOW() - INTERVAL '90 days', 'default')",
        )
        .bind(node)
        .execute(&pool)
        .await
        .unwrap();

        // A retrieval that returns the node must bump its clock back to ~now.
        let _ = assemble_inner(&pool, &StubNlp, "leslie", ctx_req("deploy target"))
            .await
            .unwrap();

        let idle_secs: f64 = sqlx::query_scalar(
            "SELECT EXTRACT(EPOCH FROM (NOW() - last_access))::float8 FROM ref_weights WHERE ref_id = $1",
        )
        .bind(node)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            idle_secs < 5.0,
            "last_access should be reset to ~now, was {idle_secs}s ago"
        );
        // And the decay weight is back to full retention.
        let w = crate::decay::decay_weight(&pool, node).await.unwrap();
        assert!((w - 1.0).abs() < 1e-3, "retention reset to 1.0, got {w}");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_never_deletes_and_leaves_raw_intact(pool: PgPool) {
        let raw = ingest_record(&pool, &StubNlp, "leslie", req("deploy target immutable"))
            .await
            .unwrap();
        let node = curated_node_from(&pool, "leslie", raw.id, "deploy target").await;
        sqlx::query(
            "INSERT INTO ref_weights (ref_id, ref_kind, weight, last_access, decay_class) \
             VALUES ($1, 'curated', 1.0, NOW() - INTERVAL '365 days', 'ephemeral')",
        )
        .bind(node)
        .execute(&pool)
        .await
        .unwrap();

        let nodes_before = count_curated(&pool).await;
        let raw_before = count_raw(&pool).await;

        // Even a deeply-decayed node is still returned (demoted, never deleted).
        let out = assemble_inner(&pool, &StubNlp, "leslie", ctx_req("deploy target"))
            .await
            .unwrap();
        assert!(out.records.iter().any(|r| r.id == raw.id));

        assert_eq!(count_curated(&pool).await, nodes_before, "nothing deleted");
        assert_eq!(count_raw(&pool).await, raw_before, "raw untouched");
        // ref_weights row still there (tombstone-not-delete).
        let rw: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ref_weights WHERE ref_id = $1")
            .bind(node)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(rw, 1);
        // Raw content byte-for-byte unchanged.
        let content: String = sqlx::query_scalar("SELECT content FROM raw_records WHERE id = $1")
            .bind(raw.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(content, "deploy target immutable");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_scope_isolated_across_users(pool: PgPool) {
        let mine = ingest_record(&pool, &StubNlp, "leslie", req("deploy target mine"))
            .await
            .unwrap();
        let theirs = ingest_record(&pool, &StubNlp, "bob", req("deploy target theirs"))
            .await
            .unwrap();
        curated_node_from(&pool, "leslie", mine.id, "deploy target").await;
        curated_node_from(&pool, "bob", theirs.id, "deploy target").await;

        let out = assemble_inner(&pool, &StubNlp, "leslie", ctx_req("deploy target"))
            .await
            .unwrap();
        assert!(out.records.iter().all(|r| r.user_id == "leslie"));
        assert!(out.records.iter().any(|r| r.id == mine.id));
        assert!(!out.records.iter().any(|r| r.id == theirs.id));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn summaries_endpoint_returns_summary_nodes_with_children(pool: PgPool) {
        let raw = ingest_record(&pool, &StubNlp, "leslie", req("deploy target summary src"))
            .await
            .unwrap();
        let child = curated_node_from(&pool, "leslie", raw.id, "deploy target").await;
        // A level-1 summary over that child.
        let summary_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO curated_nodes (id, kind, content, level, user_id) \
             VALUES ($1, 'summary', 'rollup', 1, 'leslie')",
        )
        .bind(summary_id)
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO curated_edges (from_id, to_id, kind) VALUES ($1, $2, 'summarizes')",
        )
        .bind(summary_id)
        .bind(child)
        .execute(&pool)
        .await
        .unwrap();

        let all = summaries_inner(
            &pool,
            "leslie",
            SummariesQuery {
                level: None,
                project: None,
                mode: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, summary_id);
        assert_eq!(all[0].level, 1);
        assert_eq!(all[0].children, vec![child]);

        // level filter: level 0 has no summaries.
        let lvl0 = summaries_inner(
            &pool,
            "leslie",
            SummariesQuery {
                level: Some(0),
                project: None,
                mode: None,
            },
        )
        .await
        .unwrap();
        assert!(lvl0.is_empty());

        // Scope isolation: bob sees none of leslie's summaries.
        let bobs = summaries_inner(
            &pool,
            "bob",
            SummariesQuery {
                level: None,
                project: None,
                mode: None,
            },
        )
        .await
        .unwrap();
        assert!(bobs.is_empty());
    }

    // -- reference retrieval bias ------------------------------------------

    #[test]
    fn present_tense_detects_current_queries() {
        assert!(is_present_tense("what am I working on"));
        assert!(is_present_tense("what's on my todo list"));
        assert!(is_present_tense("current project status"));
        // Past-tense / "what happened" gets no reference boost.
        assert!(!is_present_tense("what did I eat last tuesday"));
        assert!(!is_present_tense("the meeting we had yesterday"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_biases_reference_above_episodic_noise(pool: PgPool) {
        // Episodic noise on the same topic.
        for note in [
            "started working on the deploy pipeline",
            "the deploy pipeline had a flaky test",
            "notes about the deploy pipeline meeting",
        ] {
            ingest_record(&pool, &StubNlp, "leslie", req(note))
                .await
                .unwrap();
        }
        // A reference (state_object) capturing the current working state.
        let reference = crate::references::put_value_inner(
            &pool,
            &StubNlp,
            "leslie",
            "todo_list",
            "deploy_pipeline",
            crate::references::PutValueRequest {
                data: serde_json::json!({ "items": [{ "text": "finish the deploy pipeline" }] }),
                project_id: Some("health".into()),
                container_id: None,
                importance: None,
            },
        )
        .await
        .unwrap();

        // Present-tense query: the reference must lead the episodic noise.
        let out = assemble_inner(
            &pool,
            &StubNlp,
            "leslie",
            ctx_req("what am I working on deploy pipeline"),
        )
        .await
        .unwrap();
        assert_eq!(
            out.records[0].id, reference.id,
            "the reference should be biased above episodic noise for a present-tense query"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_no_reference_bias_on_past_tense_query(pool: PgPool) {
        // A reference exists but the query is past-tense, so no bias applies and
        // the response shape is unchanged (references compete on the normal score).
        crate::references::put_value_inner(
            &pool,
            &StubNlp,
            "leslie",
            "todo_list",
            "deploy_pipeline",
            crate::references::PutValueRequest {
                data: serde_json::json!({ "items": [] }),
                project_id: Some("health".into()),
                container_id: None,
                importance: None,
            },
        )
        .await
        .unwrap();
        let noise = ingest_record(
            &pool,
            &StubNlp,
            "leslie",
            req("deploy pipeline retro from last week"),
        )
        .await
        .unwrap();

        let out = assemble_inner(
            &pool,
            &StubNlp,
            "leslie",
            ctx_req("what happened in the deploy pipeline retro"),
        )
        .await
        .unwrap();
        // No panic, records returned, and the past-tense episodic hit is present.
        assert!(out.records.iter().any(|r| r.id == noise.id));
    }

    async fn count_curated(pool: &PgPool) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM curated_nodes")
            .fetch_one(pool)
            .await
            .unwrap()
    }

    async fn count_raw(pool: &PgPool) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM raw_records")
            .fetch_one(pool)
            .await
            .unwrap()
    }

    // -- modes (cognitive registers) ---------------------------------------

    /// An ingest request in a specific mode, no project scope (so the mode is
    /// the only scoping axis under test).
    fn req_in_mode(content: &str, mode: &str) -> IngestRecordRequest {
        IngestRecordRequest {
            r#type: "document".into(),
            content: content.into(),
            event_time: None,
            source: "test".into(),
            source_ref: None,
            project_id: None,
            container_id: None,
            mode: Some(mode.into()),
            importance: None,
            supersedes: None,
            payload: None,
        }
    }

    fn ctx_mode(query: &str, mode: &str) -> AssembleRequest {
        AssembleRequest {
            include_sandbox: false,
            project_id: None,
            container_id: None,
            mode: Some(mode.into()),
            modes: None,
            query: Some(query.into()),
            limit: None,
            exclude_container_id: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn code_mode_embeds_in_its_column_and_scopes_retrieval(pool: PgPool) {
        // Ingest a record in `code` mode: it resolves to the jina-code embedder
        // (768-dim), so its vector lands in embedding_768 and the row is tagged
        // mode='code'.
        let coded = ingest_record(
            &pool,
            &ModeAwareStub,
            "leslie",
            req_in_mode("Arc<Mutex<T>>", "code"),
        )
        .await
        .unwrap();

        // The derived embedding went into the 768 column, not the 384 default.
        let row: (Option<i32>, Option<i32>, Option<i32>) = sqlx::query_as(
            "SELECT (embedding IS NOT NULL)::int, (embedding_768 IS NOT NULL)::int, \
                    (embedding_1024 IS NOT NULL)::int \
             FROM raw_embeddings WHERE record_id = $1",
        )
        .bind(coded.id)
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            row,
            (Some(0), Some(1), Some(0)),
            "code record uses embedding_768"
        );
        let mode: Option<String> = sqlx::query_scalar("SELECT mode FROM raw_records WHERE id = $1")
            .bind(coded.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(mode.as_deref(), Some("code"));

        // A code-scoped query finds it (same 768 geometry, same mode scope).
        let hit = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            ctx_mode("Arc Mutex", "code"),
        )
        .await
        .unwrap();
        assert!(!hit.degraded);
        assert!(
            hit.records.iter().any(|r| r.id == coded.id),
            "code query must retrieve the code record"
        );

        // A general-scoped query must NOT — different mode scope AND different
        // (384) geometry / column.
        let miss = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            ctx_mode("Arc Mutex", "general"),
        )
        .await
        .unwrap();
        assert!(
            !miss.records.iter().any(|r| r.id == coded.id),
            "general query must not retrieve the code record"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn cross_mode_query_is_keyword_degraded_with_flag(pool: PgPool) {
        // Two records in two different registers/geometries.
        ingest_record(
            &pool,
            &ModeAwareStub,
            "leslie",
            req_in_mode("deploy target is staging", "code"),
        )
        .await
        .unwrap();
        ingest_record(
            &pool,
            &ModeAwareStub,
            "leslie",
            req_in_mode("felt calm about the deploy today", "journal"),
        )
        .await
        .unwrap();

        // A cross-mode ("all") request can't compare cosine across dims — it
        // degrades to keyword/entity/recency and flags it.
        let out = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: Some("all".into()),
                modes: None,
                query: Some("deploy".into()),
                limit: None,
                exclude_container_id: None,
            },
        )
        .await
        .unwrap();
        assert!(out.degraded, "cross-mode request must set degraded");
        assert!(
            out.warning.is_some(),
            "cross-mode request must carry a warning"
        );
        // Keyword recall still works across both registers — the code record
        // (keyword 'deploy') surfaces, and the search spans modes.
        assert!(
            out.records
                .iter()
                .any(|r| r.content.contains("deploy target")),
            "keyword recall should surface the matching record across modes"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn cross_mode_via_modes_list_also_degrades(pool: PgPool) {
        ingest_record(&pool, &ModeAwareStub, "leslie", req_in_mode("x", "code"))
            .await
            .unwrap();
        // Two distinct named modes is also a cross-mode request.
        let out = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: None,
                modes: Some(vec!["code".into(), "journal".into()]),
                query: Some("x".into()),
                limit: None,
                exclude_container_id: None,
            },
        )
        .await
        .unwrap();
        assert!(out.degraded);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn mode_records_are_scope_isolated_by_user(pool: PgPool) {
        let mine = ingest_record(
            &pool,
            &ModeAwareStub,
            "leslie",
            req_in_mode("Arc Mutex mine", "code"),
        )
        .await
        .unwrap();
        ingest_record(
            &pool,
            &ModeAwareStub,
            "bob",
            req_in_mode("Arc Mutex theirs", "code"),
        )
        .await
        .unwrap();

        let out = assemble_inner(
            &pool,
            &ModeAwareStub,
            "leslie",
            ctx_mode("Arc Mutex", "code"),
        )
        .await
        .unwrap();
        assert!(out.records.iter().all(|r| r.user_id == "leslie"));
        assert!(out.records.iter().any(|r| r.id == mine.id));
    }

    // -- resource bounds: limit is clamped, never unbounded ----------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn query_limit_is_clamped_to_max(pool: PgPool) {
        // Ingest a handful; an absurd limit must not error or over-return — the
        // handler clamps to at most 1000.
        for i in 0..5 {
            ingest_record(&pool, &StubNlp, "leslie", req(&format!("note {i}")))
                .await
                .unwrap();
        }
        let mut oversized = q();
        oversized.limit = Some(1_000_000_000);
        let rows = query_records_inner(&pool, "leslie", oversized)
            .await
            .unwrap();
        assert_eq!(
            rows.len(),
            5,
            "returns the real rows, clamp doesn't inflate"
        );

        // A zero/negative limit clamps up to 1 (never returns an empty/invalid page).
        let mut zero = q();
        zero.limit = Some(0);
        assert!(!query_records_inner(&pool, "leslie", zero)
            .await
            .unwrap()
            .is_empty());
        let mut negative = q();
        negative.limit = Some(-42);
        assert!(!query_records_inner(&pool, "leslie", negative)
            .await
            .unwrap()
            .is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_limit_is_clamped_to_max(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("deploy target one"))
            .await
            .unwrap();
        // An oversized context limit is clamped (to 200) rather than honored or
        // erroring — a single request can't demand unbounded retrieval work.
        let out = assemble_inner(
            &pool,
            &StubNlp,
            "leslie",
            AssembleRequest {
                include_sandbox: false,
                project_id: None,
                container_id: None,
                mode: None,
                modes: None,
                query: Some("deploy".into()),
                limit: Some(10_000_000),
                exclude_container_id: None,
            },
        )
        .await
        .unwrap();
        assert!(out.records.len() <= 200, "context limit clamped to 200");
    }
}
