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
const COLS: &str = "id, type, content, content_hash, event_time, ingest_time, \
    source, source_ref, user_id, project_id, session_id, mode, importance, \
    supersedes, acl, ttl, payload";
const COLS_R: &str = "r.id, r.type, r.content, r.content_hash, r.event_time, r.ingest_time, \
    r.source, r.source_ref, r.user_id, r.project_id, r.session_id, r.mode, r.importance, \
    r.supersedes, r.acl, r.ttl, r.payload";

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/", post(ingest))
        .route("/import", post(import_records))
        .route("/query", post(query_records))
        .route("/context", post(assemble))
        .route("/summaries", get(summaries))
        .route("/rebuild", post(rebuild_curation))
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
    pub session_id: Option<String>,
    pub mode: Option<String>,
    pub importance: Option<f32>,
    pub supersedes: Option<Uuid>,
    pub acl: Option<Value>,
    pub ttl: Option<DateTime<Utc>>,
    pub payload: Option<Value>,
}

fn validate_type(t: &str) -> AppResult<()> {
    match t {
        "episodic" | "semantic" | "working" | "document" | "procedural" | "state_object" => Ok(()),
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
    pub session_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    /// id this record supersedes (forward pointer; the old row is never mutated).
    #[serde(default)]
    pub supersedes: Option<Uuid>,
    #[serde(default)]
    pub ttl_hours: Option<i64>,
    #[serde(default)]
    pub acl: Option<Value>,
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
    let ttl = req.ttl_hours.map(|h| Utc::now() + Duration::hours(h));

    // content_hash + content_tsv + ingest_time are generated/defaulted by the DB.
    sqlx::query(
        r#"
        INSERT INTO raw_records
            (id, type, content, event_time, source, source_ref,
             user_id, project_id, session_id, mode, importance, supersedes, acl, ttl, payload)
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
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
    .bind(&req.session_id)
    .bind(&req.mode)
    .bind(importance)
    .bind(req.supersedes)
    .bind(&req.acl)
    .bind(ttl)
    .bind(&req.payload)
    .execute(pool)
    .await?;

    // Derived embedding — best-effort. Raw is the source of truth; if the
    // embedder is down a backfill job can fill the gap later. Never fails ingest.
    if let Ok(embedding) = nlp.embed_one(&req.content).await {
        let _ = sqlx::query(
            "INSERT INTO raw_embeddings (record_id, model, embedding) VALUES ($1, $2, $3) \
             ON CONFLICT (record_id, model) DO NOTHING",
        )
        .bind(id)
        .bind(nlp.embedder_model_name())
        .bind(Vector::from(embedding))
        .execute(pool)
        .await;
    }

    Ok(IngestRecordResponse { id })
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
    pub session_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
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
    let mut inserted: Vec<(Uuid, String)> = Vec::new();

    let mut tx = pool.begin().await?;
    for r in req.records {
        if validate_type(&r.r#type).is_err() || r.content.trim().is_empty() {
            continue;
        }
        let id = Uuid::new_v4();
        let event_time = r.event_time.unwrap_or_else(Utc::now);
        let importance = r.importance.map(|i| i.clamp(0.0, 1.0));
        let new_id: Option<Uuid> = sqlx::query_scalar(
            r#"
            INSERT INTO raw_records
                (id, type, content, event_time, source, source_ref,
                 user_id, project_id, session_id, mode, importance, payload)
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
        .bind(&r.session_id)
        .bind(&r.mode)
        .bind(importance)
        .bind(&r.payload)
        .fetch_optional(&mut *tx)
        .await?;
        if let Some(nid) = new_id {
            inserted.push((nid, r.content));
        }
    }
    tx.commit().await?;

    // Batch-embed the freshly imported rows (best-effort; a backfill can retry).
    if !inserted.is_empty() {
        let contents: Vec<String> = inserted.iter().map(|(_, c)| c.clone()).collect();
        if let Ok(embs) = nlp.embed_batch(contents).await {
            if embs.len() == inserted.len() {
                let model = nlp.embedder_model_name().to_string();
                for ((rid, _), emb) in inserted.iter().zip(embs) {
                    let _ = sqlx::query(
                        "INSERT INTO raw_embeddings (record_id, model, embedding) \
                         VALUES ($1, $2, $3) ON CONFLICT (record_id, model) DO NOTHING",
                    )
                    .bind(rid)
                    .bind(&model)
                    .bind(Vector::from(emb))
                    .execute(pool)
                    .await;
                }
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
    pub session_id: Option<String>,
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
          AND ($3::text IS NULL OR session_id = $3)
          AND ($4::text IS NULL OR mode = $4)
          AND ($5::text IS NULL OR type = $5)
          AND ($6::timestamptz IS NULL OR event_time >= $6)
          AND ($7::timestamptz IS NULL OR event_time <= $7)
          AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $1)
          AND (ttl IS NULL OR ttl > NOW())
        ORDER BY event_time DESC
        LIMIT $8
        "#
    );
    // sql is built only from trusted column constants (COLS) + $-params.
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(&req.project_id)
        .bind(&req.session_id)
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

#[derive(Debug, Deserialize)]
pub struct AssembleRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub limit: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct AssembleResponse {
    pub records: Vec<RawRecordRow>,
}

async fn assemble(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<AssembleRequest>,
) -> AppResult<Json<AssembleResponse>> {
    Ok(Json(
        assemble_inner(&state.pool, &*state.nlp, &auth_user.user_id, req).await?,
    ))
}

pub(crate) async fn assemble_inner(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    req: AssembleRequest,
) -> AppResult<AssembleResponse> {
    let limit = req.limit.unwrap_or(50).clamp(1, 200);
    let query = req.query.clone().unwrap_or_default();

    if query.trim().is_empty() {
        // No query -> pure recency over raw (the backstop feed; nothing to score).
        let sql = format!(
            r#"
            SELECT {COLS} FROM raw_records
            WHERE user_id = $1
              AND ($2::text IS NULL OR project_id = $2)
              AND ($3::text IS NULL OR session_id = $3)
              AND ($4::text IS NULL OR mode = $4)
              AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $1)
              AND (ttl IS NULL OR ttl > NOW())
            ORDER BY event_time DESC
            LIMIT $5
            "#
        );
        let records = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
            .bind(user_id)
            .bind(&req.project_id)
            .bind(&req.session_id)
            .bind(&req.mode)
            .bind(limit)
            .fetch_all(pool)
            .await?;
        return Ok(AssembleResponse { records });
    }

    let qvec = nlp.embed_one(&query).await?;
    let model = nlp.embedder_model_name().to_string();

    // 1) Rank the CURATED layer with the full blended score. Summaries and
    // level-0 nodes compete on the same score; higher `level` breaks ties so a
    // summary that covers a cluster leads its own children ("summary first").
    // recency = e^(-age_hours/REC_S); ref_weight = e^(-idle_hours/half_life)
    // (the Ebbinghaus decay, computed inline from ref_weights; a node with no
    // ref_weights row scores 1.0 so absence never demotes it).
    let curated_scored: Vec<(Uuid, i32)> = {
        let sql = r#"
            SELECT n.id, n.level
            FROM curated_nodes n
            LEFT JOIN curated_embeddings ce ON ce.node_id = n.id AND ce.model = $1
            LEFT JOIN ref_weights rw ON rw.ref_id = n.id
            WHERE n.user_id = $2
              AND ($3::text IS NULL OR n.project_id = $3)
              AND ($4::text IS NULL OR n.mode = $4)
              AND NOT EXISTS (
                  SELECT 1 FROM curated_edges s WHERE s.to_id = n.id AND s.kind = 'supersedes'
              )
            ORDER BY (
                  $5 * COALESCE(NULLIF(1 - (ce.embedding <=> $6), 'NaN'::float8), 0)
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
        "#;
        sqlx::query_as::<_, (Uuid, i32)>(sql)
            .bind(&model)
            .bind(user_id)
            .bind(&req.project_id)
            .bind(&req.mode)
            .bind(W_SEM)
            .bind(Vector::from(qvec.clone()))
            .bind(W_KW)
            .bind(&query)
            .bind(W_REC)
            .bind(REC_S_HOURS)
            .bind(W_IMP)
            .bind(W_DECAY)
            .bind(limit)
            .fetch_all(pool)
            .await?
    };

    // Bump the decay clock for every curated node we're serving (accessing a
    // node resets its retention to 1.0). Never deletes.
    let touched: Vec<Uuid> = curated_scored.iter().map(|(id, _)| *id).collect();
    crate::decay::touch_refs(pool, crate::decay::REF_KIND_CURATED, &touched).await?;

    // 2) Resolve curated hits -> source raw ids, in curated-rank order. A
    // summary resolves through its `summarizes` chain down to level-0 nodes,
    // then each level-0 node's `derived_from` edges to raw ids.
    let mut ordered_raw_ids: Vec<Uuid> = Vec::new();
    let mut seen: std::collections::HashSet<Uuid> = std::collections::HashSet::new();
    for (node_id, _level) in &curated_scored {
        let raw_ids = raw_ids_under_node(pool, *node_id).await?;
        for rid in raw_ids {
            if seen.insert(rid) {
                ordered_raw_ids.push(rid);
            }
        }
        if ordered_raw_ids.len() as i64 >= limit {
            break;
        }
    }

    // 3) Fetch those raw records (still active + owned), preserving curated rank.
    let mut records = fetch_raw_in_order(pool, user_id, &ordered_raw_ids).await?;

    // 4) Backstop: if the curated layer under-fills the page (e.g. curation
    // hasn't run yet, or a fresh raw record isn't summarized), top up with a
    // direct raw hybrid search — the same 0.7·cosine + 0.3·keyword fallback.
    if (records.len() as i64) < limit {
        let need = limit - records.len() as i64;
        let exclude: Vec<Uuid> = records.iter().map(|r| r.id).collect();
        let backstop =
            raw_hybrid(pool, user_id, &req, &qvec, &model, &query, need, &exclude).await?;
        records.extend(backstop);
    }

    Ok(AssembleResponse { records })
}

/// Every raw id reachable from a curated node: walk `summarizes` down to the
/// level-0 nodes, then their `derived_from` edges to raw record ids. A level-0
/// node (no `summarizes` children) resolves straight through its own
/// `derived_from` edges.
async fn raw_ids_under_node(pool: &PgPool, node_id: Uuid) -> AppResult<Vec<Uuid>> {
    let sql = r#"
        WITH RECURSIVE tree AS (
            SELECT $1::uuid AS id
            UNION
            SELECT e.to_id
            FROM curated_edges e
            JOIN tree t ON e.from_id = t.id AND e.kind = 'summarizes'
        )
        SELECT DISTINCT df.to_id
        FROM tree t
        JOIN curated_edges df ON df.from_id = t.id AND df.kind = 'derived_from'
    "#;
    let ids: Vec<Uuid> = sqlx::query_scalar(sql)
        .bind(node_id)
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
) -> AppResult<Vec<RawRecordRow>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let sql = format!(
        r#"
        SELECT {COLS} FROM raw_records
        WHERE user_id = $1
          AND id = ANY($2)
          AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $1)
          AND (ttl IS NULL OR ttl > NOW())
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(ids)
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
    qvec: &[f32],
    model: &str,
    query: &str,
    limit: i64,
    exclude: &[Uuid],
) -> AppResult<Vec<RawRecordRow>> {
    let sql = format!(
        r#"
        SELECT {COLS_R} FROM raw_records r
        LEFT JOIN raw_embeddings e ON e.record_id = r.id AND e.model = $1
        WHERE r.user_id = $2
          AND ($3::text IS NULL OR r.project_id = $3)
          AND ($4::text IS NULL OR r.session_id = $4)
          AND ($5::text IS NULL OR r.mode = $5)
          AND r.id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL AND user_id = $2)
          AND (r.ttl IS NULL OR r.ttl > NOW())
          AND NOT (r.id = ANY($9))
        ORDER BY (
            0.7 * COALESCE(NULLIF(1 - (e.embedding <=> $6), 'NaN'::float8), 0)
            + 0.3 * ts_rank(r.content_tsv, plainto_tsquery('english', $7))
        ) DESC, r.event_time DESC
        LIMIT $8
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(model)
        .bind(user_id)
        .bind(&req.project_id)
        .bind(&req.session_id)
        .bind(&req.mode)
        .bind(Vector::from(qvec.to_vec()))
        .bind(query)
        .bind(limit)
        .bind(exclude)
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

    fn req(content: &str) -> IngestRecordRequest {
        IngestRecordRequest {
            r#type: "episodic".into(),
            content: content.into(),
            event_time: None,
            source: "test".into(),
            source_ref: None,
            project_id: Some("health".into()),
            session_id: None,
            mode: None,
            importance: None,
            supersedes: None,
            ttl_hours: None,
            acl: None,
            payload: None,
        }
    }

    fn q() -> QueryRecordsRequest {
        QueryRecordsRequest {
            project_id: None,
            session_id: None,
            mode: None,
            r#type: None,
            since: None,
            until: None,
            limit: None,
        }
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
                project_id: None,
                session_id: None,
                mode: None,
                query: Some("lisinopril".into()),
                limit: None,
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
                project_id: None,
                session_id: None,
                mode: None,
                query: None,
                limit: None,
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
            r#type: "episodic".into(),
            content: content.into(),
            event_time: None,
            source: "chatgpt".into(),
            source_ref: source_ref.map(|s| s.into()),
            project_id: None,
            session_id: None,
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
                project_id: None,
                session_id: None,
                mode: None,
                query: Some("lisinopril".into()),
                limit: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(ctx.records.len(), 1);
    }

    // -- summary-first + decay-weighted retrieval --------------------------

    /// Insert a level-0 curated node deriving from `raw_id`, embed it (uniform
    /// via StubNlp so cosine is constant across nodes), and return the node id.
    async fn curated_node_from(pool: &PgPool, user_id: &str, raw_id: Uuid, content: &str) -> Uuid {
        let node_id = Uuid::new_v4();
        sqlx::query(
            "INSERT INTO curated_nodes (id, kind, content, level, user_id, importance, event_time) \
             VALUES ($1, 'episodic', $2, 0, $3, 0.5, NOW())",
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
        .bind(StubNlp.embedder_model_name())
        .bind(Vector::from(emb))
        .execute(pool)
        .await
        .unwrap();
        node_id
    }

    fn ctx_req(query: &str) -> AssembleRequest {
        AssembleRequest {
            project_id: None,
            session_id: None,
            mode: None,
            query: Some(query.into()),
            limit: None,
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
}
