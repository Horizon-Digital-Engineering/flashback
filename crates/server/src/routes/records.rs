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
        .route("/{id}", get(get_record))
        .route("/{id}/lineage", get(lineage))
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
            _ => return Err(AppError::bad_request(format!("supersedes target {sid} not found"))),
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

    Ok(ImportResponse { imported: inserted.len(), skipped: total - inserted.len() })
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
    Ok(Json(query_records_inner(&state.pool, &auth_user.user_id, req).await?))
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
          AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL)
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
// Context assemble (getContext): hybrid vector+keyword when a query is present,
// recency when it isn't. The dumb-but-real fill; the curated view feeds in later.
// ---------------------------------------------------------------------------

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

    let records = if query.trim().is_empty() {
        // No query -> pure recency.
        let sql = format!(
            r#"
            SELECT {COLS} FROM raw_records
            WHERE user_id = $1
              AND ($2::text IS NULL OR project_id = $2)
              AND ($3::text IS NULL OR session_id = $3)
              AND ($4::text IS NULL OR mode = $4)
              AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL)
              AND (ttl IS NULL OR ttl > NOW())
            ORDER BY event_time DESC
            LIMIT $5
            "#
        );
        sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
            .bind(user_id)
            .bind(&req.project_id)
            .bind(&req.session_id)
            .bind(&req.mode)
            .bind(limit)
            .fetch_all(pool)
            .await?
    } else {
        // Hybrid: 0.7 * cosine-similarity(query, record) + 0.3 * keyword rank.
        // LEFT JOIN so records without an embedding still rank on keyword.
        let qvec = nlp.embed_one(&query).await?;
        let model = nlp.embedder_model_name().to_string();
        let sql = format!(
            r#"
            SELECT {COLS_R} FROM raw_records r
            LEFT JOIN raw_embeddings e ON e.record_id = r.id AND e.model = $1
            WHERE r.user_id = $2
              AND ($3::text IS NULL OR r.project_id = $3)
              AND ($4::text IS NULL OR r.session_id = $4)
              AND ($5::text IS NULL OR r.mode = $5)
              AND r.id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL)
              AND (r.ttl IS NULL OR r.ttl > NOW())
            ORDER BY (
                0.7 * COALESCE(1 - (e.embedding <=> $6), 0)
                + 0.3 * ts_rank(r.content_tsv, plainto_tsquery('english', $7))
            ) DESC, r.event_time DESC
            LIMIT $8
            "#
        );
        sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
            .bind(model)
            .bind(user_id)
            .bind(&req.project_id)
            .bind(&req.session_id)
            .bind(&req.mode)
            .bind(Vector::from(qvec))
            .bind(&query)
            .bind(limit)
            .fetch_all(pool)
            .await?
    };
    Ok(AssembleResponse { records })
}

// ---------------------------------------------------------------------------
// Get by id
// ---------------------------------------------------------------------------

async fn get_record(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<RawRecordRow>> {
    Ok(Json(get_record_inner(&state.pool, &auth_user.user_id, id).await?))
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
    Ok(Json(lineage_inner(&state.pool, &auth_user.user_id, id).await?))
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
            SELECT {COLS_R} FROM raw_records r JOIN back b ON r.id = b.supersedes
        ),
        fwd AS (
            SELECT {COLS} FROM raw_records WHERE id = $1
            UNION ALL
            SELECT {COLS_R} FROM raw_records r JOIN fwd f ON r.supersedes = f.id
        )
        SELECT {COLS} FROM (
            SELECT {COLS} FROM back UNION SELECT {COLS} FROM fwd
        ) u
        ORDER BY ingest_time ASC
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
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
        fn provider_name(&self) -> &'static str { "stub" }
        fn provider_can_distill(&self) -> bool { false }
        fn embedder_model_name(&self) -> &str { "stub-embedder" }
        fn embedder_dimension(&self) -> usize { 384 }
        async fn embed_one(&self, _text: &str) -> Result<Vec<f32>, AppError> {
            Ok(vec![0.1_f32; 384]) // non-zero to keep cosine defined
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| vec![0.1_f32; 384]).collect())
        }
        fn extract_entities(&self, _text: &str) -> Vec<String> { Vec::new() }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(&self, _e: &[EpisodeRef]) -> Result<Vec<DistilledFact>, ProviderError> {
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
            project_id: None, session_id: None, mode: None,
            r#type: None, since: None, until: None, limit: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_and_query_roundtrip(pool: PgPool) {
        let out = ingest_record(&pool, &StubNlp, "leslie", req("took 5mg lisinopril")).await.unwrap();
        let rows = query_records_inner(&pool, "leslie", q()).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, out.id);
        assert_eq!(rows[0].content, "took 5mg lisinopril");
        assert_eq!(rows[0].content_hash.len(), 32); // md5 hex

        // The derived embedding landed in the separate table.
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM raw_embeddings WHERE record_id = $1")
            .bind(out.id).fetch_one(&pool).await.unwrap();
        assert_eq!(n, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn raw_records_are_immutable(pool: PgPool) {
        let out = ingest_record(&pool, &StubNlp, "leslie", req("original")).await.unwrap();
        let upd = sqlx::query("UPDATE raw_records SET content = 'tampered' WHERE id = $1")
            .bind(out.id).execute(&pool).await;
        assert!(upd.is_err(), "UPDATE on raw_records should be blocked");
        let del = sqlx::query("DELETE FROM raw_records WHERE id = $1")
            .bind(out.id).execute(&pool).await;
        assert!(del.is_err(), "DELETE on raw_records should be blocked");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_forward_pointer_hides_old_keeps_it(pool: PgPool) {
        let v1 = ingest_record(&pool, &StubNlp, "leslie", req("weight 180")).await.unwrap();
        let mut r2 = req("weight 178");
        r2.supersedes = Some(v1.id);
        let v2 = ingest_record(&pool, &StubNlp, "leslie", r2).await.unwrap();

        let active = query_records_inner(&pool, "leslie", q()).await.unwrap();
        assert_eq!(active.iter().map(|r| r.id).collect::<Vec<_>>(), vec![v2.id]);

        let old = get_record_inner(&pool, "leslie", v1.id).await.unwrap();
        assert_eq!(old.content, "weight 180");
        assert_eq!(old.supersedes, None); // never mutated

        let line = lineage_inner(&pool, "leslie", v2.id).await.unwrap();
        assert_eq!(line.iter().map(|r| r.id).collect::<Vec<_>>(), vec![v1.id, v2.id]);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_hybrid_ranks_keyword_over_recency(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("weighed 180 lbs")).await.unwrap();
        ingest_record(&pool, &StubNlp, "leslie", req("discussed lisinopril dosage")).await.unwrap();
        ingest_record(&pool, &StubNlp, "leslie", req("ate lunch")).await.unwrap();
        // Stub embeddings are uniform, so the vector term is constant and the
        // keyword term decides — the lisinopril record must come first.
        let out = assemble_inner(&pool, &StubNlp, "leslie", AssembleRequest {
            project_id: None, session_id: None, mode: None,
            query: Some("lisinopril".into()), limit: None,
        }).await.unwrap();
        assert!(out.records[0].content.contains("lisinopril"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_empty_query_is_recency(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("first")).await.unwrap();
        let latest = ingest_record(&pool, &StubNlp, "leslie", req("second")).await.unwrap();
        let out = assemble_inner(&pool, &StubNlp, "leslie", AssembleRequest {
            project_id: None, session_id: None, mode: None, query: None, limit: None,
        }).await.unwrap();
        assert_eq!(out.records[0].id, latest.id); // most recent first
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn scoped_and_isolated_by_user(pool: PgPool) {
        ingest_record(&pool, &StubNlp, "leslie", req("mine")).await.unwrap();
        ingest_record(&pool, &StubNlp, "bob", req("theirs")).await.unwrap();
        assert_eq!(query_records_inner(&pool, "leslie", q()).await.unwrap().len(), 1);
        assert_eq!(query_records_inner(&pool, "bob", q()).await.unwrap().len(), 1);
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
            project_id: None, session_id: None, mode: None,
            importance: None, payload: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn import_bulk_then_dedup_on_reimport(pool: PgPool) {
        let batch = ImportRequest { records: vec![imp("chat one", Some("m1")), imp("chat two", Some("m2"))] };
        let out = import_records_inner(&pool, &StubNlp, "leslie", batch).await.unwrap();
        assert_eq!((out.imported, out.skipped), (2, 0));

        // Re-import the same source_refs -> all deduped, nothing duplicated.
        let batch2 = ImportRequest { records: vec![imp("chat one", Some("m1")), imp("chat two edited", Some("m2"))] };
        let out2 = import_records_inner(&pool, &StubNlp, "leslie", batch2).await.unwrap();
        assert_eq!((out2.imported, out2.skipped), (0, 2));
        assert_eq!(query_records_inner(&pool, "leslie", q()).await.unwrap().len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn import_without_source_ref_always_inserts(pool: PgPool) {
        let b1 = ImportRequest { records: vec![imp("ok", None)] };
        let b2 = ImportRequest { records: vec![imp("ok", None)] };
        assert_eq!(import_records_inner(&pool, &StubNlp, "leslie", b1).await.unwrap().imported, 1);
        assert_eq!(import_records_inner(&pool, &StubNlp, "leslie", b2).await.unwrap().imported, 1);
        assert_eq!(query_records_inner(&pool, "leslie", q()).await.unwrap().len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn imported_records_are_embedded_and_searchable(pool: PgPool) {
        let batch = ImportRequest { records: vec![imp("discussed lisinopril", Some("x1"))] };
        assert_eq!(import_records_inner(&pool, &StubNlp, "leslie", batch).await.unwrap().imported, 1);
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM raw_embeddings").fetch_one(&pool).await.unwrap();
        assert_eq!(n, 1);
        let ctx = assemble_inner(&pool, &StubNlp, "leslie", AssembleRequest {
            project_id: None, session_id: None, mode: None, query: Some("lisinopril".into()), limit: None,
        }).await.unwrap();
        assert_eq!(ctx.records.len(), 1);
    }
}
