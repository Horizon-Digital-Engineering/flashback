//! The RAW-records door (medallion "bronze"). Universal typed records, immutable
//! and append-only. No embeddings/NLP here — raw is pre-curation; the curation
//! pipeline derives embeddings + views + references from these rows later.
//!
//! Endpoints (nested under /records):
//!   POST /records          ingest a typed record
//!   POST /records/query    structured reads (dashboards)
//!   POST /records/context  dumb keyword+recency context (the walking-skeleton getContext)
//!   GET  /records/:id       one record
//!   GET  /records/:id/lineage   the supersede chain

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::{AssertSqlSafe, FromRow, PgPool};
use uuid::Uuid;

use crate::{auth::AuthUser, error::AppResult, error::AppError, AppState};

/// The columns of raw_records, in RawRecordRow field order. `content_tsv` is a
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
        .route("/query", post(query_records))
        .route("/context", post(assemble))
        .route("/:id", get(get_record))
        .route("/:id/lineage", get(lineage))
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
    Ok(Json(ingest_record(&state.pool, &auth_user.user_id, req).await?))
}

pub(crate) async fn ingest_record(
    pool: &PgPool,
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

    Ok(IngestRecordResponse { id })
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
    // sql is built only from trusted column constants (COLS/COLS_R) + $-params.
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
// Context assemble (the walking-skeleton getContext: keyword + recency)
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
    Ok(Json(assemble_inner(&state.pool, &auth_user.user_id, req).await?))
}

pub(crate) async fn assemble_inner(
    pool: &PgPool,
    user_id: &str,
    req: AssembleRequest,
) -> AppResult<AssembleResponse> {
    let limit = req.limit.unwrap_or(50).clamp(1, 200);
    let query = req.query.unwrap_or_default();
    // Empty query -> ts_rank 0 everywhere -> pure recency. This is the dumb fill;
    // the smart relevance ranking replaces this SQL behind the same endpoint.
    let sql = format!(
        r#"
        SELECT {COLS} FROM raw_records
        WHERE user_id = $1
          AND ($2::text IS NULL OR project_id = $2)
          AND ($3::text IS NULL OR session_id = $3)
          AND ($4::text IS NULL OR mode = $4)
          AND id NOT IN (SELECT supersedes FROM raw_records WHERE supersedes IS NOT NULL)
          AND (ttl IS NULL OR ttl > NOW())
        ORDER BY ts_rank(content_tsv, plainto_tsquery('english', $5)) DESC, event_time DESC
        LIMIT $6
        "#
    );
    let records = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(&req.project_id)
        .bind(&req.session_id)
        .bind(&req.mode)
        .bind(&query)
        .bind(limit)
        .fetch_all(pool)
        .await?;
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
    // sql is built only from trusted column constants (COLS/COLS_R) + $-params.
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let out = ingest_record(&pool, "leslie", req("took 5mg lisinopril")).await.unwrap();
        let rows = query_records_inner(&pool, "leslie", q()).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, out.id);
        assert_eq!(rows[0].content, "took 5mg lisinopril");
        assert_eq!(rows[0].content_hash.len(), 32); // md5 hex
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn raw_records_are_immutable(pool: PgPool) {
        let out = ingest_record(&pool, "leslie", req("original")).await.unwrap();
        // A direct UPDATE must be rejected by the trigger.
        let err = sqlx::query("UPDATE raw_records SET content = 'tampered' WHERE id = $1")
            .bind(out.id)
            .execute(&pool)
            .await;
        assert!(err.is_err(), "UPDATE on raw_records should be blocked");
        let err2 = sqlx::query("DELETE FROM raw_records WHERE id = $1")
            .bind(out.id)
            .execute(&pool)
            .await;
        assert!(err2.is_err(), "DELETE on raw_records should be blocked");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_forward_pointer_hides_old_keeps_it(pool: PgPool) {
        let v1 = ingest_record(&pool, "leslie", req("weight 180")).await.unwrap();
        let mut r2 = req("weight 178");
        r2.supersedes = Some(v1.id);
        let v2 = ingest_record(&pool, "leslie", r2).await.unwrap();

        // Active query shows only the new one.
        let active = query_records_inner(&pool, "leslie", q()).await.unwrap();
        assert_eq!(active.iter().map(|r| r.id).collect::<Vec<_>>(), vec![v2.id]);

        // The old row STILL EXISTS, unmutated (its supersedes is null; we never touched it).
        let old = get_record_inner(&pool, "leslie", v1.id).await.unwrap();
        assert_eq!(old.content, "weight 180");
        assert_eq!(old.supersedes, None);

        // Lineage returns both, oldest-first.
        let line = lineage_inner(&pool, "leslie", v2.id).await.unwrap();
        assert_eq!(line.iter().map(|r| r.id).collect::<Vec<_>>(), vec![v1.id, v2.id]);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn context_ranks_keyword_over_recency(pool: PgPool) {
        ingest_record(&pool, "leslie", req("weighed 180 lbs")).await.unwrap();
        ingest_record(&pool, "leslie", req("discussed lisinopril dosage")).await.unwrap();
        ingest_record(&pool, "leslie", req("ate lunch")).await.unwrap();
        let out = assemble_inner(&pool, "leslie", AssembleRequest {
            project_id: None, session_id: None, mode: None,
            query: Some("lisinopril".into()), limit: None,
        }).await.unwrap();
        assert!(out.records[0].content.contains("lisinopril"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn scoped_and_isolated_by_user(pool: PgPool) {
        ingest_record(&pool, "leslie", req("mine")).await.unwrap();
        ingest_record(&pool, "bob", req("theirs")).await.unwrap();
        assert_eq!(query_records_inner(&pool, "leslie", q()).await.unwrap().len(), 1);
        assert_eq!(query_records_inner(&pool, "bob", q()).await.unwrap().len(), 1);
        // bob cannot read leslie's record by id.
        let leslies = query_records_inner(&pool, "leslie", q()).await.unwrap()[0].id;
        assert!(get_record_inner(&pool, "bob", leslies).await.is_err());
    }
}
