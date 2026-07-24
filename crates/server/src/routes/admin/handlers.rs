//! Axum handlers for the admin UI. Hand-rolled HTML — no templating crate.
//!
//! The admin renders the canonical world: `raw_records` (the immutable raw
//! layer), the `curated_*` derived layer, references (state_object raw records),
//! the data catalog, and the operator proposal queue.

use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
    Form, Json,
};
use chrono::{DateTime, Utc};
use pgvector::Vector;
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    auth::{self, AuthUser},
    AppState,
};

use super::{
    projection::{build_graph, GraphInput, GraphLayout},
    style::STYLE_CSS,
    views,
};

// ---------------------------------------------------------------------------
// Raw record row (admin view shape)
// ---------------------------------------------------------------------------

/// A raw record enriched with the derived facts the admin needs: whether a
/// newer row supersedes it, and the entities the index maps to it. Raw records
/// carry no `entities` column and no `superseded_by` — both are derived here.
#[derive(Debug, Clone)]
pub(crate) struct RawAdminRow {
    pub id: Uuid,
    pub r#type: String,
    pub content: String,
    pub source: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub mode: Option<String>,
    pub importance: Option<f32>,
    pub superseded: bool,
    pub state_kind: Option<String>,
    pub state_key: Option<String>,
    pub payload: Option<Value>,
    pub entities: Vec<String>,
    pub event_time: DateTime<Utc>,
}

#[derive(Debug, Clone, sqlx::FromRow)]
struct RawRecordDbRow {
    id: Uuid,
    r#type: String,
    content: String,
    source: String,
    // Selected so the row shape matches the SELECT; ownership is now enforced
    // in-query (WHERE user_id = $), so nothing reads this after mapping.
    #[allow(dead_code)]
    user_id: String,
    project_id: Option<String>,
    session_id: Option<String>,
    mode: Option<String>,
    importance: Option<f32>,
    superseded: bool,
    state_kind: Option<String>,
    state_key: Option<String>,
    payload: Option<Value>,
    event_time: DateTime<Utc>,
}

impl RawRecordDbRow {
    fn into_admin(self, entities: Vec<String>) -> RawAdminRow {
        RawAdminRow {
            id: self.id,
            r#type: self.r#type,
            content: self.content,
            source: self.source,
            project_id: self.project_id,
            session_id: self.session_id,
            mode: self.mode,
            importance: self.importance,
            superseded: self.superseded,
            state_kind: self.state_kind,
            state_key: self.state_key,
            payload: self.payload,
            entities,
            event_time: self.event_time,
        }
    }
}

/// Entities the index maps to a record, for the user.
async fn entities_for(
    pool: &sqlx::PgPool,
    user_id: &str,
    record_id: Uuid,
) -> Result<Vec<String>, sqlx::Error> {
    sqlx::query_scalar(
        "SELECT entity FROM entity_index WHERE user_id = $1 AND record_id = $2 ORDER BY entity",
    )
    .bind(user_id)
    .bind(record_id)
    .fetch_all(pool)
    .await
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

pub async fn style_css() -> Response {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLE_CSS,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Login / logout
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct LoginQuery {
    #[serde(default)]
    pub reason: Option<String>,
}

pub async fn login_form(Query(q): Query<LoginQuery>) -> Response {
    let err = match q.reason.as_deref() {
        Some("unauth") => Some("Your session expired or no token was set."),
        Some("bad-token") => Some("That token is invalid or revoked."),
        _ => None,
    };
    Html(super::views::login_page(err)).into_response()
}

#[derive(Deserialize)]
pub struct LoginForm {
    pub token: String,
}

pub async fn login_submit(State(state): State<AppState>, Form(form): Form<LoginForm>) -> Response {
    let token = form.token.trim();
    if token.is_empty() {
        return Redirect::to("/admin/login?reason=bad-token").into_response();
    }
    // Validate by attempting a token lookup. Same algorithm as the auth
    // middleware uses for Bearer headers.
    let hash = auth::sha256_hex(token);
    let row: Result<Option<(Uuid, String)>, _> = sqlx::query_as(
        r#"
        SELECT id, user_id FROM tokens
        WHERE token_hash = $1 AND revoked_at IS NULL
        LIMIT 1
        "#,
    )
    .bind(&hash)
    .fetch_optional(&state.pool)
    .await;

    let valid = matches!(row, Ok(Some(_)));
    if !valid {
        return Redirect::to("/admin/login?reason=bad-token").into_response();
    }

    let cookie =
        format!("flashback_token={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000");
    (
        StatusCode::SEE_OTHER,
        [
            (header::LOCATION, "/admin".to_string()),
            (header::SET_COOKIE, cookie),
        ],
    )
        .into_response()
}

pub async fn logout() -> Response {
    let cookie = "flashback_token=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0";
    (
        StatusCode::SEE_OTHER,
        [
            (header::LOCATION, "/admin/login"),
            (header::SET_COOKIE, cookie),
        ],
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

pub async fn dashboard(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let counts = dashboard_counts(&state.pool, &user.user_id).await?;
    let recent = fetch_records(&state.pool, &user.user_id, &RecordQuery::default(), 10, 0).await?;
    let stats = views::DashboardStats {
        records_total: counts.records_total,
        records_terminal: counts.records_terminal,
        state_objects: counts.state_objects,
        curated_nodes: counts.curated_nodes,
        proposals_pending: counts.proposals_pending,
        tokens_active: counts.tokens_active,
        provider: state.nlp.provider_name().to_string(),
        embedder_model: state.nlp.embedder_model_name().to_string(),
        embedder_dim: state.nlp.embedder_dimension(),
    };
    Ok(Html(views::dashboard(&user.user_id, stats, &recent)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DashboardCounts {
    pub records_total: i64,
    pub records_terminal: i64,
    pub state_objects: i64,
    pub curated_nodes: i64,
    pub proposals_pending: i64,
    pub tokens_active: i64,
}

/// The `SELECT COUNT(*)` queries that feed the dashboard. A raw record is
/// "terminal" when no newer row supersedes it. Extracted so the summary is
/// unit-testable without rendering HTML or constructing an AppState.
pub(crate) async fn dashboard_counts(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<DashboardCounts, super::Error> {
    let records_total: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM raw_records WHERE user_id = $1")
            .bind(user_id)
            .fetch_one(pool)
            .await?;
    let records_terminal: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM raw_records r
        WHERE r.user_id = $1
          AND NOT EXISTS (
              SELECT 1 FROM raw_records n
              WHERE n.supersedes = r.id AND n.user_id = $1
          )
        "#,
    )
    .bind(user_id)
    .fetch_one(pool)
    .await?;
    let state_objects: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM raw_records r
        WHERE r.user_id = $1 AND r.type = 'state_object'
          AND NOT EXISTS (
              SELECT 1 FROM raw_records n
              WHERE n.supersedes = r.id AND n.user_id = $1
          )
        "#,
    )
    .bind(user_id)
    .fetch_one(pool)
    .await?;
    let curated_nodes: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM curated_nodes WHERE user_id = $1")
            .bind(user_id)
            .fetch_one(pool)
            .await?;
    let proposals_pending: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM proposals WHERE user_id = $1 AND status = 'proposed'",
    )
    .bind(user_id)
    .fetch_one(pool)
    .await?;
    let tokens_active: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM tokens WHERE user_id = $1 AND revoked_at IS NULL")
            .bind(user_id)
            .fetch_one(pool)
            .await?;
    Ok(DashboardCounts {
        records_total,
        records_terminal,
        state_objects,
        curated_nodes,
        proposals_pending,
        tokens_active,
    })
}

// ---------------------------------------------------------------------------
// Records list
// ---------------------------------------------------------------------------

#[derive(Default, Deserialize)]
pub struct RecordQuery {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub include_superseded: Option<String>,
}

impl RecordQuery {
    fn include_super(&self) -> bool {
        matches!(
            self.include_superseded.as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    }
    fn clean(&self) -> Self {
        Self {
            r#type: self.r#type.as_deref().and_then(empty_to_none),
            project_id: self.project_id.as_deref().and_then(empty_to_none),
            session_id: self.session_id.as_deref().and_then(empty_to_none),
            mode: self.mode.as_deref().and_then(empty_to_none),
            include_superseded: if self.include_super() {
                Some("1".to_string())
            } else {
                None
            },
        }
    }
}

fn empty_to_none(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

pub async fn records_list(
    State(state): State<AppState>,
    user: AuthUser,
    Query(q): Query<RecordQuery>,
) -> Result<Html<String>, super::Error> {
    let q = q.clean();
    let records = fetch_records(&state.pool, &user.user_id, &q, 200, 0).await?;
    let total = count_records(&state.pool, &user.user_id, &q).await?;

    // The user's registers, for the (native) mode-filter dropdown.
    let mode_names: Vec<String> = crate::modes::list_modes(&state.pool, &user.user_id)
        .await
        .map(|ms| ms.into_iter().map(|m| m.name).collect())
        .unwrap_or_default();

    let filter = views::RecordsFilter {
        r#type: q.r#type.clone(),
        project_id: q.project_id.clone(),
        session_id: q.session_id.clone(),
        mode: q.mode.clone(),
        include_superseded: q.include_super(),
    };
    Ok(Html(views::records_list(
        &user.user_id,
        &filter,
        &mode_names,
        &records,
        total,
    )))
}

/// Fetch raw records for the list, with the derived `superseded` flag and each
/// row's entities. The `mode` filter is a NATIVE `raw_records.mode` predicate.
async fn fetch_records(
    pool: &sqlx::PgPool,
    user_id: &str,
    q: &RecordQuery,
    limit: i64,
    offset: i64,
) -> Result<Vec<RawAdminRow>, sqlx::Error> {
    let rows = sqlx::query_as::<_, RawRecordDbRow>(
        r#"
        SELECT r.id, r.type, r.content, r.source, r.user_id, r.project_id, r.session_id,
               r.mode, r.importance, r.state_kind, r.state_key, r.payload,
               r.event_time,
               EXISTS (
                   SELECT 1 FROM raw_records n
                   WHERE n.supersedes = r.id AND n.user_id = r.user_id
               ) AS superseded
        FROM raw_records r
        WHERE r.user_id = $1
          AND ($2::TEXT IS NULL OR r.type = $2)
          AND ($3::TEXT IS NULL OR r.project_id = $3)
          AND ($4::TEXT IS NULL OR r.session_id = $4)
          AND ($5::TEXT IS NULL OR r.mode = $5)
          AND ($6::BOOLEAN OR NOT EXISTS (
              SELECT 1 FROM raw_records n
              WHERE n.supersedes = r.id AND n.user_id = r.user_id
          ))
        ORDER BY r.event_time DESC
        LIMIT $7 OFFSET $8
        "#,
    )
    .bind(user_id)
    .bind(&q.r#type)
    .bind(&q.project_id)
    .bind(&q.session_id)
    .bind(&q.mode)
    .bind(q.include_super())
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await?;

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let entities = entities_for(pool, user_id, row.id).await?;
        out.push(row.into_admin(entities));
    }
    Ok(out)
}

async fn count_records(
    pool: &sqlx::PgPool,
    user_id: &str,
    q: &RecordQuery,
) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM raw_records r
        WHERE r.user_id = $1
          AND ($2::TEXT IS NULL OR r.type = $2)
          AND ($3::TEXT IS NULL OR r.project_id = $3)
          AND ($4::TEXT IS NULL OR r.session_id = $4)
          AND ($5::TEXT IS NULL OR r.mode = $5)
          AND ($6::BOOLEAN OR NOT EXISTS (
              SELECT 1 FROM raw_records n
              WHERE n.supersedes = r.id AND n.user_id = r.user_id
          ))
        "#,
    )
    .bind(user_id)
    .bind(&q.r#type)
    .bind(&q.project_id)
    .bind(&q.session_id)
    .bind(&q.mode)
    .bind(q.include_super())
    .fetch_one(pool)
    .await
}

// ---------------------------------------------------------------------------
// Record detail
// ---------------------------------------------------------------------------

pub async fn record_detail(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Html<String>, super::Error> {
    let (row, chain) = load_record_detail(&state.pool, &user.user_id, id).await?;
    Ok(Html(views::record_detail(&user.user_id, &row, &chain)))
}

/// Load a single raw record + its supersede chain (oldest → newest), each row
/// enriched with entities and the derived `superseded` flag. NotFound if the
/// row is missing or owned by another user.
pub(crate) async fn load_record_detail(
    pool: &sqlx::PgPool,
    user_id: &str,
    id: Uuid,
) -> Result<(RawAdminRow, Vec<RawAdminRow>), super::Error> {
    let base = fetch_one_record(pool, user_id, id)
        .await?
        .ok_or(super::Error::NotFound)?;

    // Walk the chain: back via supersedes forward-pointers, forward via rows
    // whose supersedes point at us. raw_records.supersedes forms a linked list.
    // Every hop is scoped to the caller so the chain can never cross a user
    // boundary (supersede targets are same-user at ingest, but defense in depth).
    let chain_rows: Vec<RawRecordDbRow> = sqlx::query_as::<_, RawRecordDbRow>(
        r#"
        WITH RECURSIVE
        back AS (
            SELECT r.* FROM raw_records r WHERE r.id = $1 AND r.user_id = $2
            UNION ALL
            SELECT prev.* FROM raw_records prev JOIN back b ON prev.id = b.supersedes
                WHERE prev.user_id = $2
        ),
        forward AS (
            SELECT r.* FROM raw_records r WHERE r.id = $1 AND r.user_id = $2
            UNION ALL
            SELECT nxt.* FROM raw_records nxt JOIN forward f ON nxt.supersedes = f.id
                WHERE nxt.user_id = $2
        ),
        chain AS (
            SELECT id FROM back UNION SELECT id FROM forward
        )
        SELECT r.id, r.type, r.content, r.source, r.user_id, r.project_id, r.session_id,
               r.mode, r.importance, r.state_kind, r.state_key, r.payload,
               r.event_time,
               EXISTS (
                   SELECT 1 FROM raw_records n
                   WHERE n.supersedes = r.id AND n.user_id = r.user_id
               ) AS superseded
        FROM raw_records r
        WHERE r.id IN (SELECT id FROM chain) AND r.user_id = $2
        ORDER BY r.event_time ASC
        "#,
    )
    .bind(id)
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    let mut chain = Vec::with_capacity(chain_rows.len());
    for r in chain_rows {
        let entities = entities_for(pool, user_id, r.id).await?;
        chain.push(r.into_admin(entities));
    }

    let entities = entities_for(pool, user_id, base.id).await?;
    Ok((base.into_admin(entities), chain))
}

async fn fetch_one_record(
    pool: &sqlx::PgPool,
    user_id: &str,
    id: Uuid,
) -> Result<Option<RawRecordDbRow>, sqlx::Error> {
    sqlx::query_as::<_, RawRecordDbRow>(
        r#"
        SELECT r.id, r.type, r.content, r.source, r.user_id, r.project_id, r.session_id,
               r.mode, r.importance, r.state_kind, r.state_key, r.payload,
               r.event_time,
               EXISTS (
                   SELECT 1 FROM raw_records n
                   WHERE n.supersedes = r.id AND n.user_id = r.user_id
               ) AS superseded
        FROM raw_records r
        WHERE r.id = $1 AND r.user_id = $2
        "#,
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(pool)
    .await
}

// ---------------------------------------------------------------------------
// State objects (references over raw)
// ---------------------------------------------------------------------------

pub async fn state_list(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let rows = list_state_objects_for(&state.pool, &user.user_id).await?;
    Ok(Html(views::state_list(&user.user_id, &rows)))
}

/// Terminal (non-superseded) state_object raw records owned by `user_id`.
pub(crate) async fn list_state_objects_for(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<Vec<RawAdminRow>, sqlx::Error> {
    let rows = sqlx::query_as::<_, RawRecordDbRow>(
        r#"
        SELECT r.id, r.type, r.content, r.source, r.user_id, r.project_id, r.session_id,
               r.mode, r.importance, r.state_kind, r.state_key, r.payload,
               r.event_time, FALSE AS superseded
        FROM raw_records r
        WHERE r.user_id = $1 AND r.type = 'state_object'
          AND NOT EXISTS (
              SELECT 1 FROM raw_records n
              WHERE n.supersedes = r.id AND n.user_id = r.user_id
          )
        ORDER BY r.event_time DESC
        LIMIT 200
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|r| r.into_admin(Vec::new())).collect())
}

// ---------------------------------------------------------------------------
// Curated layer
// ---------------------------------------------------------------------------

pub async fn curated_list(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let nodes = list_curated_nodes_for(&state.pool, &user.user_id).await?;
    Ok(Html(views::curated_list(&user.user_id, &nodes)))
}

pub(crate) async fn list_curated_nodes_for(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<Vec<views::CuratedNodeView>, sqlx::Error> {
    let rows: Vec<views::CuratedNodeView> = sqlx::query_as::<_, CuratedNodeDbRow>(
        r#"
        SELECT kind, content, level, created_at
        FROM curated_nodes
        WHERE user_id = $1
        ORDER BY level DESC, created_at DESC
        LIMIT 200
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(|r| views::CuratedNodeView {
        kind: r.kind,
        content: r.content,
        level: r.level,
        created_at: r.created_at,
    })
    .collect();
    Ok(rows)
}

#[derive(sqlx::FromRow)]
struct CuratedNodeDbRow {
    kind: String,
    content: String,
    level: i32,
    created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

pub async fn catalog_view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let catalog = crate::catalog::list_catalog_inner(&state.pool, &user.user_id)
        .await
        .map_err(app_err_to_admin)?;
    Ok(Html(views::catalog_view(&user.user_id, &catalog)))
}

// ---------------------------------------------------------------------------
// Proposals
// ---------------------------------------------------------------------------

pub async fn proposals_view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let proposals = crate::proposals::list_proposals_inner(&state.pool, &user.user_id, None)
        .await
        .map_err(app_err_to_admin)?;
    Ok(Html(views::proposals_view(&user.user_id, &proposals)))
}

pub async fn proposal_approve(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Response, super::Error> {
    // Best-effort decision; a stale/foreign id just refreshes the queue.
    let _ =
        crate::proposals::approve_inner(&state.pool, &user.user_id, id, Some(&user.user_id)).await;
    Ok(Redirect::to("/admin/proposals").into_response())
}

pub async fn proposal_deny(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Response, super::Error> {
    let _ = crate::proposals::deny_inner(&state.pool, &user.user_id, id, Some(&user.user_id)).await;
    Ok(Redirect::to("/admin/proposals").into_response())
}

/// Bridge an AppError from the shared modules into the admin UI's Error type.
/// A DB failure surfaces as a 500; anything else (not-found/bad-request) is a
/// 404 in the admin context — the admin pages don't distinguish finer than that.
fn app_err_to_admin(e: crate::error::AppError) -> super::Error {
    match e {
        crate::error::AppError::Database(db) => super::Error::Db(db),
        _ => super::Error::NotFound,
    }
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

pub async fn tokens_list(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let views = load_token_views_for(&state.pool, &user.user_id).await?;
    Ok(Html(views::tokens_list(&user.user_id, &views)))
}

/// Fetch every token owned by `user_id` (active + revoked, newest first) and
/// map into the admin-UI view shape.
pub(crate) async fn load_token_views_for(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<Vec<views::TokenView>, sqlx::Error> {
    #[derive(sqlx::FromRow)]
    struct Row {
        id: Uuid,
        token_prefix: String,
        user_id: String,
        name: Option<String>,
        created_at: DateTime<Utc>,
        last_used_at: Option<DateTime<Utc>>,
        revoked_at: Option<DateTime<Utc>>,
    }
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT id, token_prefix, user_id, name, created_at, last_used_at, revoked_at
        FROM tokens WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| views::TokenView {
            id: r.id,
            prefix: r.token_prefix,
            user_id: r.user_id,
            name: r.name,
            created_at: r.created_at,
            last_used_at: r.last_used_at,
            revoked_at: r.revoked_at,
        })
        .collect())
}

pub async fn token_revoke(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Response, super::Error> {
    revoke_token_owned_by(&state.pool, &user.user_id, id).await?;
    Ok(Redirect::to("/admin/tokens").into_response())
}

/// Revoke a token (if currently active) belonging to `user_id`. Returns Ok
/// even if the row is missing or already revoked — the admin UI just refreshes
/// the list on success.
pub(crate) async fn revoke_token_owned_by(
    pool: &sqlx::PgPool,
    user_id: &str,
    id: Uuid,
) -> Result<(), super::Error> {
    sqlx::query(
        "UPDATE tokens SET revoked_at = NOW() WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL",
    )
    .bind(id)
    .bind(user_id)
    .execute(pool)
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Map — the embedding scatterplot over raw_records + raw_embeddings.
// ---------------------------------------------------------------------------

pub async fn map_view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let (nodes, edges) = compute_graph(&state.pool, &user.user_id).await?;
    Ok(Html(views::map_view(&user.user_id, nodes.len(), edges)))
}

pub async fn map_data(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<Value>, super::Error> {
    // Build the layout once and derive both nodes and edges from it — the
    // fetch + projection pipeline is the expensive part on this polled endpoint.
    let (nodes, layout) = compute_graph_with_layout(&state.pool, &user.user_id).await?;

    let json_nodes: Vec<Value> = nodes
        .into_iter()
        .map(|n| {
            json!({
                "id": n.id,
                "type": n.type_,
                "label": n.label,
                "content": n.content,
                "importance": n.importance,
                "superseded": n.superseded,
                "x": n.x,
                "y": n.y,
                "x3": n.x3,
                "y3": n.y3,
                "z3": n.z3,
            })
        })
        .collect();

    let json_edges: Vec<Value> = layout
        .edges
        .into_iter()
        .map(|e| {
            json!({
                "source": e.source,
                "target": e.target,
                "kind": e.kind,
                "weight": e.weight,
            })
        })
        .collect();

    Ok(Json(json!({
        "nodes": json_nodes,
        "edges": json_edges,
    })))
}

// ---------------------------------------------------------------------------
// Curation — the trigger for the raw-derived curated layer.
// ---------------------------------------------------------------------------

pub async fn curate_view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let counts = curated_summary(&state.pool, &user.user_id).await?;
    Ok(Html(views::curate_view(
        &user.user_id,
        &counts,
        state.nlp.provider_can_distill(),
        state.nlp.provider_name(),
    )))
}

/// Per-kind curated node counts (episodic / semantic / …) for the curation page.
pub(crate) async fn curated_summary(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<Vec<(String, i64)>, super::Error> {
    let rows: Vec<(String, i64)> = sqlx::query_as(
        r#"
        SELECT kind, COUNT(*) AS n
        FROM curated_nodes
        WHERE user_id = $1
        GROUP BY kind
        ORDER BY kind
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Glass-box trigger for the curated layer: promote working raw records to
/// episodic + distill semantic facts for the caller, then redirect back.
pub async fn curate_trigger(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Response, super::Error> {
    match crate::curation::rebuild(&state.pool, &*state.nlp, &user.user_id).await {
        Ok(stats) => {
            tracing::info!(
                promoted = stats.promoted,
                distilled = stats.distilled,
                clusters = stats.clusters_seen,
                distill_skipped = stats.skipped_distill,
                "manual curation run complete"
            );
            Ok(Redirect::to("/admin/curate").into_response())
        }
        Err(e) => {
            tracing::error!("curation run failed: {e}");
            Ok((StatusCode::INTERNAL_SERVER_ERROR, "curation failed").into_response())
        }
    }
}

// ---------------------------------------------------------------------------
// Graph plumbing
// ---------------------------------------------------------------------------

pub(crate) struct MapNode {
    pub id: Uuid,
    pub type_: String,
    pub label: String,
    pub content: String,
    pub importance: f32,
    pub superseded: bool,
    pub x: f32,
    pub y: f32,
    pub x3: f32,
    pub y3: f32,
    pub z3: f32,
}

struct GraphRow {
    id: Uuid,
    r#type: String,
    content: String,
    importance: f32,
    superseded: bool,
    session_id: Option<String>,
    supersedes: Option<Uuid>,
    state_key: Option<String>,
    embedding: Option<Vector>,
    entities: Vec<String>,
}

/// Edge count only — for the map page header. Delegates to the full builder.
async fn compute_graph(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<(Vec<MapNode>, usize), super::Error> {
    let (nodes, layout) = compute_graph_with_layout(pool, user_id).await?;
    let edge_count = layout.edges.len();
    Ok((nodes, edge_count))
}

/// Fetch the caller's graph rows once, project them, and return both the render
/// nodes and the full layout (coords + edges). `map_data` needs the edges too,
/// so it reuses this instead of running the fetch+projection pipeline twice.
async fn compute_graph_with_layout(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<(Vec<MapNode>, GraphLayout), super::Error> {
    let rows = fetch_for_graph(pool, user_id).await?;
    let inputs: Vec<GraphInput> = rows
        .iter()
        .map(|r| GraphInput {
            id: r.id,
            embedding: r.embedding.as_ref().map(|v| v.to_vec()).unwrap_or_default(),
            entities: r.entities.clone(),
            session_id: r.session_id.clone(),
            supersedes: r.supersedes,
        })
        .collect();
    let layout = build_graph(&inputs);

    let nodes = rows
        .into_iter()
        .map(|r| {
            let (x, y) = layout.coords.get(&r.id).copied().unwrap_or((0.0, 0.0));
            let (x3, y3, z3) = layout
                .coords_3d
                .get(&r.id)
                .copied()
                .unwrap_or((0.0, 0.0, 0.0));
            let label = node_label(&r);
            let content = preview(&r.content, 220);
            MapNode {
                id: r.id,
                type_: r.r#type,
                label,
                content,
                importance: r.importance,
                superseded: r.superseded,
                x,
                y,
                x3,
                y3,
                z3,
            }
        })
        .collect::<Vec<_>>();
    Ok((nodes, layout))
}

/// The graph pulls terminal + superseded raw records with their default-model
/// embedding (LEFT JOIN — a record with no embedding yet still plots at origin)
/// and their indexed entities. Capped at 500 to keep the O(N²) Jaccard scan +
/// browser-side force sim happy.
async fn fetch_for_graph(pool: &sqlx::PgPool, user_id: &str) -> Result<Vec<GraphRow>, sqlx::Error> {
    #[derive(sqlx::FromRow)]
    struct Row {
        id: Uuid,
        r#type: String,
        content: String,
        importance: Option<f32>,
        superseded: bool,
        session_id: Option<String>,
        supersedes: Option<Uuid>,
        state_key: Option<String>,
        embedding: Option<Vector>,
    }
    let rows = sqlx::query_as::<_, Row>(
        r#"
        SELECT r.id, r.type, r.content, r.importance, r.session_id, r.supersedes, r.state_key,
               EXISTS (
                   SELECT 1 FROM raw_records n
                   WHERE n.supersedes = r.id AND n.user_id = r.user_id
               ) AS superseded,
               e.embedding AS embedding
        FROM raw_records r
        LEFT JOIN raw_embeddings e ON e.record_id = r.id
        WHERE r.user_id = $1
        ORDER BY r.event_time DESC
        LIMIT 500
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    let mut out = Vec::with_capacity(rows.len());
    for r in rows {
        let entities = entities_for(pool, user_id, r.id).await?;
        out.push(GraphRow {
            id: r.id,
            r#type: r.r#type,
            content: r.content,
            importance: r.importance.unwrap_or(0.5),
            superseded: r.superseded,
            session_id: r.session_id,
            supersedes: r.supersedes,
            state_key: r.state_key,
            embedding: r.embedding,
            entities,
        });
    }
    Ok(out)
}

fn node_label(r: &GraphRow) -> String {
    if let Some(key) = &r.state_key {
        return key.clone();
    }
    if let Some(first_entity) = r.entities.iter().find(|e| e.contains(' ')) {
        return first_entity.clone();
    }
    if let Some(first) = r.entities.first() {
        return first.clone();
    }
    preview(&r.content, 28)
}

fn preview(s: &str, n: usize) -> String {
    let cleaned: String = s
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");
    if cleaned.chars().count() <= n {
        cleaned
    } else {
        let mut out: String = cleaned.chars().take(n).collect();
        out.push('…');
        out
    }
}

// Suppress unused-headers warning on imports we keep for future use.
#[allow(dead_code)]
fn _hdr(_: HeaderMap) {}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    /// Insert a raw record and return its id. Optionally set `supersedes`.
    async fn insert_raw(
        pool: &PgPool,
        user_id: &str,
        r#type: &str,
        content: &str,
        supersedes: Option<Uuid>,
    ) -> Uuid {
        sqlx::query_scalar(
            r#"
            INSERT INTO raw_records (type, content, event_time, source, user_id, importance, supersedes)
            VALUES ($1, $2, NOW(), 'admin-test', $3, 0.5, $4)
            RETURNING id
            "#,
        )
        .bind(r#type)
        .bind(content)
        .bind(user_id)
        .bind(supersedes)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    // ---- dashboard_counts -------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn dashboard_counts_reflect_raw_world(pool: PgPool) {
        let old = insert_raw(&pool, "alice", "working", "first", None).await;
        // A newer row supersedes `old`; `old` is no longer terminal.
        insert_raw(&pool, "alice", "episodic", "second", Some(old)).await;
        insert_raw(&pool, "alice", "state_object", "todo", None).await;

        let c = dashboard_counts(&pool, "alice").await.unwrap();
        assert_eq!(c.records_total, 3);
        // old is superseded → 2 terminal (the episodic + the state_object).
        assert_eq!(c.records_terminal, 2);
        assert_eq!(c.state_objects, 1);
        assert_eq!(c.curated_nodes, 0);
        assert_eq!(c.proposals_pending, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn dashboard_counts_scoped_by_user(pool: PgPool) {
        insert_raw(&pool, "alice", "working", "a", None).await;
        insert_raw(&pool, "bob", "working", "b", None).await;
        let c = dashboard_counts(&pool, "alice").await.unwrap();
        assert_eq!(c.records_total, 1);
    }

    // ---- fetch_records / mode filter -------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_records_native_mode_filter(pool: PgPool) {
        sqlx::query(
            "INSERT INTO raw_records (type, content, event_time, source, user_id, mode)
             VALUES ('working', 'code note', NOW(), 't', 'alice', 'code')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO raw_records (type, content, event_time, source, user_id, mode)
             VALUES ('working', 'journal note', NOW(), 't', 'alice', 'journal')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let q = RecordQuery {
            mode: Some("code".to_string()),
            ..RecordQuery::default()
        };
        let rows = fetch_records(&pool, "alice", &q, 50, 0).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content, "code note");
        assert_eq!(count_records(&pool, "alice", &q).await.unwrap(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_records_excludes_superseded_by_default(pool: PgPool) {
        let old = insert_raw(&pool, "alice", "working", "old", None).await;
        insert_raw(&pool, "alice", "episodic", "new", Some(old)).await;

        let terminal_only = fetch_records(&pool, "alice", &RecordQuery::default(), 50, 0)
            .await
            .unwrap();
        assert_eq!(terminal_only.len(), 1);
        assert_eq!(terminal_only[0].content, "new");

        let with_super = RecordQuery {
            include_superseded: Some("1".to_string()),
            ..RecordQuery::default()
        };
        let all = fetch_records(&pool, "alice", &with_super, 50, 0)
            .await
            .unwrap();
        assert_eq!(all.len(), 2);
    }

    // ---- record detail ----------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_record_detail_returns_chain(pool: PgPool) {
        let old = insert_raw(&pool, "alice", "working", "v1", None).await;
        let new = insert_raw(&pool, "alice", "episodic", "v2", Some(old)).await;
        let (row, chain) = load_record_detail(&pool, "alice", new).await.unwrap();
        assert_eq!(row.id, new);
        assert_eq!(chain.len(), 2);
        // oldest first
        assert_eq!(chain[0].content, "v1");
        assert_eq!(chain[1].content, "v2");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_record_detail_rejects_wrong_user(pool: PgPool) {
        let id = insert_raw(&pool, "alice", "working", "x", None).await;
        let err = load_record_detail(&pool, "bob", id).await.unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_record_detail_404_when_missing(pool: PgPool) {
        let err = load_record_detail(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_record_detail_chain_never_crosses_user_boundary(pool: PgPool) {
        // Alice's record; then a row owned by BOB whose supersedes points at
        // alice's id (simulates a cross-user supersede pointer — a corrupted or
        // hostile insert). The chain walk must stay within alice and never pull
        // bob's row into her detail page.
        let alice = insert_raw(&pool, "alice", "working", "alice v1", None).await;
        let bob_cross = insert_raw(&pool, "bob", "episodic", "bob's row", Some(alice)).await;

        let (row, chain) = load_record_detail(&pool, "alice", alice).await.unwrap();
        assert_eq!(row.id, alice);
        // Only alice's own row is in the chain — bob's cross-user row is excluded.
        assert!(chain.iter().all(|r| r.id != bob_cross));
        assert!(chain.iter().any(|r| r.id == alice));
    }

    // ---- state object list is capped -------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_state_objects_is_capped(pool: PgPool) {
        // Insert more terminal state_objects than the 200 cap and confirm the
        // list never returns an unbounded page.
        for i in 0..205 {
            sqlx::query(
                "INSERT INTO raw_records (type, content, event_time, source, user_id, payload) \
                 VALUES ('state_object', $1, NOW(), 't', 'alice', \
                         jsonb_build_object('kind','note','key', $2, 'data', $1))",
            )
            .bind(format!("note {i}"))
            .bind(format!("k{i}"))
            .execute(&pool)
            .await
            .unwrap();
        }
        let rows = list_state_objects_for(&pool, "alice").await.unwrap();
        assert!(
            rows.len() <= 200,
            "state list must be capped, got {}",
            rows.len()
        );
    }

    // ---- state objects ----------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_state_objects_returns_terminal_only(pool: PgPool) {
        let old = insert_raw(&pool, "alice", "state_object", "todo v1", None).await;
        insert_raw(&pool, "alice", "state_object", "todo v2", Some(old)).await;
        let rows = list_state_objects_for(&pool, "alice").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].content, "todo v2");
    }

    // ---- revoke_token_owned_by -------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoke_token_owned_by_marks_revoked(pool: PgPool) {
        let minted = crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        revoke_token_owned_by(&pool, "alice", minted.id)
            .await
            .unwrap();
        let revoked_at: Option<chrono::DateTime<chrono::Utc>> =
            sqlx::query_scalar("SELECT revoked_at FROM tokens WHERE id = $1")
                .bind(minted.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(revoked_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoke_token_owned_by_doesnt_affect_other_users_tokens(pool: PgPool) {
        let alice_token = crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        revoke_token_owned_by(&pool, "bob", alice_token.id)
            .await
            .unwrap();
        let revoked_at: Option<chrono::DateTime<chrono::Utc>> =
            sqlx::query_scalar("SELECT revoked_at FROM tokens WHERE id = $1")
                .bind(alice_token.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(
            revoked_at.is_none(),
            "bob's call should not revoke alice's token"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoke_token_owned_by_unknown_id_is_silent_ok(pool: PgPool) {
        revoke_token_owned_by(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap();
    }

    // ---- pure helpers ----------------------------------------------------

    #[test]
    fn empty_to_none_collapses_whitespace() {
        assert_eq!(empty_to_none(""), None);
        assert_eq!(empty_to_none("   "), None);
        assert_eq!(empty_to_none("abc"), Some("abc".to_string()));
        assert_eq!(empty_to_none("  trimmed  "), Some("trimmed".to_string()));
    }

    #[test]
    fn record_query_include_super_truthy_values() {
        for v in ["1", "true", "on"] {
            let q = RecordQuery {
                include_superseded: Some(v.to_string()),
                ..RecordQuery::default()
            };
            assert!(q.include_super(), "{v} should be truthy");
        }
        for v in ["0", "false", "off", ""] {
            let q = RecordQuery {
                include_superseded: Some(v.to_string()),
                ..RecordQuery::default()
            };
            assert!(!q.include_super(), "{v} should be falsy");
        }
    }

    #[test]
    fn preview_truncates_and_joins_lines() {
        assert_eq!(preview("a\nb\nc", 100), "a · b · c");
        let long = "x".repeat(50);
        let p = preview(&long, 10);
        assert!(p.chars().count() <= 11); // 10 + ellipsis
        assert!(p.ends_with('…'));
    }
}
