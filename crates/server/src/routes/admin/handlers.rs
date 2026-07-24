//! Axum handlers for the admin UI. Hand-rolled HTML — no templating crate.

use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
    Form, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    auth::{self, AuthUser},
    models::{MemoryRow, MemoryView},
    AppState,
};

use super::{
    projection::{build_graph, GraphInput},
    style::STYLE_CSS,
    views,
};
use crate::consolidation;

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
    let recent_rows =
        fetch_memories(&state.pool, &user.user_id, &MemoryQuery::default(), 10, 0).await?;
    let stats = views::DashboardStats {
        memories_total: counts.memories_total,
        memories_terminal: counts.memories_terminal,
        state_objects: counts.state_objects,
        tokens_active: counts.tokens_active,
        provider: state.nlp.provider_name().to_string(),
        embedder_model: state.nlp.embedder_model_name().to_string(),
        embedder_dim: state.nlp.embedder_dimension(),
    };
    let recent: Vec<MemoryView> = recent_rows.into_iter().map(MemoryView::from).collect();
    Ok(Html(views::dashboard(&user.user_id, stats, &recent)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DashboardCounts {
    pub memories_total: i64,
    pub memories_terminal: i64,
    pub state_objects: i64,
    pub tokens_active: i64,
}

/// The four `SELECT COUNT(*)` queries that feed the dashboard sidebar.
/// Extracted so the dashboard summary is unit-testable without rendering
/// HTML or constructing an AppState — call with a pool + user_id.
pub(crate) async fn dashboard_counts(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<DashboardCounts, super::Error> {
    let memories_total: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memories WHERE user_id = $1")
            .bind(user_id)
            .fetch_one(pool)
            .await?;
    let memories_terminal: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND superseded_by IS NULL",
    )
    .bind(user_id)
    .fetch_one(pool)
    .await?;
    let state_objects: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND type = 'state_object' AND superseded_by IS NULL",
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
        memories_total,
        memories_terminal,
        state_objects,
        tokens_active,
    })
}

// ---------------------------------------------------------------------------
// Memories list
// ---------------------------------------------------------------------------

#[derive(Default, Deserialize)]
pub struct MemoryQuery {
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

impl MemoryQuery {
    fn include_super(&self) -> bool {
        matches!(
            self.include_superseded.as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    }
    fn clean(&self) -> Self {
        Self {
            r#type: self.r#type.as_ref().and_then(empty_to_none),
            project_id: self.project_id.as_ref().and_then(empty_to_none),
            session_id: self.session_id.as_ref().and_then(empty_to_none),
            mode: self.mode.as_ref().and_then(empty_to_none),
            include_superseded: if self.include_super() {
                Some("1".to_string())
            } else {
                None
            },
        }
    }
}

fn empty_to_none(s: &String) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

pub async fn memories_list(
    State(state): State<AppState>,
    user: AuthUser,
    Query(q): Query<MemoryQuery>,
) -> Result<Html<String>, super::Error> {
    let q = q.clean();
    let rows = fetch_memories(&state.pool, &user.user_id, &q, 200, 0).await?;
    let total = count_memories(&state.pool, &user.user_id, &q).await?;
    let memories: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();

    // The user's registers, for the mode-filter dropdown.
    let mode_names: Vec<String> = crate::modes::list_modes(&state.pool, &user.user_id)
        .await
        .map(|ms| ms.into_iter().map(|m| m.name).collect())
        .unwrap_or_default();

    let filter = views::MemoriesFilter {
        r#type: q.r#type.clone(),
        project_id: q.project_id.clone(),
        session_id: q.session_id.clone(),
        mode: q.mode.clone(),
        include_superseded: q.include_super(),
    };
    Ok(Html(views::memories_list(
        &user.user_id,
        &filter,
        &mode_names,
        &memories,
        total,
    )))
}

async fn fetch_memories(
    pool: &sqlx::PgPool,
    user_id: &str,
    q: &MemoryQuery,
    limit: i64,
    offset: i64,
) -> Result<Vec<MemoryRow>, sqlx::Error> {
    sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT m.id, m.type, m.content, m.embedding, m.importance, m.access_count, m.decay_class,
               m.user_id, m.project_id, m.session_id, m.entities, m.superseded_by, m.supersedes,
               m.source_path, m.chunk_index, m.content_hash, m.state_kind, m.state_key, m.state_data,
               m.expires_at, m.created_at, m.updated_at, m.last_accessed_at
        FROM memories m
        WHERE m.user_id = $1
          AND ($2::TEXT IS NULL OR m.type = $2)
          AND ($3::TEXT IS NULL OR m.project_id = $3)
          AND ($4::TEXT IS NULL OR m.session_id = $4)
          AND ($5::BOOLEAN OR m.superseded_by IS NULL)
          AND ($8::TEXT IS NULL OR EXISTS (
              SELECT 1 FROM raw_records rr
              WHERE rr.user_id = m.user_id AND rr.content = m.content AND rr.mode = $8
          ))
        ORDER BY m.created_at DESC
        LIMIT $6 OFFSET $7
        "#,
    )
    .bind(user_id)
    .bind(&q.r#type)
    .bind(&q.project_id)
    .bind(&q.session_id)
    .bind(q.include_super())
    .bind(limit)
    .bind(offset)
    .bind(&q.mode)
    .fetch_all(pool)
    .await
}

async fn count_memories(
    pool: &sqlx::PgPool,
    user_id: &str,
    q: &MemoryQuery,
) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM memories m
        WHERE m.user_id = $1
          AND ($2::TEXT IS NULL OR m.type = $2)
          AND ($3::TEXT IS NULL OR m.project_id = $3)
          AND ($4::TEXT IS NULL OR m.session_id = $4)
          AND ($5::BOOLEAN OR m.superseded_by IS NULL)
          AND ($6::TEXT IS NULL OR EXISTS (
              SELECT 1 FROM raw_records rr
              WHERE rr.user_id = m.user_id AND rr.content = m.content AND rr.mode = $6
          ))
        "#,
    )
    .bind(user_id)
    .bind(&q.r#type)
    .bind(&q.project_id)
    .bind(&q.session_id)
    .bind(q.include_super())
    .bind(&q.mode)
    .fetch_one(pool)
    .await
}

// ---------------------------------------------------------------------------
// Memory detail
// ---------------------------------------------------------------------------

pub async fn memory_detail(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Html<String>, super::Error> {
    let (row, chain_rows, extraction) = load_memory_detail(&state.pool, &user.user_id, id).await?;
    let chain: Vec<MemoryView> = chain_rows.into_iter().map(MemoryView::from).collect();
    let view = MemoryView::from(row);
    Ok(Html(views::memory_detail(
        &user.user_id,
        &view,
        &chain,
        extraction.as_ref(),
    )))
}

/// Load a single memory + its supersede chain + the stored extraction blob.
/// Returns NotFound if the row is missing or owned by a different user — the
/// handler turns that into a 404.
pub(crate) async fn load_memory_detail(
    pool: &sqlx::PgPool,
    user_id: &str,
    id: Uuid,
) -> Result<(MemoryRow, Vec<MemoryRow>, Option<Value>), super::Error> {
    let row = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?
    .ok_or(super::Error::NotFound)?;

    if row.user_id != user_id {
        return Err(super::Error::NotFound);
    }

    let chain_rows = sqlx::query_as::<_, MemoryRow>(
        r#"
        WITH RECURSIVE
        back AS (
            SELECT m.*, 0 AS depth FROM memories m WHERE m.id = $1
            UNION ALL
            SELECT prev.*, b.depth - 1 FROM memories prev JOIN back b ON prev.id = b.supersedes
        ),
        forward AS (
            SELECT m.*, 0 AS depth FROM memories m WHERE m.id = $1
            UNION ALL
            SELECT nxt.*, f.depth + 1 FROM memories nxt JOIN forward f ON nxt.supersedes = f.id
        )
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM (SELECT id, type, content, embedding, importance, access_count, decay_class,
                     user_id, project_id, session_id, entities, superseded_by, supersedes,
                     source_path, chunk_index, content_hash, state_kind, state_key, state_data,
                     expires_at, created_at, updated_at, last_accessed_at, depth FROM back
              UNION
              SELECT id, type, content, embedding, importance, access_count, decay_class,
                     user_id, project_id, session_id, entities, superseded_by, supersedes,
                     source_path, chunk_index, content_hash, state_kind, state_key, state_data,
                     expires_at, created_at, updated_at, last_accessed_at, depth FROM forward) AS u
        ORDER BY created_at ASC
        "#,
    )
    .bind(id)
    .fetch_all(pool)
    .await?;

    let extraction: Option<Value> =
        sqlx::query_scalar("SELECT extraction FROM memories WHERE id = $1")
            .bind(id)
            .fetch_optional(pool)
            .await?
            .flatten();

    Ok((row, chain_rows, extraction))
}

pub async fn memory_delete(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Response, super::Error> {
    delete_memory_owned_by(&state.pool, &user.user_id, id).await?;
    Ok(Redirect::to("/admin/memories").into_response())
}

/// Hard-delete a memory after verifying it belongs to `user_id`. Nulls out
/// inbound supersede pointers so the DELETE doesn't violate FK constraints.
pub(crate) async fn delete_memory_owned_by(
    pool: &sqlx::PgPool,
    user_id: &str,
    id: Uuid,
) -> Result<(), super::Error> {
    let owner: Option<String> = sqlx::query_scalar("SELECT user_id FROM memories WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    if owner.as_deref() != Some(user_id) {
        return Err(super::Error::NotFound);
    }
    sqlx::query("UPDATE memories SET superseded_by = NULL WHERE superseded_by = $1")
        .bind(id)
        .execute(pool)
        .await?;
    sqlx::query("UPDATE memories SET supersedes = NULL WHERE supersedes = $1")
        .bind(id)
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM memories WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// State objects
// ---------------------------------------------------------------------------

pub async fn state_list(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let rows = list_state_objects_for(&state.pool, &user.user_id).await?;
    let views: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();
    Ok(Html(views::state_list(&user.user_id, &views)))
}

/// Fetch terminal (non-superseded) state objects owned by `user_id`, ordered
/// by most-recently-accessed first.
pub(crate) async fn list_state_objects_for(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<Vec<MemoryRow>, sqlx::Error> {
    sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1 AND type = 'state_object' AND superseded_by IS NULL
        ORDER BY last_accessed_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
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
// Map
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
    let (nodes, _) = compute_graph(&state.pool, &user.user_id).await?;
    // Re-run to get the structured edge list — separated so map_view can
    // count without doing the full work twice.
    let rows = fetch_for_graph(&state.pool, &user.user_id).await?;
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
// Consolidation
// ---------------------------------------------------------------------------

#[derive(Debug, sqlx::FromRow, Serialize)]
struct ConsolidationRunRow {
    id: Uuid,
    kind: String,
    trigger: String,
    started_at: DateTime<Utc>,
    finished_at: Option<DateTime<Utc>>,
    promoted_count: i32,
    expired_count: i32,
    distilled_count: i32,
    clusters_seen: i32,
    user_id: Option<String>,
    error: Option<String>,
}

pub async fn consolidate_view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let rows: Vec<ConsolidationRunRow> = sqlx::query_as(
        r#"
        SELECT id, kind, trigger, started_at, finished_at, promoted_count,
               expired_count, distilled_count, clusters_seen, user_id, error
        FROM consolidation_runs
        ORDER BY started_at DESC
        LIMIT 50
        "#,
    )
    .fetch_all(&state.pool)
    .await?;
    let provider_can_distill = state.nlp.provider_can_distill();
    Ok(Html(views::consolidate_view(
        &user.user_id,
        &rows,
        provider_can_distill,
        state.nlp.provider_name(),
    )))
}

pub async fn consolidate_trigger(
    State(state): State<AppState>,
    user: AuthUser,
    Path(kind): Path<String>,
) -> Result<Response, super::Error> {
    match run_consolidate_kind(&state.pool, &state.nlp, &user.user_id, &kind).await {
        Ok(stats) => {
            tracing::info!(?stats, "manual consolidation run complete");
            Ok(Redirect::to("/admin/consolidate").into_response())
        }
        Err(ConsolidateError::UnknownKind) => {
            Ok((StatusCode::BAD_REQUEST, "kind must be daily|weekly").into_response())
        }
    }
}

#[derive(Debug)]
pub(crate) enum ConsolidateError {
    UnknownKind,
}

/// Dispatches a manual consolidation run for the given kind. Pulled out of
/// the handler so it's testable without the HTML redirect or AppState.
/// Takes the Arc directly so it can forward into `run_weekly`'s signature.
pub(crate) async fn run_consolidate_kind(
    pool: &sqlx::PgPool,
    nlp: &std::sync::Arc<dyn crate::nlp::NlpService>,
    user_id: &str,
    kind: &str,
) -> Result<consolidation::RunStats, ConsolidateError> {
    match kind {
        "daily" => Ok(consolidation::run_daily(pool, user_id).await),
        "weekly" => Ok(consolidation::run_weekly(pool, nlp, user_id).await),
        _ => Err(ConsolidateError::UnknownKind),
    }
}

/// Glass-box trigger for the NEW curation layer: promote working raw records to
/// episodic + distill semantic facts for the caller's user, then redirect back
/// to the consolidate view. Separate from the legacy consolidate trigger above
/// — this drives `curation`, never the legacy `memories` path.
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
            Ok(Redirect::to("/admin/consolidate").into_response())
        }
        Err(e) => {
            tracing::error!("curation run failed: {e}");
            Ok((StatusCode::INTERNAL_SERVER_ERROR, "curation failed").into_response())
        }
    }
}

struct MapNode {
    id: Uuid,
    type_: String,
    label: String,
    content: String,
    importance: f32,
    superseded: bool,
    x: f32,
    y: f32,
    x3: f32,
    y3: f32,
    z3: f32,
}

async fn compute_graph(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<(Vec<MapNode>, usize), super::Error> {
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
            let preview = preview(&r.content, 220);
            MapNode {
                id: r.id,
                type_: r.r#type,
                label,
                content: preview,
                importance: r.importance,
                superseded: r.superseded_by.is_some(),
                x,
                y,
                x3,
                y3,
                z3,
            }
        })
        .collect::<Vec<_>>();
    let edge_count = layout.edges.len();
    Ok((nodes, edge_count))
}

async fn fetch_for_graph(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<Vec<MemoryRow>, sqlx::Error> {
    // Cap at 500 to keep the O(N²) Jaccard scan + browser-side force sim happy.
    sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1
        ORDER BY last_accessed_at DESC
        LIMIT 500
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
}

fn node_label(r: &MemoryRow) -> String {
    // Prefer extracted topic if present in `extraction.topic`.
    // For state_object, prefer state_key.
    if let Some(key) = &r.state_key {
        return key.clone();
    }
    // We don't deserialize the full extraction here for cost reasons; just
    // pull the topic via a quick JSON string lookup.
    // (For now we rely on entities — the topic column can be used later.)
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

    async fn insert_memory_for(pool: &PgPool, user_id: &str) -> Uuid {
        let id = Uuid::new_v4();
        let emb = pgvector::Vector::from(vec![0.0_f32; 384]);
        sqlx::query(
            "INSERT INTO memories (id, type, content, embedding, importance, decay_class, user_id, entities)
             VALUES ($1, 'episodic', 'admin test', $2, 0.5, 'medium', $3, '{}')",
        )
        .bind(id)
        .bind(emb)
        .bind(user_id)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    // ---- delete_memory_owned_by ------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_removes_row(pool: PgPool) {
        let id = insert_memory_for(&pool, "alice").await;
        delete_memory_owned_by(&pool, "alice", id).await.unwrap();
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories WHERE id = $1")
            .bind(id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_rejects_wrong_user(pool: PgPool) {
        let id = insert_memory_for(&pool, "alice").await;
        let err = delete_memory_owned_by(&pool, "bob", id).await.unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
        // Row still exists.
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories WHERE id = $1")
            .bind(id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_404_when_missing(pool: PgPool) {
        let err = delete_memory_owned_by(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_nulls_supersede_pointers(pool: PgPool) {
        let old = insert_memory_for(&pool, "alice").await;
        let new = insert_memory_for(&pool, "alice").await;
        // Wire supersede chain: new supersedes old, old.superseded_by = new.
        sqlx::query("UPDATE memories SET supersedes = $1 WHERE id = $2")
            .bind(old)
            .bind(new)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new)
            .bind(old)
            .execute(&pool)
            .await
            .unwrap();

        // Deleting `old` should null the inbound pointer on `new` and not FK-violate.
        delete_memory_owned_by(&pool, "alice", old).await.unwrap();

        let supersedes_on_new: Option<Uuid> =
            sqlx::query_scalar("SELECT supersedes FROM memories WHERE id = $1")
                .bind(new)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(supersedes_on_new.is_none());
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
        // Bob tries to revoke alice's token — should silently no-op.
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
        // No row → no error. This matches the handler's UX contract
        // (refresh the tokens page either way).
        revoke_token_owned_by(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap();
    }

    // ---- pure helpers ----------------------------------------------------

    #[test]
    fn empty_to_none_collapses_whitespace() {
        assert_eq!(empty_to_none(&"".to_string()), None);
        assert_eq!(empty_to_none(&"   ".to_string()), None);
        assert_eq!(empty_to_none(&"abc".to_string()), Some("abc".to_string()));
        // Trim is applied.
        assert_eq!(
            empty_to_none(&"  trimmed  ".to_string()),
            Some("trimmed".to_string())
        );
    }

    #[test]
    fn memory_query_include_super_truthy_values() {
        for v in ["1", "true", "on"] {
            let q = MemoryQuery {
                include_superseded: Some(v.to_string()),
                ..MemoryQuery::default()
            };
            assert!(q.include_super(), "{v:?} should be truthy");
        }
    }

    #[test]
    fn memory_query_include_super_falsy_and_unset() {
        let unset = MemoryQuery::default();
        assert!(!unset.include_super());

        for v in ["", "0", "false", "anything-else"] {
            let q = MemoryQuery {
                include_superseded: Some(v.to_string()),
                ..MemoryQuery::default()
            };
            assert!(!q.include_super(), "{v:?} should be falsy");
        }
    }

    #[test]
    fn memory_query_clean_collapses_empties() {
        let raw = MemoryQuery {
            r#type: Some("  ".to_string()),
            project_id: Some("real-project".to_string()),
            session_id: Some("".to_string()),
            mode: None,
            include_superseded: Some("0".to_string()),
        };
        let cleaned = raw.clean();
        assert_eq!(cleaned.r#type, None);
        assert_eq!(cleaned.project_id.as_deref(), Some("real-project"));
        assert_eq!(cleaned.session_id, None);
        // include_superseded normalized to None when falsy.
        assert_eq!(cleaned.include_superseded, None);
    }

    #[test]
    fn memory_query_clean_preserves_truthy_include_super() {
        let raw = MemoryQuery {
            include_superseded: Some("true".to_string()),
            ..MemoryQuery::default()
        };
        let cleaned = raw.clean();
        // Normalized to canonical "1".
        assert_eq!(cleaned.include_superseded.as_deref(), Some("1"));
    }

    // ---- dashboard_counts -----------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn dashboard_counts_empty_db(pool: PgPool) {
        let c = dashboard_counts(&pool, "alice").await.unwrap();
        assert_eq!(c.memories_total, 0);
        assert_eq!(c.memories_terminal, 0);
        assert_eq!(c.state_objects, 0);
        assert_eq!(c.tokens_active, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn dashboard_counts_includes_only_user_rows(pool: PgPool) {
        insert_memory_for(&pool, "alice").await;
        insert_memory_for(&pool, "alice").await;
        insert_memory_for(&pool, "bob").await;
        crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        crate::auth::mint_token(&pool, "bob", None).await.unwrap();

        let alice = dashboard_counts(&pool, "alice").await.unwrap();
        assert_eq!(alice.memories_total, 2);
        assert_eq!(alice.memories_terminal, 2);
        assert_eq!(alice.tokens_active, 1);

        let bob = dashboard_counts(&pool, "bob").await.unwrap();
        assert_eq!(bob.memories_total, 1);
        assert_eq!(bob.tokens_active, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn dashboard_counts_terminal_excludes_superseded(pool: PgPool) {
        let old = insert_memory_for(&pool, "alice").await;
        let new = insert_memory_for(&pool, "alice").await;
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new)
            .bind(old)
            .execute(&pool)
            .await
            .unwrap();

        let c = dashboard_counts(&pool, "alice").await.unwrap();
        assert_eq!(c.memories_total, 2);
        assert_eq!(c.memories_terminal, 1, "superseded row should not count");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn dashboard_counts_tokens_active_excludes_revoked(pool: PgPool) {
        let t1 = crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        crate::auth::revoke_token(&pool, t1.id).await.unwrap();

        let c = dashboard_counts(&pool, "alice").await.unwrap();
        assert_eq!(c.tokens_active, 1);
    }

    // ---- run_consolidate_kind -------------------------------------------

    use async_trait::async_trait;
    use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};

    struct StubNlp;

    #[async_trait]
    impl crate::nlp::NlpService for StubNlp {
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
        async fn embed_one(&self, _: &str) -> Result<Vec<f32>, crate::error::AppError> {
            Ok(vec![0.0_f32; 384])
        }
        async fn embed_batch(
            &self,
            t: Vec<String>,
        ) -> Result<Vec<Vec<f32>>, crate::error::AppError> {
            Ok((0..t.len()).map(|_| vec![0.0_f32; 384]).collect())
        }
        fn extract_entities(&self, _: &str) -> Vec<String> {
            Vec::new()
        }
        async fn extract_full(&self, _: &str) -> Result<Extraction, crate::error::AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            _: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("stub".into()))
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_consolidate_kind_daily_runs(pool: PgPool) {
        let nlp: std::sync::Arc<dyn crate::nlp::NlpService> = std::sync::Arc::new(StubNlp);
        let stats = run_consolidate_kind(&pool, &nlp, "alice", "daily")
            .await
            .unwrap();
        // Empty DB so no promotions / expirations; just an audit row.
        assert_eq!(stats.promoted_count, 0);
        assert_eq!(stats.expired_count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_consolidate_kind_weekly_runs(pool: PgPool) {
        let nlp: std::sync::Arc<dyn crate::nlp::NlpService> = std::sync::Arc::new(StubNlp);
        let stats = run_consolidate_kind(&pool, &nlp, "alice", "weekly")
            .await
            .unwrap();
        // Heuristic provider can't distill — the weekly run logs a warning
        // and reports zero distillations. The fact that this returned Ok
        // (not the UnknownKind error) is the assertion.
        assert_eq!(stats.distilled_count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_consolidate_kind_rejects_unknown(pool: PgPool) {
        let nlp: std::sync::Arc<dyn crate::nlp::NlpService> = std::sync::Arc::new(StubNlp);
        let err = run_consolidate_kind(&pool, &nlp, "alice", "monthly")
            .await
            .unwrap_err();
        assert!(matches!(err, ConsolidateError::UnknownKind));
    }

    // ---- fetch_memories / count_memories --------------------------------

    async fn insert_typed(
        pool: &PgPool,
        user_id: &str,
        mtype: &str,
        project_id: Option<&str>,
        session_id: Option<&str>,
    ) -> Uuid {
        let id = Uuid::new_v4();
        let emb = pgvector::Vector::from(vec![0.0_f32; 384]);
        sqlx::query(
            "INSERT INTO memories (id, type, content, embedding, importance, decay_class, user_id, project_id, session_id, entities)
             VALUES ($1, $2, 'test content', $3, 0.5, 'medium', $4, $5, $6, '{}')",
        )
        .bind(id)
        .bind(mtype)
        .bind(emb)
        .bind(user_id)
        .bind(project_id)
        .bind(session_id)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_memories_empty(pool: PgPool) {
        let q = MemoryQuery::default();
        let rows = fetch_memories(&pool, "alice", &q, 200, 0).await.unwrap();
        assert!(rows.is_empty());
        let total = count_memories(&pool, "alice", &q).await.unwrap();
        assert_eq!(total, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_memories_filters_by_type(pool: PgPool) {
        insert_typed(&pool, "alice", "episodic", None, None).await;
        insert_typed(&pool, "alice", "semantic", None, None).await;
        insert_typed(&pool, "alice", "semantic", None, None).await;

        let only_semantic = MemoryQuery {
            r#type: Some("semantic".to_string()),
            ..MemoryQuery::default()
        };
        let rows = fetch_memories(&pool, "alice", &only_semantic, 200, 0)
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.r#type == "semantic"));

        let total = count_memories(&pool, "alice", &only_semantic)
            .await
            .unwrap();
        assert_eq!(total, 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_memories_filters_by_project_and_session(pool: PgPool) {
        insert_typed(&pool, "alice", "episodic", Some("proj-a"), Some("sess-1")).await;
        insert_typed(&pool, "alice", "episodic", Some("proj-a"), Some("sess-2")).await;
        insert_typed(&pool, "alice", "episodic", Some("proj-b"), Some("sess-1")).await;

        let proj_a = MemoryQuery {
            project_id: Some("proj-a".to_string()),
            ..MemoryQuery::default()
        };
        assert_eq!(count_memories(&pool, "alice", &proj_a).await.unwrap(), 2);

        let proj_a_sess_1 = MemoryQuery {
            project_id: Some("proj-a".to_string()),
            session_id: Some("sess-1".to_string()),
            ..MemoryQuery::default()
        };
        assert_eq!(
            count_memories(&pool, "alice", &proj_a_sess_1)
                .await
                .unwrap(),
            1
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_memories_excludes_superseded_by_default(pool: PgPool) {
        let old = insert_typed(&pool, "alice", "episodic", None, None).await;
        let new = insert_typed(&pool, "alice", "episodic", None, None).await;
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new)
            .bind(old)
            .execute(&pool)
            .await
            .unwrap();

        let default_q = MemoryQuery::default();
        assert_eq!(count_memories(&pool, "alice", &default_q).await.unwrap(), 1);

        let with_super = MemoryQuery {
            include_superseded: Some("1".to_string()),
            ..MemoryQuery::default()
        };
        assert_eq!(
            count_memories(&pool, "alice", &with_super).await.unwrap(),
            2
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_memories_scoped_to_user(pool: PgPool) {
        insert_typed(&pool, "alice", "episodic", None, None).await;
        insert_typed(&pool, "bob", "episodic", None, None).await;
        insert_typed(&pool, "bob", "episodic", None, None).await;

        assert_eq!(
            count_memories(&pool, "alice", &MemoryQuery::default())
                .await
                .unwrap(),
            1
        );
        assert_eq!(
            count_memories(&pool, "bob", &MemoryQuery::default())
                .await
                .unwrap(),
            2
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_memories_respects_limit_offset(pool: PgPool) {
        for _ in 0..5 {
            insert_typed(&pool, "alice", "episodic", None, None).await;
        }
        let rows = fetch_memories(&pool, "alice", &MemoryQuery::default(), 2, 0)
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        let page2 = fetch_memories(&pool, "alice", &MemoryQuery::default(), 2, 2)
            .await
            .unwrap();
        assert_eq!(page2.len(), 2);
        // No overlap.
        assert!(!rows.iter().any(|r| page2.iter().any(|p| p.id == r.id)));
    }

    // ---- load_memory_detail ---------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_memory_detail_404_when_missing(pool: PgPool) {
        let err = load_memory_detail(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_memory_detail_404_for_other_user(pool: PgPool) {
        let id = insert_memory_for(&pool, "alice").await;
        let err = load_memory_detail(&pool, "bob", id).await.unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_memory_detail_returns_singleton_chain(pool: PgPool) {
        let id = insert_memory_for(&pool, "alice").await;
        let (row, chain, _extraction) = load_memory_detail(&pool, "alice", id).await.unwrap();
        assert_eq!(row.id, id);
        // Chain is just this row (no supersede before/after).
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].id, id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_memory_detail_walks_supersede_chain(pool: PgPool) {
        let old = insert_memory_for(&pool, "alice").await;
        let mid = insert_memory_for(&pool, "alice").await;
        let new = insert_memory_for(&pool, "alice").await;
        // old <- mid <- new
        for (prev, next) in [(old, mid), (mid, new)] {
            sqlx::query("UPDATE memories SET supersedes = $1 WHERE id = $2")
                .bind(prev)
                .bind(next)
                .execute(&pool)
                .await
                .unwrap();
            sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
                .bind(next)
                .bind(prev)
                .execute(&pool)
                .await
                .unwrap();
        }

        let (_row, chain, _) = load_memory_detail(&pool, "alice", mid).await.unwrap();
        // All three nodes should be in the chain, in created_at order.
        assert_eq!(chain.len(), 3);
        let ids: Vec<Uuid> = chain.iter().map(|r| r.id).collect();
        assert!(ids.contains(&old));
        assert!(ids.contains(&mid));
        assert!(ids.contains(&new));
    }

    // ---- list_state_objects_for -----------------------------------------

    async fn insert_state(pool: &PgPool, user_id: &str, kind: &str, key: &str) -> Uuid {
        let id = Uuid::new_v4();
        let emb = pgvector::Vector::from(vec![0.0_f32; 384]);
        sqlx::query(
            "INSERT INTO memories (id, type, content, embedding, importance, decay_class, user_id, entities, state_kind, state_key, state_data)
             VALUES ($1, 'state_object', $2, $3, 0.5, 'medium', $4, '{}', $5, $6, '{}'::jsonb)",
        )
        .bind(id)
        .bind(format!("{kind}/{key}"))
        .bind(emb)
        .bind(user_id)
        .bind(kind)
        .bind(key)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_state_objects_empty(pool: PgPool) {
        let rows = list_state_objects_for(&pool, "alice").await.unwrap();
        assert!(rows.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_state_objects_excludes_episodic(pool: PgPool) {
        insert_memory_for(&pool, "alice").await; // episodic
        insert_state(&pool, "alice", "todo", "list-1").await;
        let rows = list_state_objects_for(&pool, "alice").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].r#type, "state_object");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_state_objects_excludes_superseded(pool: PgPool) {
        let old = insert_state(&pool, "alice", "todo", "list-1").await;
        let new = insert_state(&pool, "alice", "todo", "list-1").await;
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new)
            .bind(old)
            .execute(&pool)
            .await
            .unwrap();

        let rows = list_state_objects_for(&pool, "alice").await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, new);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_state_objects_scoped_to_user(pool: PgPool) {
        insert_state(&pool, "alice", "todo", "a").await;
        insert_state(&pool, "bob", "todo", "b").await;
        assert_eq!(
            list_state_objects_for(&pool, "alice").await.unwrap().len(),
            1
        );
        assert_eq!(list_state_objects_for(&pool, "bob").await.unwrap().len(), 1);
    }

    // ---- load_token_views_for -------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_token_views_empty(pool: PgPool) {
        let v = load_token_views_for(&pool, "alice").await.unwrap();
        assert!(v.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_token_views_includes_revoked(pool: PgPool) {
        let t = crate::auth::mint_token(&pool, "alice", Some("primary"))
            .await
            .unwrap();
        crate::auth::revoke_token(&pool, t.id).await.unwrap();
        let v = load_token_views_for(&pool, "alice").await.unwrap();
        assert_eq!(v.len(), 1);
        assert!(v[0].revoked_at.is_some());
        assert_eq!(v[0].name.as_deref(), Some("primary"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_token_views_scoped_to_user(pool: PgPool) {
        crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        crate::auth::mint_token(&pool, "bob", None).await.unwrap();
        crate::auth::mint_token(&pool, "bob", None).await.unwrap();
        assert_eq!(load_token_views_for(&pool, "alice").await.unwrap().len(), 1);
        assert_eq!(load_token_views_for(&pool, "bob").await.unwrap().len(), 2);
    }

    // ---- fetch_for_graph + compute_graph --------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_for_graph_empty(pool: PgPool) {
        let rows = fetch_for_graph(&pool, "alice").await.unwrap();
        assert!(rows.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn fetch_for_graph_scoped_to_user(pool: PgPool) {
        insert_memory_for(&pool, "alice").await;
        insert_memory_for(&pool, "bob").await;
        let alice = fetch_for_graph(&pool, "alice").await.unwrap();
        let bob = fetch_for_graph(&pool, "bob").await.unwrap();
        assert_eq!(alice.len(), 1);
        assert_eq!(bob.len(), 1);
        assert_ne!(alice[0].id, bob[0].id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn compute_graph_returns_node_per_row(pool: PgPool) {
        let a = insert_memory_for(&pool, "alice").await;
        let b = insert_memory_for(&pool, "alice").await;
        let (nodes, _edges) = compute_graph(&pool, "alice").await.unwrap();
        assert_eq!(nodes.len(), 2);
        let ids: Vec<Uuid> = nodes.iter().map(|n| n.id).collect();
        assert!(ids.contains(&a));
        assert!(ids.contains(&b));
        // Sanity: nodes carry preview content and type.
        for n in &nodes {
            assert_eq!(n.type_, "episodic");
            assert!(!n.label.is_empty());
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn compute_graph_marks_superseded_nodes(pool: PgPool) {
        let old = insert_memory_for(&pool, "alice").await;
        let new = insert_memory_for(&pool, "alice").await;
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new)
            .bind(old)
            .execute(&pool)
            .await
            .unwrap();
        let (nodes, _) = compute_graph(&pool, "alice").await.unwrap();
        let old_node = nodes.iter().find(|n| n.id == old).unwrap();
        let new_node = nodes.iter().find(|n| n.id == new).unwrap();
        assert!(old_node.superseded);
        assert!(!new_node.superseded);
    }
}
