//! state_object routes — the reference half of memory.
//!
//! POST   /state/:kind              create
//! GET    /state/:kind/:key         current value (terminal node)
//! PATCH  /state/:kind/:key         apply op → new terminal node
//! GET    /state/:kind/:key/history supersede chain
//!
//! user_id is always derived from the auth token; never accepted from request
//! bodies or query strings.

use axum::{
    extract::{Path, State},
    routing::get,
    Json, Router,
};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    models::{MemoryRow, MemoryView},
    state::StateKind,
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/:kind", axum::routing::post(create))
        .route("/:kind/:key", get(get_current).patch(patch).put(replace))
        .route("/:kind/:key/history", get(history))
        .with_state(state)
}

#[derive(Debug)]
struct OwnedScope {
    user_id: String,
    project_id: Option<String>,
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    pub state_key: String,
    #[serde(default)]
    pub initial: Option<Value>,
    #[serde(default)]
    pub importance: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct StateView {
    pub id: Uuid,
    pub state_kind: String,
    pub state_key: String,
    pub user_id: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub state_data: Value,
    pub rendered: String,
    pub supersedes: Option<Uuid>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

async fn create(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(kind_str): Path<String>,
    Json(req): Json<CreateRequest>,
) -> AppResult<Json<StateView>> {
    let kind = StateKind::parse(&kind_str)?;

    if req.state_key.trim().is_empty() {
        return Err(AppError::bad_request("state_key cannot be empty"));
    }

    let existing: Option<(Uuid,)> = sqlx::query_as(
        r#"
        SELECT id FROM memories
        WHERE user_id = $1 AND state_kind = $2 AND state_key = $3
          AND superseded_by IS NULL
        LIMIT 1
        "#,
    )
    .bind(&auth_user.user_id)
    .bind(kind.as_str())
    .bind(&req.state_key)
    .fetch_optional(&app.pool)
    .await?;

    if let Some((id,)) = existing {
        return Err(AppError::Conflict(format!(
            "state object {}/{} already exists ({id})",
            kind.as_str(),
            req.state_key
        )));
    }

    let initial_data = match req.initial {
        Some(v) => kind.validate_initial(&v)?,
        None => kind.empty(),
    };

    let scope = OwnedScope {
        user_id: auth_user.user_id.clone(),
        project_id: req.project_id,
        session_id: req.session_id,
    };

    let new_id = insert_state(
        &app,
        kind,
        &req.state_key,
        &scope,
        &initial_data,
        req.importance,
        None,
    )
    .await?;

    let row = fetch_state(&app.pool, new_id).await?;
    Ok(Json(to_state_view(kind, row)?))
}

async fn get_current(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
) -> AppResult<Json<StateView>> {
    let kind = StateKind::parse(&kind_str)?;
    let row = fetch_terminal(&app.pool, &auth_user.user_id, kind, &key).await?;
    Ok(Json(to_state_view(kind, row)?))
}

#[derive(Debug, Deserialize)]
pub struct PatchRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(flatten)]
    pub op: Value,
}

async fn patch(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
    Json(req): Json<PatchRequest>,
) -> AppResult<Json<StateView>> {
    let kind = StateKind::parse(&kind_str)?;

    let current = fetch_terminal(&app.pool, &auth_user.user_id, kind, &key).await?;
    let current_data = current
        .state_data
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state row missing state_data")))?;

    let new_data = kind.apply(&current_data, &req.op)?;

    let scope = OwnedScope {
        user_id: auth_user.user_id.clone(),
        project_id: req.project_id.or(current.project_id.clone()),
        session_id: req.session_id.or(current.session_id.clone()),
    };

    let new_id = insert_state(
        &app,
        kind,
        &key,
        &scope,
        &new_data,
        Some(current.importance),
        Some(current.id),
    )
    .await?;

    sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
        .bind(new_id)
        .bind(current.id)
        .execute(&app.pool)
        .await?;

    let row = fetch_state(&app.pool, new_id).await?;
    Ok(Json(to_state_view(kind, row)?))
}

#[derive(Debug, Deserialize)]
pub struct ReplaceRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    pub state_data: Value,
    #[serde(default)]
    pub importance: Option<f32>,
}

async fn replace(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
    Json(req): Json<ReplaceRequest>,
) -> AppResult<Json<StateView>> {
    let kind = StateKind::parse(&kind_str)?;
    let current = fetch_terminal(&app.pool, &auth_user.user_id, kind, &key).await?;
    let validated = kind.validate_initial(&req.state_data)?;

    let scope = OwnedScope {
        user_id: auth_user.user_id.clone(),
        project_id: req.project_id.or(current.project_id.clone()),
        session_id: req.session_id.or(current.session_id.clone()),
    };

    let new_id = insert_state(
        &app,
        kind,
        &key,
        &scope,
        &validated,
        req.importance.or(Some(current.importance)),
        Some(current.id),
    )
    .await?;

    sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
        .bind(new_id)
        .bind(current.id)
        .execute(&app.pool)
        .await?;

    let row = fetch_state(&app.pool, new_id).await?;
    Ok(Json(to_state_view(kind, row)?))
}

async fn history(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    let kind = StateKind::parse(&kind_str)?;
    let rows: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1 AND state_kind = $2 AND state_key = $3
        ORDER BY created_at ASC
        "#,
    )
    .bind(&auth_user.user_id)
    .bind(kind.as_str())
    .bind(&key)
    .fetch_all(&app.pool)
    .await?;

    if rows.is_empty() {
        return Err(AppError::not_found(format!(
            "state object {}/{key}",
            kind.as_str()
        )));
    }

    let terminal = rows
        .iter()
        .find(|r| r.superseded_by.is_none())
        .map(|r| r.id);
    let chain: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();

    Ok(Json(json!({
        "state_kind": kind.as_str(),
        "state_key": key,
        "terminal": terminal,
        "length": chain.len(),
        "chain": chain,
    })))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn fetch_state(pool: &sqlx::PgPool, id: Uuid) -> AppResult<MemoryRow> {
    sqlx::query_as::<_, MemoryRow>(
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
    .ok_or_else(|| AppError::not_found(format!("memory {id}")))
}

async fn fetch_terminal(
    pool: &sqlx::PgPool,
    user_id: &str,
    kind: StateKind,
    key: &str,
) -> AppResult<MemoryRow> {
    sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1 AND state_kind = $2 AND state_key = $3
          AND superseded_by IS NULL
        LIMIT 1
        "#,
    )
    .bind(user_id)
    .bind(kind.as_str())
    .bind(key)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| AppError::not_found(format!("state object {}/{key}", kind.as_str())))
}

async fn insert_state(
    app: &AppState,
    kind: StateKind,
    key: &str,
    scope: &OwnedScope,
    data: &Value,
    importance: Option<f32>,
    supersedes: Option<Uuid>,
) -> AppResult<Uuid> {
    let rendered = kind.render(data, key);
    let embedding = app.nlp.embed_one(&rendered).await?;
    let vector = Vector::from(embedding);

    let new_id = Uuid::new_v4();
    sqlx::query(
        r#"
        INSERT INTO memories (
            id, type, content, embedding, importance, decay_class,
            user_id, project_id, session_id, entities,
            state_kind, state_key, state_data,
            supersedes
        ) VALUES (
            $1, 'state_object', $2, $3, $4, 'none',
            $5, $6, $7, '{}',
            $8, $9, $10,
            $11
        )
        "#,
    )
    .bind(new_id)
    .bind(&rendered)
    .bind(vector)
    .bind(importance.unwrap_or(0.8).clamp(0.0, 1.0))
    .bind(&scope.user_id)
    .bind(&scope.project_id)
    .bind(&scope.session_id)
    .bind(kind.as_str())
    .bind(key)
    .bind(data)
    .bind(supersedes)
    .execute(&app.pool)
    .await?;
    Ok(new_id)
}

fn to_state_view(kind: StateKind, row: MemoryRow) -> AppResult<StateView> {
    let data = row
        .state_data
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state row missing state_data")))?;
    let key = row
        .state_key
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state row missing state_key")))?;
    let rendered = kind.render(&data, &key);

    Ok(StateView {
        id: row.id,
        state_kind: kind.as_str().to_string(),
        state_key: key,
        user_id: row.user_id,
        project_id: row.project_id,
        session_id: row.session_id,
        state_data: data,
        rendered,
        supersedes: row.supersedes,
        created_at: row.created_at,
    })
}

// Used by /context/assemble Layer 2.
pub async fn fetch_active_state_objects(
    pool: &sqlx::PgPool,
    user_id: &str,
    project_id: Option<&str>,
    session_id: Option<&str>,
    limit: usize,
) -> AppResult<Vec<MemoryView>> {
    let rows: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1
          AND type = 'state_object'
          AND superseded_by IS NULL
          AND ($2::TEXT IS NULL OR project_id = $2 OR project_id IS NULL)
          AND ($3::TEXT IS NULL OR session_id = $3 OR session_id IS NULL)
        ORDER BY (CASE WHEN session_id = $3 THEN 0 ELSE 1 END),
                 (CASE WHEN project_id = $2 THEN 0 ELSE 1 END),
                 importance DESC,
                 last_accessed_at DESC
        LIMIT $4
        "#,
    )
    .bind(user_id)
    .bind(project_id)
    .bind(session_id)
    .bind(limit as i64)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(MemoryView::from).collect())
}
