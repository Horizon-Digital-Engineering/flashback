//! Core memory: pinned, always-injected context.
//!
//! user_id always comes from the auth token.

use axum::{
    extract::{Path, State},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    models::CoreMemoryRow,
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/", get(list).post(create))
        .route("/:id", get(get_one).put(update).delete(delete))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
pub struct CreateRequest {
    pub content: String,
    #[serde(default)]
    pub importance: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateRequest {
    pub content: Option<String>,
    pub importance: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ListResponse {
    pub items: Vec<CoreMemoryRow>,
    pub count: usize,
}

async fn list(State(app): State<AppState>, auth_user: AuthUser) -> AppResult<Json<ListResponse>> {
    let items: Vec<CoreMemoryRow> = sqlx::query_as::<_, CoreMemoryRow>(
        r#"SELECT id, user_id, content, importance, pinned_at, updated_at
           FROM core_memory WHERE user_id = $1
           ORDER BY importance DESC, pinned_at ASC"#,
    )
    .bind(&auth_user.user_id)
    .fetch_all(&app.pool)
    .await?;

    let count = items.len();
    Ok(Json(ListResponse { items, count }))
}

async fn create(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<CreateRequest>,
) -> AppResult<Json<CoreMemoryRow>> {
    if req.content.trim().is_empty() {
        return Err(AppError::bad_request("content cannot be empty"));
    }
    let importance = req.importance.unwrap_or(1.0).clamp(0.0, 1.0);
    let id = Uuid::new_v4();

    let row: CoreMemoryRow = sqlx::query_as::<_, CoreMemoryRow>(
        r#"INSERT INTO core_memory (id, user_id, content, importance)
           VALUES ($1, $2, $3, $4)
           RETURNING id, user_id, content, importance, pinned_at, updated_at"#,
    )
    .bind(id)
    .bind(&auth_user.user_id)
    .bind(req.content.trim())
    .bind(importance)
    .fetch_one(&app.pool)
    .await?;
    Ok(Json(row))
}

async fn get_one(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<CoreMemoryRow>> {
    let row: Option<CoreMemoryRow> = sqlx::query_as::<_, CoreMemoryRow>(
        r#"SELECT id, user_id, content, importance, pinned_at, updated_at
           FROM core_memory WHERE id = $1 AND user_id = $2"#,
    )
    .bind(id)
    .bind(&auth_user.user_id)
    .fetch_optional(&app.pool)
    .await?;
    row.map(Json)
        .ok_or_else(|| AppError::not_found(format!("core memory {id}")))
}

async fn update(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateRequest>,
) -> AppResult<Json<CoreMemoryRow>> {
    let row: Option<CoreMemoryRow> = sqlx::query_as::<_, CoreMemoryRow>(
        r#"UPDATE core_memory
           SET content    = COALESCE($3, content),
               importance = COALESCE($4, importance)
           WHERE id = $1 AND user_id = $2
           RETURNING id, user_id, content, importance, pinned_at, updated_at"#,
    )
    .bind(id)
    .bind(&auth_user.user_id)
    .bind(req.content.as_deref())
    .bind(req.importance.map(|v| v.clamp(0.0, 1.0)))
    .fetch_optional(&app.pool)
    .await?;

    row.map(Json)
        .ok_or_else(|| AppError::not_found(format!("core memory {id}")))
}

async fn delete(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<serde_json::Value>> {
    let result = sqlx::query("DELETE FROM core_memory WHERE id = $1 AND user_id = $2")
        .bind(id)
        .bind(&auth_user.user_id)
        .execute(&app.pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(AppError::not_found(format!("core memory {id}")));
    }
    Ok(Json(json!({ "deleted": id })))
}
