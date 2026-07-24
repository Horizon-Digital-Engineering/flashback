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
use sqlx::PgPool;
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
        .route("/{id}", get(get_one).put(update).delete(delete))
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

// ---- Handlers (3-line shells over the testable *_impl fns) -----------

async fn list(State(app): State<AppState>, auth_user: AuthUser) -> AppResult<Json<ListResponse>> {
    Ok(Json(list_core(&app.pool, &auth_user.user_id).await?))
}

async fn create(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<CreateRequest>,
) -> AppResult<Json<CoreMemoryRow>> {
    Ok(Json(create_core(&app.pool, &auth_user.user_id, req).await?))
}

async fn get_one(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<CoreMemoryRow>> {
    Ok(Json(get_core(&app.pool, &auth_user.user_id, id).await?))
}

async fn update(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateRequest>,
) -> AppResult<Json<CoreMemoryRow>> {
    Ok(Json(
        update_core(&app.pool, &auth_user.user_id, id, req).await?,
    ))
}

async fn delete(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<serde_json::Value>> {
    delete_core(&app.pool, &auth_user.user_id, id).await?;
    Ok(Json(json!({ "deleted": id })))
}

// ---- Pool-only logic — directly unit-testable -------------------------

pub(crate) async fn list_core(pool: &PgPool, user_id: &str) -> AppResult<ListResponse> {
    let items: Vec<CoreMemoryRow> = sqlx::query_as::<_, CoreMemoryRow>(
        r#"SELECT id, user_id, content, importance, pinned_at, updated_at
           FROM core_memory WHERE user_id = $1
           ORDER BY importance DESC, pinned_at ASC"#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    let count = items.len();
    Ok(ListResponse { items, count })
}

pub(crate) async fn create_core(
    pool: &PgPool,
    user_id: &str,
    req: CreateRequest,
) -> AppResult<CoreMemoryRow> {
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
    .bind(user_id)
    .bind(req.content.trim())
    .bind(importance)
    .fetch_one(pool)
    .await?;
    Ok(row)
}

pub(crate) async fn get_core(pool: &PgPool, user_id: &str, id: Uuid) -> AppResult<CoreMemoryRow> {
    let row: Option<CoreMemoryRow> = sqlx::query_as::<_, CoreMemoryRow>(
        r#"SELECT id, user_id, content, importance, pinned_at, updated_at
           FROM core_memory WHERE id = $1 AND user_id = $2"#,
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(pool)
    .await?;
    row.ok_or_else(|| AppError::not_found(format!("core memory {id}")))
}

pub(crate) async fn update_core(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
    req: UpdateRequest,
) -> AppResult<CoreMemoryRow> {
    let row: Option<CoreMemoryRow> = sqlx::query_as::<_, CoreMemoryRow>(
        r#"UPDATE core_memory
           SET content    = COALESCE($3, content),
               importance = COALESCE($4, importance)
           WHERE id = $1 AND user_id = $2
           RETURNING id, user_id, content, importance, pinned_at, updated_at"#,
    )
    .bind(id)
    .bind(user_id)
    .bind(req.content.as_deref())
    .bind(req.importance.map(|v| v.clamp(0.0, 1.0)))
    .fetch_optional(pool)
    .await?;
    row.ok_or_else(|| AppError::not_found(format!("core memory {id}")))
}

pub(crate) async fn delete_core(pool: &PgPool, user_id: &str, id: Uuid) -> AppResult<()> {
    let result = sqlx::query("DELETE FROM core_memory WHERE id = $1 AND user_id = $2")
        .bind(id)
        .bind(user_id)
        .execute(pool)
        .await?;
    if result.rows_affected() == 0 {
        return Err(AppError::not_found(format!("core memory {id}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_req(content: &str, importance: Option<f32>) -> CreateRequest {
        CreateRequest {
            content: content.to_string(),
            importance,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_empty_returns_no_items(pool: PgPool) {
        let resp = list_core(&pool, "alice").await.unwrap();
        assert_eq!(resp.count, 0);
        assert!(resp.items.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_then_list_round_trip(pool: PgPool) {
        let created = create_core(&pool, "alice", create_req("alpha", Some(0.7)))
            .await
            .unwrap();
        assert_eq!(created.user_id, "alice");
        assert_eq!(created.content, "alpha");
        assert!((created.importance - 0.7).abs() < 1e-6);

        let listed = list_core(&pool, "alice").await.unwrap();
        assert_eq!(listed.count, 1);
        assert_eq!(listed.items[0].id, created.id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_rejects_empty_content(pool: PgPool) {
        let err = create_core(&pool, "alice", create_req("", None))
            .await
            .unwrap_err();
        let _ = err;
        let err2 = create_core(&pool, "alice", create_req("   ", None))
            .await
            .unwrap_err();
        let _ = err2;
        // No rows inserted.
        assert_eq!(list_core(&pool, "alice").await.unwrap().count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_clamps_importance_and_defaults_to_one(pool: PgPool) {
        let over = create_core(&pool, "alice", create_req("over", Some(5.0)))
            .await
            .unwrap();
        assert!((over.importance - 1.0).abs() < 1e-6);

        let under = create_core(&pool, "alice", create_req("under", Some(-3.0)))
            .await
            .unwrap();
        assert!(under.importance.abs() < 1e-6);

        let default = create_core(&pool, "alice", create_req("default", None))
            .await
            .unwrap();
        assert!((default.importance - 1.0).abs() < 1e-6);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_trims_content(pool: PgPool) {
        let row = create_core(&pool, "alice", create_req("  hello  ", None))
            .await
            .unwrap();
        assert_eq!(row.content, "hello");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_scopes_to_user_id(pool: PgPool) {
        create_core(&pool, "alice", create_req("a1", None))
            .await
            .unwrap();
        create_core(&pool, "alice", create_req("a2", None))
            .await
            .unwrap();
        create_core(&pool, "bob", create_req("b1", None))
            .await
            .unwrap();

        assert_eq!(list_core(&pool, "alice").await.unwrap().count, 2);
        assert_eq!(list_core(&pool, "bob").await.unwrap().count, 1);
        assert_eq!(list_core(&pool, "carol").await.unwrap().count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_returns_404_for_wrong_user(pool: PgPool) {
        let row = create_core(&pool, "alice", create_req("private", None))
            .await
            .unwrap();
        // Bob can't read alice's core memory.
        assert!(get_core(&pool, "bob", row.id).await.is_err());
        // Alice can.
        assert_eq!(get_core(&pool, "alice", row.id).await.unwrap().id, row.id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_returns_404_for_unknown_id(pool: PgPool) {
        let bogus = Uuid::new_v4();
        assert!(get_core(&pool, "alice", bogus).await.is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn update_changes_content_and_importance(pool: PgPool) {
        let row = create_core(&pool, "alice", create_req("v1", Some(0.5)))
            .await
            .unwrap();

        let updated = update_core(
            &pool,
            "alice",
            row.id,
            UpdateRequest {
                content: Some("v2".to_string()),
                importance: Some(0.9),
            },
        )
        .await
        .unwrap();
        assert_eq!(updated.content, "v2");
        assert!((updated.importance - 0.9).abs() < 1e-6);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn update_with_none_fields_is_a_partial_update(pool: PgPool) {
        let row = create_core(&pool, "alice", create_req("keep_me", Some(0.5)))
            .await
            .unwrap();

        // Only update importance; content stays.
        let updated = update_core(
            &pool,
            "alice",
            row.id,
            UpdateRequest {
                content: None,
                importance: Some(0.8),
            },
        )
        .await
        .unwrap();
        assert_eq!(updated.content, "keep_me");
        assert!((updated.importance - 0.8).abs() < 1e-6);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn update_rejects_wrong_user(pool: PgPool) {
        let row = create_core(&pool, "alice", create_req("alice's", None))
            .await
            .unwrap();
        let err = update_core(
            &pool,
            "bob",
            row.id,
            UpdateRequest {
                content: Some("hijacked".to_string()),
                importance: None,
            },
        )
        .await
        .unwrap_err();
        let _ = err;

        // Alice's data is unchanged.
        let still = get_core(&pool, "alice", row.id).await.unwrap();
        assert_eq!(still.content, "alice's");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_removes_only_owners_row(pool: PgPool) {
        let row = create_core(&pool, "alice", create_req("doomed", None))
            .await
            .unwrap();

        // Bob trying to delete fails.
        assert!(delete_core(&pool, "bob", row.id).await.is_err());
        // Alice still sees it.
        assert!(get_core(&pool, "alice", row.id).await.is_ok());

        // Alice deletes; now it's gone.
        delete_core(&pool, "alice", row.id).await.unwrap();
        assert!(get_core(&pool, "alice", row.id).await.is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_unknown_id_errors(pool: PgPool) {
        let bogus = Uuid::new_v4();
        assert!(delete_core(&pool, "alice", bogus).await.is_err());
    }
}
