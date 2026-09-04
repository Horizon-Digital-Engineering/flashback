//! The modes (cognitive registers) door. Behind the same bearer auth as every
//! other endpoint; every operation is scoped to the caller's user.
//!
//!   GET    /modes         list the caller's registers (built-ins auto-seeded)
//!   POST   /modes         create a user-defined register
//!   GET    /modes/:name   one register
//!   PUT    /modes/:name   update a register (built-in or user-defined)
//!   DELETE /modes/:name   delete a user-defined register (built-ins protected)

use axum::{
    extract::{Path, State},
    routing::get,
    Json, Router,
};
use serde::Deserialize;

use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    modes::{self, Mode, UpsertModeRequest},
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/", get(list).post(create))
        .route("/{name}", get(get_one).put(update).delete(remove))
        .with_state(state)
}

async fn list(State(state): State<AppState>, auth: AuthUser) -> AppResult<Json<Vec<Mode>>> {
    Ok(Json(modes::list_modes(&state.pool, &auth.user_id).await?))
}

#[derive(Debug, Deserialize)]
struct CreateModeRequest {
    name: String,
    #[serde(flatten)]
    body: UpsertModeRequest,
}

async fn create(
    State(state): State<AppState>,
    auth: AuthUser,
    Json(req): Json<CreateModeRequest>,
) -> AppResult<Json<Mode>> {
    Ok(Json(
        modes::create_mode(&state.pool, &auth.user_id, &req.name, req.body).await?,
    ))
}

async fn get_one(
    State(state): State<AppState>,
    auth: AuthUser,
    Path(name): Path<String>,
) -> AppResult<Json<Mode>> {
    modes::get_mode(&state.pool, &auth.user_id, &name)
        .await?
        .map(Json)
        .ok_or_else(|| AppError::not_found(format!("mode {name}")))
}

async fn update(
    State(state): State<AppState>,
    auth: AuthUser,
    Path(name): Path<String>,
    Json(req): Json<UpsertModeRequest>,
) -> AppResult<Json<Mode>> {
    Ok(Json(
        modes::update_mode(&state.pool, &auth.user_id, &name, req).await?,
    ))
}

async fn remove(
    State(state): State<AppState>,
    auth: AuthUser,
    Path(name): Path<String>,
) -> AppResult<Json<serde_json::Value>> {
    modes::delete_mode(&state.pool, &auth.user_id, &name).await?;
    Ok(Json(serde_json::json!({ "deleted": name })))
}

#[cfg(test)]
mod tests {
    use crate::testsupport::{authed_router, state_from};
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use sqlx::PgPool;
    use tower::ServiceExt;

    async fn call(
        pool: PgPool,
        user: &str,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder().method(method).uri(path);
        let req = match body {
            Some(b) => req
                .header("content-type", "application/json")
                .body(Body::from(b.to_string()))
                .unwrap(),
            None => req.body(Body::empty()).unwrap(),
        };
        let r = authed_router(state_from(pool), user, crate::auth::TokenRole::Service)
            .oneshot(req)
            .await
            .unwrap();
        let status = r.status();
        let bytes = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
        (status, json)
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn the_builtins_are_there_before_anyone_creates_anything(pool: PgPool) {
        let (code, body) = call(pool, "alice", "GET", "/modes", None).await;
        assert_eq!(code, StatusCode::OK);
        let names: Vec<&str> = body
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        for builtin in crate::modes::BUILTIN_MODES {
            assert!(names.contains(builtin), "{builtin} missing from {names:?}");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_register_belongs_to_the_user_who_made_it(pool: PgPool) {
        let (code, _) = call(
            pool.clone(),
            "alice",
            "POST",
            "/modes",
            Some(serde_json::json!({
                "name": "recipes",
                "embedder": "all-MiniLM-L6-v2",
                "description": "cooking"
            })),
        )
        .await;
        assert_eq!(code, StatusCode::OK);

        let (code, _) = call(pool.clone(), "alice", "GET", "/modes/recipes", None).await;
        assert_eq!(code, StatusCode::OK);

        let (code, body) = call(pool.clone(), "bob", "GET", "/modes/recipes", None).await;
        assert_eq!(
            code,
            StatusCode::NOT_FOUND,
            "bob reached alice's register: {body}"
        );

        let (_, list) = call(pool, "bob", "GET", "/modes", None).await;
        let names: Vec<&str> = list
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["name"].as_str().unwrap())
            .collect();
        assert!(!names.contains(&"recipes"), "{names:?}");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_builtin_cannot_be_deleted_but_can_be_edited(pool: PgPool) {
        let (code, _) = call(pool.clone(), "alice", "DELETE", "/modes/general", None).await;
        assert_ne!(code, StatusCode::OK, "the default register was deletable");

        let (code, body) = call(
            pool.clone(),
            "alice",
            "PUT",
            "/modes/general",
            Some(serde_json::json!({
                "embedder": "all-MiniLM-L6-v2",
                "description": "mine now"
            })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{body}");
        assert_eq!(body["description"], "mine now");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn an_embedder_the_server_cannot_load_is_refused_at_the_door(pool: PgPool) {
        // A register whose embedder does not exist would accept records and
        // then fail to embed every one of them.
        let (code, _) = call(
            pool.clone(),
            "alice",
            "POST",
            "/modes",
            Some(serde_json::json!({ "name": "bad", "embedder": "not-a-model" })),
        )
        .await;
        assert_eq!(code, StatusCode::BAD_REQUEST);

        let (code, _) = call(
            pool.clone(),
            "alice",
            "POST",
            "/modes",
            Some(serde_json::json!({
                "name": "mismatched",
                "embedder": "all-MiniLM-L6-v2",
                "embedding_dim": 768
            })),
        )
        .await;
        assert_eq!(
            code,
            StatusCode::BAD_REQUEST,
            "a dim that disagrees with the embedder writes into the wrong column"
        );

        let (code, _) = call(pool, "alice", "GET", "/modes/bad", None).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn deleting_a_register_that_was_never_there_is_not_a_success(pool: PgPool) {
        let (code, _) = call(pool, "alice", "DELETE", "/modes/nothing-here", None).await;
        assert_ne!(code, StatusCode::OK);
    }
}
