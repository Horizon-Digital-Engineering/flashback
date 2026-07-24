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
