//! Browser admin UI.
//!
//! Server-rendered HTML, cookie-based session, zero external JS/CSS deps
//! (per the project's "minimize third parties" stance). Routes:
//!
//!   GET  /admin/login          — login form
//!   POST /admin/login          — accept token, set cookie, redirect
//!   GET  /admin/logout         — clear cookie, redirect to login
//!   GET  /admin                — dashboard
//!   GET  /admin/memories       — list (filterable, paginated)
//!   GET  /admin/memories/:id   — detail (extraction + lineage)
//!   POST /admin/memories/:id/delete       — hard delete
//!   POST /admin/memories/:id/supersede    — mark superseded by a new content
//!   GET  /admin/state          — list state objects
//!   GET  /admin/tokens         — token management
//!   POST /admin/tokens/:id/revoke         — revoke a token
//!   GET  /admin/map            — interactive embedding scatterplot
//!   GET  /admin/api/map.json   — projected coordinates for the scatterplot
//!   GET  /admin/style.css      — embedded stylesheet
//!
//! Phase 3 evolution path: the /admin/map view will get a Three.js 3D
//! renderer + true UMAP projection. Today it ships as 2D PCA in SVG —
//! same data, less spectacle, zero JS dependencies.

use axum::{
    routing::{get, post},
    Router,
};

use crate::AppState;

mod handlers;
mod projection;
mod style;
pub mod views;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};

#[derive(Debug)]
pub enum Error {
    Db(sqlx::Error),
    NotFound,
}

impl From<sqlx::Error> for Error {
    fn from(e: sqlx::Error) -> Self {
        Error::Db(e)
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        match self {
            Error::NotFound => (StatusCode::NOT_FOUND, "not found").into_response(),
            Error::Db(e) => {
                tracing::error!("admin db error: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, "internal error").into_response()
            }
        }
    }
}

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/", get(handlers::dashboard))
        .route(
            "/login",
            get(handlers::login_form).post(handlers::login_submit),
        )
        .route("/logout", get(handlers::logout))
        .route("/style.css", get(handlers::style_css))
        .route("/memories", get(handlers::memories_list))
        .route("/memories/{id}", get(handlers::memory_detail))
        .route("/memories/{id}/delete", post(handlers::memory_delete))
        .route("/state", get(handlers::state_list))
        .route("/catalog", get(handlers::catalog_view))
        .route("/proposals", get(handlers::proposals_view))
        .route("/proposals/{id}/approve", post(handlers::proposal_approve))
        .route("/proposals/{id}/deny", post(handlers::proposal_deny))
        .route("/tokens", get(handlers::tokens_list))
        .route("/tokens/{id}/revoke", post(handlers::token_revoke))
        .route("/map", get(handlers::map_view))
        .route("/api/map.json", get(handlers::map_data))
        .route("/consolidate", get(handlers::consolidate_view))
        .route(
            "/api/consolidate/{kind}",
            post(handlers::consolidate_trigger),
        )
        .route("/api/curate", post(handlers::curate_trigger))
        .with_state(state)
}
