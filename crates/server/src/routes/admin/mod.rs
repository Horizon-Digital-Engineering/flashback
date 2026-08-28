//! Browser admin UI.
//!
//! Server-rendered HTML, cookie-based session, zero external JS/CSS deps
//! (per the project's "minimize third parties" stance). It renders the
//! canonical world: raw records, the curated layer, references, the data
//! catalog, and the operator proposal queue. Routes:
//!
//!   GET  /admin/login          — login form
//!   POST /admin/login          — accept token, set cookie, redirect
//!   GET  /admin/logout         — clear cookie, redirect to login
//!   GET  /admin                — dashboard
//!   GET  /admin/records        — raw record list (filterable, native mode filter)
//!   GET  /admin/records/:id    — raw record detail (entities + supersede chain)
//!   GET  /admin/curated        — curated node list
//!   GET  /admin/state          — references (state_object raw records)
//!   GET  /admin/catalog        — data catalog
//!   GET  /admin/proposals      — proposal queue
//!   POST /admin/proposals/:id/approve|deny
//!   GET  /admin/tokens         — token management
//!   POST /admin/tokens/:id/revoke         — revoke a token
//!   GET  /admin/map            — interactive embedding scatterplot
//!   GET  /admin/api/map.json   — projected coordinates for the scatterplot
//!   GET  /admin/curate         — curation status + trigger
//!   POST /admin/api/curate     — run an incremental curation pass for this user
//!   GET  /admin/settings       — system provider settings page
//!   POST /admin/api/settings   — save settings + hot-swap the provider
//!   POST /admin/api/settings/models — models the configured endpoint serves
//!   POST /admin/api/settings/test   — one real extraction with draft settings
//!   GET  /admin/style.css      — embedded stylesheet

use axum::{
    routing::{get, post},
    Router,
};

use crate::AppState;

mod handlers;
pub(crate) mod playground;
mod projection;
mod settings_api;
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
        .route("/records", get(handlers::records_list))
        .route("/records/{id}", get(handlers::record_detail))
        .route("/curated", get(handlers::curated_list))
        .route("/state", get(handlers::state_list))
        .route("/catalog", get(handlers::catalog_view))
        .route("/proposals", get(handlers::proposals_view))
        .route("/proposals/{id}/approve", post(handlers::proposal_approve))
        .route("/proposals/{id}/deny", post(handlers::proposal_deny))
        .route("/tokens", get(handlers::tokens_list))
        .route("/tokens/{id}/revoke", post(handlers::token_revoke))
        .route("/map", get(handlers::map_view))
        .route("/api/map.json", get(handlers::map_data))
        .route("/curate", get(handlers::curate_view))
        .route("/api/curate", post(handlers::curate_trigger))
        .route("/playground", get(handlers::playground_view))
        .route("/api/playground/turn", post(playground::turn))
        .route("/api/playground/settings", post(playground::save_settings))
        .route("/settings", get(settings_api::view))
        .route("/api/settings", post(settings_api::save))
        .route("/api/settings/models", post(settings_api::models))
        .route("/api/settings/test", post(settings_api::test_extraction))
        .with_state(state)
}
