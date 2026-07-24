pub mod admin;
pub mod context;
pub mod core;
pub mod health;
pub mod memory;
pub mod records;
pub mod state;

use axum::{routing::get, Router};

use crate::AppState;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health::health_check))
        .nest("/admin", admin::router(state.clone()))
        .nest("/core", core::router(state.clone()))
        .nest("/memory", memory::router(state.clone()))
        .nest("/state", state::router(state.clone()))
        .nest("/context", context::router(state.clone()))
        .nest("/records", records::router(state.clone()))
        .route("/lineage/{id}", get(memory::lineage))
        .with_state(state)
}
