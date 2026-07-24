pub mod admin;
pub mod health;
pub mod modes;
pub mod records;

use crate::AppState;
use axum::Router;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", axum::routing::get(health::health_check))
        .nest("/admin", admin::router(state.clone()))
        .nest("/records", records::router(state.clone()))
        .nest("/modes", modes::router(state.clone()))
        .nest("/catalog", crate::catalog::router(state.clone()))
        .nest("/proposals", crate::proposals::router(state.clone()))
        .with_state(state)
}
