pub mod core_memory;
pub mod health;
pub mod ingest;
pub mod query;

use axum::{routing::get, Router};
use sqlx::PgPool;

pub fn router(pool: PgPool) -> Router {
    Router::new()
        .route("/health", get(health::health_check))
        .nest("/ingest", ingest::router(pool.clone()))
        .nest("/query", query::router(pool.clone()))
        .nest("/memory", core_memory::router(pool))
}
