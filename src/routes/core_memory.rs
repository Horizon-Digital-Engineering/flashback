use axum::{
    extract::{Json, Path, State},
    routing::{get, put},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::PgPool;

#[derive(Debug, Deserialize, Serialize)]
pub struct CoreMemoryEntry {
    pub key: String,
    pub value: Value,
}

pub fn router(pool: PgPool) -> Router {
    Router::new()
        .route("/:key", get(get_memory))
        .route("/:key", put(put_memory))
        .with_state(pool)
}

async fn get_memory(
    State(_pool): State<PgPool>,
    Path(key): Path<String>,
) -> Json<Value> {
    tracing::info!(key = %key, "Get core memory");
    Json(json!({ "key": key, "value": null, "exists": false }))
}

async fn put_memory(
    State(_pool): State<PgPool>,
    Path(key): Path<String>,
    Json(body): Json<Value>,
) -> Json<Value> {
    tracing::info!(key = %key, "Put core memory");
    Json(json!({ "key": key, "updated": true }))
}
