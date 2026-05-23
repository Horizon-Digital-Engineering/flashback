use axum::{extract::State, Json};
use serde_json::{json, Value};

use crate::AppState;

pub async fn health_check(State(state): State<AppState>) -> Json<Value> {
    let db_ok = sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&state.pool)
        .await
        .is_ok();

    let embedder = state.nlp.embedder();

    Json(json!({
        "status": if db_ok { "ok" } else { "degraded" },
        "service": "flashback",
        "dev_mode": state.cfg.dev_mode,
        "db": { "ok": db_ok },
        "nlp": {
            "embedder": {
                "loaded": true,
                "name": embedder.model_name(),
                "dimension": embedder.dimension(),
            },
            "extractor": {
                "provider": state.nlp.provider_name(),
            }
        }
    }))
}
