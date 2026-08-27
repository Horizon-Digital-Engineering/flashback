use axum::{extract::State, Json};
use serde_json::{json, Value};

use crate::AppState;

pub async fn health_check(State(state): State<AppState>) -> Json<Value> {
    let db_ok = sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&state.pool)
        .await
        .is_ok();

    Json(json!({
        "status": if db_ok { "ok" } else { "degraded" },
        "service": "flashback",
        "build": crate::build_info::as_json(),
        "dev_mode": state.cfg.dev_mode,
        "db": { "ok": db_ok },
        "nlp": {
            "embedder": {
                "loaded": true,
                "name": state.nlp.embedder_model_name(),
                "dimension": state.nlp.embedder_dimension(),
            },
            "extractor": {
                "provider": state.nlp.provider_name(),
            }
        }
    }))
}
