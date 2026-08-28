use axum::{extract::State, Json};
use serde_json::{json, Value};

use crate::AppState;

pub async fn health_check(State(state): State<AppState>) -> Json<Value> {
    let db_ok = sqlx::query_scalar::<_, i32>("SELECT 1")
        .fetch_one(&state.pool)
        .await
        .is_ok();
    let models = state.nlp.provider_models();

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
                // The models the live provider resolved to — null for
                // model-less providers. A box silently downgraded to the
                // heuristic provider is visible right here.
                "extract_model": models.as_ref().map(|(e, _)| e.clone()),
                "distill_model": models.as_ref().map(|(_, d)| d.clone()),
            }
        }
    }))
}
