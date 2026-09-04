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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::state_from;
    use sqlx::PgPool;

    #[sqlx::test(migrations = "../../migrations")]
    async fn health_reports_ok_when_the_database_answers(pool: PgPool) {
        let Json(v) = health_check(State(state_from(pool))).await;
        assert_eq!(v["status"], "ok");
        assert_eq!(v["service"], "flashback");
        assert_eq!(v["db"]["ok"], true);
        assert_eq!(v["dev_mode"], false);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn health_names_the_embedder_and_the_extractor(pool: PgPool) {
        // This endpoint is how you notice a box silently running the heuristic
        // provider, so the provider name has to be real rather than hardcoded.
        let Json(v) = health_check(State(state_from(pool))).await;
        assert_eq!(v["nlp"]["embedder"]["name"], "test-embedder");
        assert_eq!(v["nlp"]["embedder"]["dimension"], 384);
        assert_eq!(v["nlp"]["extractor"]["provider"], "test");
        assert!(
            v["nlp"]["extractor"]["extract_model"].is_null(),
            "a model-less provider reports null, not a guess"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn health_reports_degraded_when_the_database_is_gone(pool: PgPool) {
        let state = state_from(pool);
        state.pool.close().await;
        let Json(v) = health_check(State(state)).await;
        assert_eq!(
            v["status"], "degraded",
            "a dead database must not report ok"
        );
        assert_eq!(v["db"]["ok"], false);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn health_carries_build_information(pool: PgPool) {
        let Json(v) = health_check(State(state_from(pool))).await;
        assert!(
            v["build"].is_object(),
            "build info is how you tell which commit is deployed"
        );
    }
}
