use axum::{
    extract::{Json, Query, State},
    routing::get,
    Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use sqlx::PgPool;

#[derive(Debug, Deserialize)]
pub struct SearchParams {
    pub q: String,
    pub limit: Option<usize>,
    pub threshold: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct TimelineParams {
    pub entity_id: Option<String>,
    pub from: Option<String>,
    pub until: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct GraphParams {
    pub entity_id: String,
    pub depth: Option<usize>,
    pub relation_types: Option<String>,
}

pub fn router(pool: PgPool) -> Router {
    Router::new()
        .route("/search", get(search))
        .route("/timeline", get(timeline))
        .route("/graph", get(graph))
        .with_state(pool)
}

async fn search(
    State(_pool): State<PgPool>,
    Query(params): Query<SearchParams>,
) -> Json<Value> {
    tracing::info!(query = %params.q, "Vector search");
    Json(json!({ "results": [], "query": params.q, "total": 0 }))
}

async fn timeline(
    State(_pool): State<PgPool>,
    Query(params): Query<TimelineParams>,
) -> Json<Value> {
    tracing::info!("Timeline query");
    Json(json!({ "events": [], "entity_id": params.entity_id }))
}

async fn graph(
    State(_pool): State<PgPool>,
    Query(params): Query<GraphParams>,
) -> Json<Value> {
    let depth = params.depth.unwrap_or(2);
    tracing::info!(entity_id = %params.entity_id, depth, "Graph traversal");
    Json(json!({ "nodes": [], "edges": [], "root": params.entity_id }))
}
