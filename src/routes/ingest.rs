use axum::{
    extract::{Json, State},
    http::StatusCode,
    routing::post,
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::PgPool;

#[derive(Debug, Deserialize)]
pub struct ConversationIngest {
    pub conversation_id: String,
    pub messages: Vec<Value>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct DocumentIngest {
    pub title: String,
    pub content: String,
    pub source: Option<String>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct BulkIngest {
    pub items: Vec<Value>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub accepted: usize,
    pub job_id: String,
}

pub fn router(pool: PgPool) -> Router {
    Router::new()
        .route("/conversation", post(ingest_conversation))
        .route("/document", post(ingest_document))
        .route("/bulk", post(ingest_bulk))
        .with_state(pool)
}

async fn ingest_conversation(
    State(_pool): State<PgPool>,
    Json(payload): Json<ConversationIngest>,
) -> (StatusCode, Json<Value>) {
    let job_id = uuid::Uuid::new_v4().to_string();
    tracing::info!(conversation_id = %payload.conversation_id, "Ingesting conversation");
    (
        StatusCode::ACCEPTED,
        Json(json!({ "job_id": job_id, "status": "queued" })),
    )
}

async fn ingest_document(
    State(_pool): State<PgPool>,
    Json(payload): Json<DocumentIngest>,
) -> (StatusCode, Json<Value>) {
    let job_id = uuid::Uuid::new_v4().to_string();
    tracing::info!(title = %payload.title, "Ingesting document");
    (
        StatusCode::ACCEPTED,
        Json(json!({ "job_id": job_id, "status": "queued" })),
    )
}

async fn ingest_bulk(
    State(_pool): State<PgPool>,
    Json(payload): Json<BulkIngest>,
) -> (StatusCode, Json<Value>) {
    let job_id = uuid::Uuid::new_v4().to_string();
    let count = payload.items.len();
    tracing::info!(count, "Bulk ingest");
    (
        StatusCode::ACCEPTED,
        Json(json!({ "job_id": job_id, "accepted": count, "status": "queued" })),
    )
}
