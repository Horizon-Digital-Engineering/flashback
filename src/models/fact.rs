use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Fact {
    pub id: Uuid,
    pub content: String,
    pub source: Option<String>,
    pub confidence: f64,
    pub entity_ids: Vec<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct CreateFact {
    pub content: String,
    pub source: Option<String>,
    pub confidence: Option<f64>,
    pub entity_ids: Option<Vec<Uuid>>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,
    pub metadata: Option<serde_json::Value>,
}
