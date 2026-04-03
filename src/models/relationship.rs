use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct Relationship {
    pub id: Uuid,
    pub from_entity_id: Uuid,
    pub to_entity_id: Uuid,
    pub relation_type: String,
    pub weight: f64,
    pub attributes: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct CreateRelationship {
    pub from_entity_id: Uuid,
    pub to_entity_id: Uuid,
    pub relation_type: String,
    pub weight: Option<f64>,
    pub attributes: Option<serde_json::Value>,
    pub valid_from: Option<DateTime<Utc>>,
    pub valid_until: Option<DateTime<Utc>>,
}
