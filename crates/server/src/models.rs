use chrono::{DateTime, Utc};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;

/// One of the six top-level memory kinds.
/// The first five are records (immutable history). The last is references
/// (mutable named state). See docs/REFERENCES.md.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    Episodic,
    Semantic,
    Working,
    Document,
    Procedural,
    StateObject,
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Episodic => "episodic",
            Self::Semantic => "semantic",
            Self::Working => "working",
            Self::Document => "document",
            Self::Procedural => "procedural",
            Self::StateObject => "state_object",
        }
    }

    pub fn default_decay(&self) -> DecayClass {
        match self {
            Self::Episodic => DecayClass::Medium,
            Self::Semantic => DecayClass::Slow,
            Self::Working => DecayClass::Fast,
            Self::Document => DecayClass::Slow,
            Self::Procedural => DecayClass::Slow,
            Self::StateObject => DecayClass::None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, sqlx::Type)]
#[sqlx(type_name = "text", rename_all = "snake_case")]
#[serde(rename_all = "snake_case")]
pub enum DecayClass {
    None,
    Slow,
    Medium,
    Fast,
}

impl DecayClass {
    /// Half-life in days. Returned as f64 because score math is f64.
    pub fn half_life_days(&self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Slow => Some(90.0),
            Self::Medium => Some(14.0),
            Self::Fast => Some(2.0),
        }
    }
}

#[derive(Debug, Clone, FromRow)]
pub struct MemoryRow {
    pub id: Uuid,
    pub r#type: String,
    pub content: String,
    pub embedding: Option<Vector>,
    pub importance: f32,
    pub access_count: i32,
    pub decay_class: String,
    pub user_id: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub entities: Vec<String>,
    pub superseded_by: Option<Uuid>,
    pub supersedes: Option<Uuid>,
    pub source_path: Option<String>,
    pub chunk_index: Option<i32>,
    pub content_hash: Option<String>,
    pub state_kind: Option<String>,
    pub state_key: Option<String>,
    pub state_data: Option<serde_json::Value>,
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
}

/// Public-facing serialization of a memory row.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryView {
    pub id: Uuid,
    #[serde(rename = "type")]
    pub type_: String,
    pub content: String,
    pub importance: f32,
    pub access_count: i32,
    pub decay_class: String,
    pub user_id: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub entities: Vec<String>,
    pub superseded_by: Option<Uuid>,
    pub supersedes: Option<Uuid>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state_data: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
}

impl From<MemoryRow> for MemoryView {
    fn from(row: MemoryRow) -> Self {
        Self {
            id: row.id,
            type_: row.r#type,
            content: row.content,
            importance: row.importance,
            access_count: row.access_count,
            decay_class: row.decay_class,
            user_id: row.user_id,
            project_id: row.project_id,
            session_id: row.session_id,
            entities: row.entities,
            superseded_by: row.superseded_by,
            supersedes: row.supersedes,
            source_path: row.source_path,
            chunk_index: row.chunk_index,
            state_kind: row.state_kind,
            state_key: row.state_key,
            state_data: row.state_data,
            expires_at: row.expires_at,
            created_at: row.created_at,
            last_accessed_at: row.last_accessed_at,
        }
    }
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct CoreMemoryRow {
    pub id: Uuid,
    pub user_id: String,
    pub content: String,
    pub importance: f32,
    pub pinned_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
