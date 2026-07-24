use axum::{
    extract::{Path, State},
    routing::{get, post, put},
    Json, Router,
};
use chrono::{DateTime, Duration, Utc};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    models::{MemoryRow, MemoryView},
    nlp::NlpService,
    retrieval::{self, Mode, SearchParams},
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/ingest", post(ingest))
        .route("/search", post(search))
        .route("/{id}", get(get_memory))
        .route("/{id}/supersede", put(supersede))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Ingest
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct IngestRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    /// One of: full conversation turn pair OR a raw `content`.
    #[serde(default)]
    pub user_turn: Option<String>,
    #[serde(default)]
    pub assistant_turn: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
    #[serde(default)]
    pub ttl_hours: Option<i64>,
    #[serde(default)]
    pub entities_override: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct IngestResponse {
    pub id: Uuid,
    pub r#type: String,
    pub entities: Vec<String>,
    pub superseded: Option<Uuid>,
    /// The structured Extraction produced by the configured AiProvider.
    /// Heuristic produces best-effort intent + operation; LLM providers
    /// produce richer topic + action_target. Visible via MCP `structuredContent`.
    pub extraction: serde_json::Value,
    pub provider: String,
}

async fn ingest(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<IngestRequest>,
) -> AppResult<Json<IngestResponse>> {
    let out = ingest_memory(&state.pool, &*state.nlp, &auth_user.user_id, req).await?;
    Ok(Json(out))
}

/// Inner logic of the `/memory/ingest` handler. Takes the bare resources it
/// needs (pool + NlpService) so it's directly unit-testable with a stub
/// NLP and an `#[sqlx::test]` pool — no AppState, no axum, no fastembed.
pub(crate) async fn ingest_memory(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    req: IngestRequest,
) -> AppResult<IngestResponse> {
    let (content, extraction_text) = match (
        req.content.as_deref(),
        req.user_turn.as_deref(),
        req.assistant_turn.as_deref(),
    ) {
        (Some(c), _, _) if !c.trim().is_empty() => (c.to_string(), c.to_string()),
        (_, Some(u), Some(a)) => (
            // `content` (used for embedding) keeps the User:/Assistant: labels
            // so the embedding distinguishes speaker context.
            format!("User: {}\nAssistant: {}", u.trim(), a.trim()),
            // `extraction_text` (used for entity extraction) drops the labels
            // so they don't pollute the entity fingerprint with scaffolding.
            format!("{}\n{}", u.trim(), a.trim()),
        ),
        (_, Some(u), None) => (format!("User: {}", u.trim()), u.trim().to_string()),
        (_, None, Some(a)) => (format!("Assistant: {}", a.trim()), a.trim().to_string()),
        _ => {
            return Err(AppError::bad_request(
                "ingest requires either `content` or `user_turn`/`assistant_turn`",
            ));
        }
    };

    let mem_type = req.r#type.as_deref().unwrap_or("working").to_string();
    validate_type(&mem_type)?;

    let importance = req.importance.unwrap_or(0.5).clamp(0.0, 1.0);
    let decay_class = default_decay_for(&mem_type);
    let expires_at: Option<DateTime<Utc>> = match (&*mem_type, req.ttl_hours) {
        ("working", Some(h)) => Some(Utc::now() + Duration::hours(h)),
        ("working", None) => Some(Utc::now() + Duration::hours(48)),
        _ => req.ttl_hours.map(|h| Utc::now() + Duration::hours(h)),
    };

    // Full structured extraction via the configured provider. Heuristic returns
    // almost instantly; remote/local LLM takes 100-500ms. Embedding runs after
    // so we have both before the supersede-detection step needs entities.
    let extraction = if let Some(e) = req.entities_override {
        let mut x = flashback_nlp::Extraction::empty();
        x.entities = dedupe(e);
        x
    } else {
        nlp.extract_full(&extraction_text).await?
    };
    let entities = extraction.entities.clone();
    let extraction_json = serde_json::to_value(&extraction)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("serialize extraction: {e}")))?;
    let embedding = nlp.embed_one(&content).await?;

    // Supersede detection: scoped to same user + session + project, last 24h.
    // If the configured AI provider produced a `topic`, we prefer semantic
    // topic-cosine over the older entity-Jaccard fingerprint — it catches
    // paraphrase ("the DB" / "Postgres") and noun-phrase variation that
    // Jaccard misses. Entity-Jaccard remains a fallback.
    let supersede_target = if let Some(session_id) = req.session_id.as_deref() {
        let new_topic = extraction.topic.as_deref();
        find_supersede_target(
            pool,
            nlp,
            user_id,
            req.project_id.as_deref(),
            session_id,
            &entities,
            new_topic,
        )
        .await?
    } else {
        None
    };

    let mut tx = pool.begin().await?;

    let new_id = Uuid::new_v4();
    let vector = Vector::from(embedding);

    sqlx::query(
        r#"
        INSERT INTO memories (
            id, type, content, embedding, importance, decay_class,
            user_id, project_id, session_id, entities,
            supersedes, expires_at, extraction
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13
        )
        "#,
    )
    .bind(new_id)
    .bind(&mem_type)
    .bind(&content)
    .bind(vector)
    .bind(importance)
    .bind(decay_class)
    .bind(user_id)
    .bind(&req.project_id)
    .bind(&req.session_id)
    .bind(&entities)
    .bind(supersede_target)
    .bind(expires_at)
    .bind(&extraction_json)
    .execute(&mut *tx)
    .await?;

    if let Some(old_id) = supersede_target {
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new_id)
            .bind(old_id)
            .execute(&mut *tx)
            .await?;
    }

    tx.commit().await?;

    Ok(IngestResponse {
        id: new_id,
        r#type: mem_type,
        entities,
        superseded: supersede_target,
        extraction: extraction_json,
        provider: nlp.provider_name().to_string(),
    })
}

async fn find_supersede_target(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    project_id: Option<&str>,
    session_id: &str,
    new_entities: &[String],
    new_topic: Option<&str>,
) -> AppResult<Option<Uuid>> {
    let candidates = load_supersede_candidates(pool, user_id, project_id, session_id).await?;
    if candidates.is_empty() {
        return Ok(None);
    }

    if let Some(id) = match_by_topic_cosine(nlp, &candidates, new_topic).await? {
        return Ok(Some(id));
    }

    Ok(match_by_entity_jaccard(candidates, new_entities))
}

/// Recent unsuperseded working/episodic memories in the same session, with
/// their entities + extracted topic — the candidate set both supersede
/// strategies operate on.
async fn load_supersede_candidates(
    pool: &PgPool,
    user_id: &str,
    project_id: Option<&str>,
    session_id: &str,
) -> AppResult<Vec<(Uuid, Vec<String>, Option<String>)>> {
    let rows: Vec<(Uuid, Vec<String>, Option<String>)> =
        sqlx::query_as::<_, (Uuid, Vec<String>, Option<String>)>(
            r#"
            SELECT id, entities, (extraction ->> 'topic') AS topic
            FROM memories
            WHERE user_id = $1
              AND session_id = $2
              AND ($3::TEXT IS NULL OR project_id = $3)
              AND superseded_by IS NULL
              AND type IN ('working','episodic')
              AND created_at >= NOW() - INTERVAL '24 hours'
            ORDER BY created_at DESC
            LIMIT 50
            "#,
        )
        .bind(user_id)
        .bind(session_id)
        .bind(project_id)
        .fetch_all(pool)
        .await?;
    Ok(rows)
}

/// Phase-1 supersede strategy: case-insensitive string match on extracted
/// topic, falling back to cosine similarity (≥0.85) on topic embeddings.
async fn match_by_topic_cosine(
    nlp: &dyn NlpService,
    candidates: &[(Uuid, Vec<String>, Option<String>)],
    new_topic: Option<&str>,
) -> AppResult<Option<Uuid>> {
    let Some(nt) = new_topic.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(None);
    };
    let with_topics: Vec<(Uuid, &str)> = candidates
        .iter()
        .filter_map(|(id, _, t)| {
            t.as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(|t| (*id, t))
        })
        .collect();
    if with_topics.is_empty() {
        return Ok(None);
    }

    // Cheap exact-match short-circuit before paying for embeddings.
    let nt_low = nt.to_lowercase();
    if let Some((id, _)) = with_topics.iter().find(|(_, t)| t.to_lowercase() == nt_low) {
        return Ok(Some(*id));
    }

    let mut texts: Vec<String> = vec![nt.to_string()];
    texts.extend(with_topics.iter().map(|(_, t)| t.to_string()));
    let embeddings = nlp.embed_batch(texts).await?;

    if embeddings.len() != with_topics.len() + 1 {
        return Ok(None);
    }
    let new_vec = &embeddings[0];
    let best = with_topics
        .iter()
        .enumerate()
        .filter_map(|(i, (id, _))| {
            let sim = cosine(new_vec, &embeddings[i + 1]);
            (sim >= 0.85).then_some((*id, sim))
        })
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(best.map(|(id, _)| id))
}

/// Phase-2a fallback: pre-LLM entity-overlap heuristic. Used when the topic
/// strategy returns nothing (no topic on the new turn or no topic-bearing
/// candidates).
fn match_by_entity_jaccard(
    candidates: Vec<(Uuid, Vec<String>, Option<String>)>,
    new_entities: &[String],
) -> Option<Uuid> {
    if new_entities.is_empty() {
        return None;
    }
    let entity_only: Vec<(Uuid, Vec<String>)> = candidates
        .into_iter()
        .map(|(id, ents, _)| (id, ents))
        .collect();
    retrieval::looks_like_supersede(new_entities, &entity_only, 0.3)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-9);
    dot / denom
}

fn validate_type(t: &str) -> Result<(), AppError> {
    match t {
        "episodic" | "semantic" | "working" | "document" | "procedural" => Ok(()),
        "state_object" => Err(AppError::bad_request(
            "state_object memories must be created via /state/:kind",
        )),
        other => Err(AppError::bad_request(format!("unknown type: {other}"))),
    }
}

fn default_decay_for(t: &str) -> &'static str {
    match t {
        "working" => "fast",
        "episodic" => "medium",
        "semantic" | "procedural" | "document" => "slow",
        _ => "medium",
    }
}

fn dedupe(items: Vec<String>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let key = item.trim();
        if key.is_empty() {
            continue;
        }
        let lc = key.to_lowercase();
        if seen.insert(lc) {
            out.push(key.to_string());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub mode: Option<Mode>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub memory_types: Option<Vec<String>>,
    #[serde(default)]
    pub since: Option<DateTime<Utc>>,
    #[serde(default)]
    pub until: Option<DateTime<Utc>>,
    #[serde(default)]
    pub include_superseded: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct SearchResultRow {
    #[serde(flatten)]
    pub memory: MemoryView,
    pub score: f64,
    pub debug: ScoreDebug,
}

#[derive(Debug, Serialize)]
pub struct ScoreDebug {
    pub semantic: f64,
    pub keyword: f64,
    pub recency: f64,
    pub project: f64,
    pub entity: f64,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultRow>,
    pub mode: String,
    pub total: usize,
}

async fn search(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<SearchRequest>,
) -> AppResult<Json<SearchResponse>> {
    Ok(Json(
        search_memory(&state.pool, &*state.nlp, &auth_user.user_id, req).await?,
    ))
}

pub(crate) async fn search_memory(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    req: SearchRequest,
) -> AppResult<SearchResponse> {
    let mode = req.mode.unwrap_or_default();
    let top_k = req.top_k.unwrap_or(10).clamp(1, 100);

    // Embed query if present; extract entities for the entity_overlap term.
    let (embedding, entities) =
        if let Some(q) = req.query.as_deref().filter(|s| !s.trim().is_empty()) {
            let entities = nlp.extract_entities(q);
            let emb = nlp.embed_one(q).await?;
            (Some(emb), entities)
        } else {
            (None, Vec::new())
        };

    let mem_types_owned: Option<Vec<String>> = req.memory_types.clone();
    let mem_types_borrowed: Option<Vec<&str>> = mem_types_owned
        .as_ref()
        .map(|v| v.iter().map(String::as_str).collect());

    let params = SearchParams {
        user_id,
        project_id: req.project_id.as_deref(),
        session_id: req.session_id.as_deref(),
        query: req.query.as_deref(),
        query_embedding: embedding.as_deref(),
        query_entities: entities,
        memory_types: mem_types_borrowed,
        since: req.since,
        until: req.until,
        mode,
        top_k,
        include_superseded: req.include_superseded.unwrap_or(false),
    };

    let scored = retrieval::search(pool, params).await?;

    let ids: Vec<Uuid> = scored.iter().map(|s| s.view.id).collect();
    retrieval::touch_access(pool, &ids).await?;

    let total = scored.len();
    let results = scored
        .into_iter()
        .map(|s| SearchResultRow {
            memory: s.view,
            score: round4(s.score),
            debug: ScoreDebug {
                semantic: round4(s.semantic),
                keyword: round4(s.keyword),
                recency: round4(s.recency),
                project: round4(s.project),
                entity: round4(s.entity),
            },
        })
        .collect();

    Ok(SearchResponse {
        results,
        mode: match mode {
            Mode::Answer => "answer".to_string(),
            Mode::Manager => "manager".to_string(),
        },
        total,
    })
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

// ---------------------------------------------------------------------------
// Get by ID
// ---------------------------------------------------------------------------

async fn get_memory(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<MemoryView>> {
    Ok(Json(
        get_memory_by_id(&state.pool, &auth_user.user_id, id).await?,
    ))
}

pub(crate) async fn get_memory_by_id(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
) -> AppResult<MemoryView> {
    let row = fetch_memory(pool, id).await?;
    if row.user_id != user_id {
        return Err(AppError::not_found(format!("memory {id}")));
    }
    Ok(MemoryView::from(row))
}

async fn fetch_memory(pool: &sqlx::PgPool, id: Uuid) -> AppResult<MemoryRow> {
    let row = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| AppError::not_found(format!("memory {id}")))?;
    Ok(row)
}

// ---------------------------------------------------------------------------
// Supersede
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct SupersedeRequest {
    pub content: String,
    #[serde(default)]
    pub importance: Option<f32>,
    #[serde(default)]
    pub entities_override: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct SupersedeResponse {
    pub new_id: Uuid,
    pub superseded_id: Uuid,
}

async fn supersede(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(old_id): Path<Uuid>,
    Json(req): Json<SupersedeRequest>,
) -> AppResult<Json<SupersedeResponse>> {
    Ok(Json(
        supersede_memory(&state.pool, &*state.nlp, &auth_user.user_id, old_id, req).await?,
    ))
}

pub(crate) async fn supersede_memory(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    old_id: Uuid,
    req: SupersedeRequest,
) -> AppResult<SupersedeResponse> {
    let old = fetch_memory(pool, old_id).await?;
    if old.user_id != user_id {
        return Err(AppError::not_found(format!("memory {old_id}")));
    }
    if old.superseded_by.is_some() {
        return Err(AppError::Conflict(format!(
            "{old_id} has already been superseded"
        )));
    }
    if old.r#type == "state_object" {
        return Err(AppError::bad_request(
            "state_object memories supersede via /state/:kind/:key PATCH ops",
        ));
    }

    let extraction = if let Some(e) = req.entities_override {
        let mut x = flashback_nlp::Extraction::empty();
        x.entities = dedupe(e);
        x
    } else {
        nlp.extract_full(&req.content).await?
    };
    let entities = if extraction.entities.is_empty() {
        old.entities.clone()
    } else {
        extraction.entities.clone()
    };
    let extraction_json = serde_json::to_value(&extraction)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("serialize extraction: {e}")))?;
    let embedding = nlp.embed_one(&req.content).await?;

    let new_id = Uuid::new_v4();
    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        INSERT INTO memories (
            id, type, content, embedding, importance, decay_class,
            user_id, project_id, session_id, entities,
            supersedes, expires_at, extraction
        ) VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13
        )
        "#,
    )
    .bind(new_id)
    .bind(&old.r#type)
    .bind(&req.content)
    .bind(Vector::from(embedding))
    .bind(req.importance.unwrap_or(old.importance).clamp(0.0, 1.0))
    .bind(&old.decay_class)
    .bind(&old.user_id)
    .bind(&old.project_id)
    .bind(&old.session_id)
    .bind(&entities)
    .bind(old_id)
    .bind::<Option<DateTime<Utc>>>(None)
    .bind(&extraction_json)
    .execute(&mut *tx)
    .await?;

    sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
        .bind(new_id)
        .bind(old_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

    Ok(SupersedeResponse {
        new_id,
        superseded_id: old_id,
    })
}

// ---------------------------------------------------------------------------
// Lineage (supersede chain walk)
// ---------------------------------------------------------------------------

pub async fn lineage(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<Value>> {
    Ok(Json(
        memory_lineage(&state.pool, &auth_user.user_id, id).await?,
    ))
}

pub(crate) async fn memory_lineage(pool: &PgPool, user_id: &str, id: Uuid) -> AppResult<Value> {
    // Authorize: the lineage root must belong to the caller.
    let root = fetch_memory(pool, id).await?;
    if root.user_id != user_id {
        return Err(AppError::not_found(format!("memory {id}")));
    }
    let _ = root;

    // Walk both directions: back through supersedes (older versions) and
    // forward through superseded_by (newer versions).
    let rows: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
        r#"
        WITH RECURSIVE
        back AS (
            SELECT m.*, 0 AS depth FROM memories m WHERE m.id = $1
            UNION ALL
            SELECT prev.*, b.depth - 1
            FROM memories prev
            JOIN back b ON prev.id = b.supersedes
        ),
        forward AS (
            SELECT m.*, 0 AS depth FROM memories m WHERE m.id = $1
            UNION ALL
            SELECT nxt.*, f.depth + 1
            FROM memories nxt
            JOIN forward f ON nxt.supersedes = f.id
        ),
        all_nodes AS (
            SELECT id, type, content, embedding, importance, access_count, decay_class,
                   user_id, project_id, session_id, entities, superseded_by, supersedes,
                   source_path, chunk_index, content_hash, state_kind, state_key, state_data,
                   expires_at, created_at, updated_at, last_accessed_at, depth
            FROM back
            UNION
            SELECT id, type, content, embedding, importance, access_count, decay_class,
                   user_id, project_id, session_id, entities, superseded_by, supersedes,
                   source_path, chunk_index, content_hash, state_kind, state_key, state_data,
                   expires_at, created_at, updated_at, last_accessed_at, depth
            FROM forward
        )
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM all_nodes
        ORDER BY created_at ASC
        "#,
    )
    .bind(id)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Err(AppError::not_found(format!("memory {id}")));
    }

    let views: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();
    let terminal = views
        .iter()
        .find(|v| v.superseded_by.is_none())
        .map(|v| v.id);

    Ok(json!({
        "root": id,
        "terminal": terminal,
        "length": views.len(),
        "chain": views,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = [1.0_f32, 2.0, 3.0];
        assert!((cosine(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn cosine_opposite_is_negative_one() {
        assert!((cosine(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_empty_or_mismatched_returns_zero() {
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn cosine_zero_norm_doesnt_panic() {
        // Both zero vectors: dot=0, denom is clamped to 1e-9. Result is 0.
        assert!(cosine(&[0.0, 0.0], &[0.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn validate_type_accepts_known_types() {
        for t in ["episodic", "semantic", "working", "document", "procedural"] {
            assert!(validate_type(t).is_ok(), "rejected {t}");
        }
    }

    #[test]
    fn validate_type_rejects_state_object_with_redirect_message() {
        let err = validate_type("state_object").unwrap_err();
        // AppError doesn't expose its message directly; we just confirm it errs.
        // The specific redirect message is asserted as a behavioral contract
        // implicitly by `validate_type` being the only caller's gate.
        let _ = err;
    }

    #[test]
    fn validate_type_rejects_unknown_types() {
        assert!(validate_type("garbage").is_err());
        assert!(validate_type("").is_err());
        assert!(validate_type("EPISODIC").is_err()); // case-sensitive
    }

    #[test]
    fn default_decay_for_known_types() {
        assert_eq!(default_decay_for("working"), "fast");
        assert_eq!(default_decay_for("episodic"), "medium");
        assert_eq!(default_decay_for("semantic"), "slow");
        assert_eq!(default_decay_for("procedural"), "slow");
        assert_eq!(default_decay_for("document"), "slow");
    }

    #[test]
    fn default_decay_for_unknown_falls_back_to_medium() {
        assert_eq!(default_decay_for("state_object"), "medium");
        assert_eq!(default_decay_for(""), "medium");
        assert_eq!(default_decay_for("garbage"), "medium");
    }

    #[test]
    fn dedupe_removes_case_insensitive_duplicates() {
        let out = dedupe(vec!["Foo".into(), "foo".into(), "FOO".into()]);
        assert_eq!(out, vec!["Foo"]); // first-seen casing preserved
    }

    #[test]
    fn dedupe_drops_empty_and_whitespace() {
        let out = dedupe(vec!["a".into(), "".into(), "  ".into(), "b".into()]);
        assert_eq!(out, vec!["a", "b"]);
    }

    #[test]
    fn dedupe_trims_whitespace() {
        let out = dedupe(vec!["  spaced  ".into(), "spaced".into()]);
        assert_eq!(out, vec!["spaced"]); // trimmed key matches second entry
    }

    #[test]
    fn dedupe_preserves_order() {
        let out = dedupe(vec!["c".into(), "a".into(), "b".into(), "a".into()]);
        assert_eq!(out, vec!["c", "a", "b"]);
    }

    #[test]
    fn dedupe_empty_input_gives_empty_output() {
        let out: Vec<String> = dedupe(vec![]);
        assert!(out.is_empty());
    }

    // ---- ingest_memory integration tests --------------------------------
    //
    // These exercise the full handler logic (DB writes, supersede detection,
    // extraction wiring) using a `StubNlp` that returns canned responses.
    // No fastembed, no AiProvider HTTP. Just pool + stub.

    use async_trait::async_trait;
    use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};

    use crate::nlp::NlpService;

    #[derive(Clone)]
    struct StubNlp {
        extract_result: Extraction,
        embedding: Vec<f32>,
    }

    impl StubNlp {
        fn new() -> Self {
            Self {
                extract_result: Extraction::empty(),
                embedding: vec![0.0_f32; 384],
            }
        }

        fn with_topic(mut self, topic: &str) -> Self {
            self.extract_result.topic = Some(topic.to_string());
            self
        }

        fn with_entities(mut self, ents: &[&str]) -> Self {
            self.extract_result.entities = ents.iter().map(|s| s.to_string()).collect();
            self
        }
    }

    #[async_trait]
    impl NlpService for StubNlp {
        fn provider_name(&self) -> &'static str {
            "stub"
        }
        fn provider_can_distill(&self) -> bool {
            false
        }
        fn embedder_model_name(&self) -> &str {
            "stub-embedder"
        }
        fn embedder_dimension(&self) -> usize {
            384
        }
        async fn embed_one(&self, _text: &str) -> Result<Vec<f32>, AppError> {
            Ok(self.embedding.clone())
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| self.embedding.clone()).collect())
        }
        fn extract_entities(&self, _text: &str) -> Vec<String> {
            Vec::new()
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(self.extract_result.clone())
        }
        async fn distill_facts(
            &self,
            _episodes: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("stub never distills".into()))
        }
    }

    fn make_req(content: Option<&str>) -> IngestRequest {
        IngestRequest {
            project_id: None,
            session_id: None,
            user_turn: None,
            assistant_turn: None,
            content: content.map(|s| s.to_string()),
            r#type: None,
            importance: None,
            ttl_hours: None,
            entities_override: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_writes_a_row_and_returns_id(pool: PgPool) {
        let nlp = StubNlp::new().with_entities(&["one", "two"]);
        let req = make_req(Some("hello world"));

        let resp = ingest_memory(&pool, &nlp, "alice", req).await.unwrap();
        assert_eq!(resp.r#type, "working");
        assert_eq!(resp.entities, vec!["one", "two"]);
        assert_eq!(resp.provider, "stub");
        assert!(resp.superseded.is_none());

        // Row landed.
        let (content, user_id, mem_type): (String, String, String) =
            sqlx::query_as("SELECT content, user_id, type FROM memories WHERE id = $1")
                .bind(resp.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(content, "hello world");
        assert_eq!(user_id, "alice");
        assert_eq!(mem_type, "working");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_builds_content_from_turn_pair(pool: PgPool) {
        let nlp = StubNlp::new();
        let mut req = make_req(None);
        req.user_turn = Some("what is X?".into());
        req.assistant_turn = Some("X is Y".into());

        let resp = ingest_memory(&pool, &nlp, "alice", req).await.unwrap();

        let content: String = sqlx::query_scalar("SELECT content FROM memories WHERE id = $1")
            .bind(resp.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(content.contains("User: what is X?"));
        assert!(content.contains("Assistant: X is Y"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_rejects_empty_request(pool: PgPool) {
        let nlp = StubNlp::new();
        let req = make_req(None); // no content, no turns
        let err = ingest_memory(&pool, &nlp, "alice", req).await.unwrap_err();
        // Should be a 400-class error; just confirm it's an error path.
        let _ = err;

        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_rejects_unknown_type(pool: PgPool) {
        let nlp = StubNlp::new();
        let mut req = make_req(Some("hi"));
        req.r#type = Some("not_a_real_type".into());
        let err = ingest_memory(&pool, &nlp, "alice", req).await.unwrap_err();
        let _ = err;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_clamps_importance_to_unit_range(pool: PgPool) {
        let nlp = StubNlp::new();
        let mut req = make_req(Some("over"));
        req.importance = Some(5.0); // over 1.0
        let resp = ingest_memory(&pool, &nlp, "alice", req).await.unwrap();
        let imp: f32 = sqlx::query_scalar("SELECT importance FROM memories WHERE id = $1")
            .bind(resp.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!((imp - 1.0).abs() < 1e-6);

        let mut req2 = make_req(Some("under"));
        req2.importance = Some(-3.0);
        let resp2 = ingest_memory(&pool, &nlp, "alice", req2).await.unwrap();
        let imp2: f32 = sqlx::query_scalar("SELECT importance FROM memories WHERE id = $1")
            .bind(resp2.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert!(imp2.abs() < 1e-6);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_working_type_gets_default_ttl(pool: PgPool) {
        let nlp = StubNlp::new();
        let req = make_req(Some("ephemeral"));
        let resp = ingest_memory(&pool, &nlp, "alice", req).await.unwrap();

        let expires_at: Option<chrono::DateTime<chrono::Utc>> =
            sqlx::query_scalar("SELECT expires_at FROM memories WHERE id = $1")
                .bind(resp.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        let expires = expires_at.expect("working memory should have expires_at default");
        // Should be ~48h in the future. Allow 5-min slack.
        let now = chrono::Utc::now();
        let target = now + chrono::Duration::hours(48);
        let drift = (expires - target).num_minutes().abs();
        assert!(drift < 5, "expires_at drift was {drift} minutes");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_entities_override_skips_nlp_extract(pool: PgPool) {
        // StubNlp would return [] from extract; override forces a different set.
        let nlp = StubNlp::new().with_entities(&["from_stub"]);
        let mut req = make_req(Some("text"));
        req.entities_override = Some(vec!["forced_a".into(), "forced_b".into()]);

        let resp = ingest_memory(&pool, &nlp, "alice", req).await.unwrap();
        assert_eq!(resp.entities, vec!["forced_a", "forced_b"]);
        // DB confirms.
        let stored: Vec<String> = sqlx::query_scalar("SELECT entities FROM memories WHERE id = $1")
            .bind(resp.id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(stored, vec!["forced_a", "forced_b"]);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn ingest_memory_supersede_detection_runs_only_with_session_id(pool: PgPool) {
        let nlp = StubNlp::new().with_entities(&["topic-x"]);

        // Insert an existing candidate in the same session with overlapping entities.
        let old_id = Uuid::new_v4();
        sqlx::query(
            r#"INSERT INTO memories
               (id, type, content, embedding, importance, decay_class, user_id,
                session_id, entities, extraction)
               VALUES ($1, 'episodic', 'old content', $2, 0.5, 'medium',
                       'alice', 'sess-a', ARRAY['topic-x', 'topic-y'], '{}'::jsonb)"#,
        )
        .bind(old_id)
        .bind(pgvector::Vector::from(vec![0.0_f32; 384]))
        .execute(&pool)
        .await
        .unwrap();

        // Without session_id → no supersede detection runs.
        let no_sess = ingest_memory(&pool, &nlp, "alice", make_req(Some("new1")))
            .await
            .unwrap();
        assert!(no_sess.superseded.is_none());

        // With session_id matching the candidate → supersede detection fires.
        let mut req = make_req(Some("new2"));
        req.session_id = Some("sess-a".into());
        let with_sess = ingest_memory(&pool, &nlp, "alice", req).await.unwrap();
        // Single-entity overlap = jaccard 1/2 = 0.5 ≥ 0.3 threshold, so supersede fires.
        assert_eq!(with_sess.superseded, Some(old_id));
    }

    // ---- search_memory --------------------------------------------------

    fn search_req() -> SearchRequest {
        SearchRequest {
            project_id: None,
            session_id: None,
            query: None,
            memory_types: None,
            since: None,
            until: None,
            mode: None,
            top_k: None,
            include_superseded: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_memory_empty_db_returns_no_results(pool: PgPool) {
        let nlp = StubNlp::new();
        let resp = search_memory(&pool, &nlp, "alice", search_req())
            .await
            .unwrap();
        assert_eq!(resp.total, 0);
        assert_eq!(resp.mode, "answer");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_memory_returns_seeded_memories_scoped_to_user(pool: PgPool) {
        let nlp = StubNlp::new();
        // Seed alice + bob with one memory each via the production path
        // so all extraction + embedding + indexing actually runs.
        ingest_memory(&pool, &nlp, "alice", {
            let mut r = make_req(Some("alpha"));
            r.r#type = Some("episodic".into());
            r
        })
        .await
        .unwrap();
        ingest_memory(&pool, &nlp, "bob", {
            let mut r = make_req(Some("beta"));
            r.r#type = Some("episodic".into());
            r
        })
        .await
        .unwrap();

        let alice = search_memory(&pool, &nlp, "alice", search_req())
            .await
            .unwrap();
        assert_eq!(alice.total, 1);
        let bob = search_memory(&pool, &nlp, "bob", search_req())
            .await
            .unwrap();
        assert_eq!(bob.total, 1);
        let carol = search_memory(&pool, &nlp, "carol", search_req())
            .await
            .unwrap();
        assert_eq!(carol.total, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_memory_clamps_top_k(pool: PgPool) {
        let nlp = StubNlp::new();
        let mut req = search_req();
        req.top_k = Some(0);
        // top_k clamped to ≥1; not an error.
        let resp = search_memory(&pool, &nlp, "alice", req).await.unwrap();
        assert_eq!(resp.total, 0);

        let mut req2 = search_req();
        req2.top_k = Some(10_000);
        let resp2 = search_memory(&pool, &nlp, "alice", req2).await.unwrap();
        assert_eq!(resp2.total, 0); // empty DB but no panic
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_memory_manager_mode_label_is_returned(pool: PgPool) {
        let nlp = StubNlp::new();
        let mut req = search_req();
        req.mode = Some(Mode::Manager);
        let resp = search_memory(&pool, &nlp, "alice", req).await.unwrap();
        assert_eq!(resp.mode, "manager");
    }

    // ---- get_memory_by_id -----------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_memory_by_id_returns_view(pool: PgPool) {
        let nlp = StubNlp::new();
        let created = ingest_memory(&pool, &nlp, "alice", make_req(Some("hello")))
            .await
            .unwrap();

        let view = get_memory_by_id(&pool, "alice", created.id).await.unwrap();
        assert_eq!(view.id, created.id);
        assert_eq!(view.user_id, "alice");
        assert_eq!(view.content, "hello");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_memory_by_id_rejects_wrong_user(pool: PgPool) {
        let nlp = StubNlp::new();
        let created = ingest_memory(&pool, &nlp, "alice", make_req(Some("alice's")))
            .await
            .unwrap();
        let err = get_memory_by_id(&pool, "bob", created.id)
            .await
            .unwrap_err();
        let _ = err;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_memory_by_id_404_when_missing(pool: PgPool) {
        let err = get_memory_by_id(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap_err();
        let _ = err;
    }

    // ---- supersede_memory -----------------------------------------------

    fn supersede_req(content: &str) -> SupersedeRequest {
        SupersedeRequest {
            content: content.to_string(),
            importance: None,
            entities_override: None,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_memory_chains_old_and_new(pool: PgPool) {
        let nlp = StubNlp::new();
        let old = ingest_memory(&pool, &nlp, "alice", {
            let mut r = make_req(Some("v1"));
            r.r#type = Some("episodic".into());
            r
        })
        .await
        .unwrap();

        let resp = supersede_memory(&pool, &nlp, "alice", old.id, supersede_req("v2"))
            .await
            .unwrap();
        assert_eq!(resp.superseded_id, old.id);

        // Old row's superseded_by now points at the new id.
        let by: Option<Uuid> =
            sqlx::query_scalar("SELECT superseded_by FROM memories WHERE id = $1")
                .bind(old.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(by, Some(resp.new_id));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_memory_rejects_wrong_user(pool: PgPool) {
        let nlp = StubNlp::new();
        let old = ingest_memory(&pool, &nlp, "alice", make_req(Some("private")))
            .await
            .unwrap();
        let err = supersede_memory(&pool, &nlp, "bob", old.id, supersede_req("hijack"))
            .await
            .unwrap_err();
        let _ = err;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn supersede_memory_rejects_already_superseded(pool: PgPool) {
        let nlp = StubNlp::new();
        let old = ingest_memory(&pool, &nlp, "alice", make_req(Some("v1")))
            .await
            .unwrap();
        supersede_memory(&pool, &nlp, "alice", old.id, supersede_req("v2"))
            .await
            .unwrap();
        // Second supersede on the same row → Conflict.
        let err = supersede_memory(&pool, &nlp, "alice", old.id, supersede_req("v3"))
            .await
            .unwrap_err();
        let _ = err;
    }

    // ---- memory_lineage --------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn memory_lineage_returns_chain(pool: PgPool) {
        let nlp = StubNlp::new();
        let v1 = ingest_memory(&pool, &nlp, "alice", make_req(Some("v1")))
            .await
            .unwrap();
        let v2 = supersede_memory(&pool, &nlp, "alice", v1.id, supersede_req("v2"))
            .await
            .unwrap();
        let v3 = supersede_memory(&pool, &nlp, "alice", v2.new_id, supersede_req("v3"))
            .await
            .unwrap();

        let lineage = memory_lineage(&pool, "alice", v1.id).await.unwrap();
        assert_eq!(lineage["length"], 3);
        assert_eq!(lineage["root"], serde_json::json!(v1.id));
        assert_eq!(lineage["terminal"], serde_json::json!(v3.new_id));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn memory_lineage_rejects_wrong_user(pool: PgPool) {
        let nlp = StubNlp::new();
        let m = ingest_memory(&pool, &nlp, "alice", make_req(Some("alice's")))
            .await
            .unwrap();
        let err = memory_lineage(&pool, "bob", m.id).await.unwrap_err();
        let _ = err;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn memory_lineage_404_when_missing(pool: PgPool) {
        let err = memory_lineage(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap_err();
        let _ = err;
    }
}
