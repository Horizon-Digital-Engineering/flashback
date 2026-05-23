use axum::{
    extract::{Path, State},
    routing::{get, post, put},
    Json, Router,
};
use chrono::{DateTime, Duration, Utc};
use pgvector::Vector;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    models::{MemoryRow, MemoryView},
    retrieval::{self, Mode, SearchParams},
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/ingest", post(ingest))
        .route("/search", post(search))
        .route("/:id", get(get_memory))
        .route("/:id/supersede", put(supersede))
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
    let user_id = auth_user.user_id;
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
        state.nlp.extract_full(&extraction_text).await?
    };
    let entities = extraction.entities.clone();
    let extraction_json = serde_json::to_value(&extraction)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("serialize extraction: {e}")))?;
    let embedding = state.nlp.embed_one(&content).await?;

    // Supersede detection: scoped to same user + session + project, last 24h.
    // If the configured AI provider produced a `topic`, we prefer semantic
    // topic-cosine over the older entity-Jaccard fingerprint — it catches
    // paraphrase ("the DB" / "Postgres") and noun-phrase variation that
    // Jaccard misses. Entity-Jaccard remains a fallback.
    let supersede_target = if let Some(session_id) = req.session_id.as_deref() {
        let new_topic = extraction.topic.as_deref();
        find_supersede_target(
            &state,
            &user_id,
            req.project_id.as_deref(),
            session_id,
            &entities,
            new_topic,
        )
        .await?
    } else {
        None
    };

    let mut tx = state.pool.begin().await?;

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
    .bind(&user_id)
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

    Ok(Json(IngestResponse {
        id: new_id,
        r#type: mem_type,
        entities,
        superseded: supersede_target,
        extraction: extraction_json,
        provider: state.nlp.provider_name().to_string(),
    }))
}

async fn find_supersede_target(
    state: &crate::AppState,
    user_id: &str,
    project_id: Option<&str>,
    session_id: &str,
    new_entities: &[String],
    new_topic: Option<&str>,
) -> AppResult<Option<Uuid>> {
    // Pull recent candidates with their entities + topic (extraction->>topic).
    let candidates: Vec<(Uuid, Vec<String>, Option<String>)> =
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
        .fetch_all(&state.pool)
        .await?;

    if candidates.is_empty() {
        return Ok(None);
    }

    // 1. Try topic-cosine if the new turn has a topic AND any candidate does.
    if let Some(nt) = new_topic.map(|s| s.trim()).filter(|s| !s.is_empty()) {
        let with_topics: Vec<(Uuid, &str)> = candidates
            .iter()
            .filter_map(|(id, _, t)| {
                t.as_deref()
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(|t| (*id, t))
            })
            .collect();
        if !with_topics.is_empty() {
            // Quick string-equality short-circuit (case-insensitive).
            let nt_low = nt.to_lowercase();
            for (id, t) in &with_topics {
                if t.to_lowercase() == nt_low {
                    return Ok(Some(*id));
                }
            }
            // Embed all topics in a single batch + cosine compare.
            let mut texts: Vec<String> = vec![nt.to_string()];
            texts.extend(with_topics.iter().map(|(_, t)| t.to_string()));
            let embeddings = state
                .nlp
                .embedder()
                .embed(texts)
                .await
                .map_err(|e| AppError::Internal(anyhow::anyhow!("embed topics: {e}")))?;

            if embeddings.len() == with_topics.len() + 1 {
                let new_vec = &embeddings[0];
                let mut best: Option<(Uuid, f32)> = None;
                for (i, (id, _)) in with_topics.iter().enumerate() {
                    let cand_vec = &embeddings[i + 1];
                    let sim = cosine(new_vec, cand_vec);
                    if sim >= 0.85 && best.map(|(_, s)| sim > s).unwrap_or(true) {
                        best = Some((*id, sim));
                    }
                }
                if let Some((id, _)) = best {
                    return Ok(Some(id));
                }
            }
        }
    }

    // 2. Fallback: entity-Jaccard (legacy Phase 2a behavior). Catches the
    // case where heuristic provider didn't populate a topic.
    if new_entities.is_empty() {
        return Ok(None);
    }
    let entity_only: Vec<(Uuid, Vec<String>)> =
        candidates.into_iter().map(|(id, ents, _)| (id, ents)).collect();
    Ok(retrieval::looks_like_supersede(new_entities, &entity_only, 0.3))
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
    let user_id = auth_user.user_id;
    let mode = req.mode.unwrap_or_default();
    let top_k = req.top_k.unwrap_or(10).clamp(1, 100);

    // Embed query if present; extract entities for the entity_overlap term.
    let (embedding, entities) = if let Some(q) = req.query.as_deref().filter(|s| !s.trim().is_empty()) {
        let entities = state.nlp.extract_entities(q);
        let emb = state.nlp.embed_one(q).await?;
        (Some(emb), entities)
    } else {
        (None, Vec::new())
    };

    let mem_types_owned: Option<Vec<String>> = req.memory_types.clone();
    let mem_types_borrowed: Option<Vec<&str>> =
        mem_types_owned.as_ref().map(|v| v.iter().map(String::as_str).collect());

    let params = SearchParams {
        user_id: &user_id,
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

    let scored = retrieval::search(&state.pool, params).await?;

    let ids: Vec<Uuid> = scored.iter().map(|s| s.view.id).collect();
    retrieval::touch_access(&state.pool, &ids).await?;

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

    Ok(Json(SearchResponse {
        results,
        mode: match mode {
            Mode::Answer => "answer".to_string(),
            Mode::Manager => "manager".to_string(),
        },
        total,
    }))
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
    let row = fetch_memory(&state.pool, id).await?;
    if row.user_id != auth_user.user_id {
        return Err(AppError::not_found(format!("memory {id}")));
    }
    Ok(Json(MemoryView::from(row)))
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
    let old = fetch_memory(&state.pool, old_id).await?;
    if old.user_id != auth_user.user_id {
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
        state.nlp.extract_full(&req.content).await?
    };
    let entities = if extraction.entities.is_empty() {
        old.entities.clone()
    } else {
        extraction.entities.clone()
    };
    let extraction_json = serde_json::to_value(&extraction)
        .map_err(|e| AppError::Internal(anyhow::anyhow!("serialize extraction: {e}")))?;
    let embedding = state.nlp.embed_one(&req.content).await?;

    let new_id = Uuid::new_v4();
    let mut tx = state.pool.begin().await?;

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

    Ok(Json(SupersedeResponse {
        new_id,
        superseded_id: old_id,
    }))
}

// ---------------------------------------------------------------------------
// Lineage (supersede chain walk)
// ---------------------------------------------------------------------------

pub async fn lineage(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<Value>> {
    // Authorize: the lineage root must belong to the caller.
    let root = fetch_memory(&state.pool, id).await?;
    if root.user_id != auth_user.user_id {
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
    .fetch_all(&state.pool)
    .await?;

    if rows.is_empty() {
        return Err(AppError::not_found(format!("memory {id}")));
    }

    let views: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();
    let terminal = views
        .iter()
        .find(|v| v.superseded_by.is_none())
        .map(|v| v.id);

    Ok(Json(json!({
        "root": id,
        "terminal": terminal,
        "length": views.len(),
        "chain": views,
    })))
}
