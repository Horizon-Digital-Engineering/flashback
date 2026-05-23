//! POST /context/assemble — 5-layer prompt context.
//!
//! Layers (most stable → most ephemeral):
//!   1. procedural
//!   2. active project (core memory + state_objects + project semantic)
//!   3. top retrieved memories (answer or manager mode)
//!   4. document chunks
//!   5. recent conversation (working memory, session-scoped, protected floor)

use std::collections::HashMap;

use axum::{extract::State, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    error::AppResult,
    models::{MemoryRow, MemoryView},
    retrieval::{self, approx_tokens, Mode, SearchParams},
    routes::state::fetch_active_state_objects,
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/assemble", post(assemble))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
pub struct AssembleRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub token_budget: Option<usize>,
    #[serde(default)]
    pub recent_turns_floor: Option<usize>,
}

#[derive(Debug, Serialize, Default)]
pub struct AssembleResponse {
    pub layers: Layers,
    pub token_counts: HashMap<String, usize>,
    pub total_tokens: usize,
    pub mode: String,
}

#[derive(Debug, Serialize, Default)]
pub struct Layers {
    pub procedural: String,
    pub project_context: String,
    pub memories: String,
    pub document_chunks: String,
    pub recent_conversation: String,
}

const BUDGET_PROCEDURAL: usize = 300;
const BUDGET_PROJECT: usize = 600;
const BUDGET_MEMORIES: usize = 1200;
const BUDGET_DOCUMENTS: usize = 800;
const BUDGET_CONVERSATION: usize = 1500;
const DEFAULT_TURN_FLOOR: usize = 3;

async fn assemble(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<AssembleRequest>,
) -> AppResult<Json<AssembleResponse>> {
    let user_id = auth_user.user_id;
    let mode = if req.query.is_some() {
        Mode::Answer
    } else {
        Mode::Manager
    };

    let layer1 = render_procedural(&app.pool, &user_id, req.project_id.as_deref()).await?;
    let layer1 = truncate_to_budget(layer1, BUDGET_PROCEDURAL);

    let layer2 = render_project_context(
        &app.pool,
        &user_id,
        req.project_id.as_deref(),
        req.session_id.as_deref(),
    )
    .await?;
    let layer2 = truncate_to_budget(layer2, BUDGET_PROJECT);

    let (layer3, layer3_query_embedding) = render_top_memories(&app, &user_id, &req, mode).await?;
    let layer3 = truncate_to_budget(layer3, BUDGET_MEMORIES);

    let layer4 = render_documents(
        &app,
        &user_id,
        req.project_id.as_deref(),
        req.query.as_deref(),
        layer3_query_embedding.as_deref(),
    )
    .await?;
    let layer4 = truncate_to_budget(layer4, BUDGET_DOCUMENTS);

    let floor = req.recent_turns_floor.unwrap_or(DEFAULT_TURN_FLOOR);
    let layer5 = render_recent_conversation(
        &app.pool,
        &user_id,
        req.session_id.as_deref(),
        floor,
    )
    .await?;
    let layer5 = truncate_protecting_floor(layer5, BUDGET_CONVERSATION, floor);

    let mut counts: HashMap<String, usize> = HashMap::new();
    counts.insert("procedural".to_string(), approx_tokens(&layer1));
    counts.insert("project_context".to_string(), approx_tokens(&layer2));
    counts.insert("memories".to_string(), approx_tokens(&layer3));
    counts.insert("document_chunks".to_string(), approx_tokens(&layer4));
    counts.insert("recent_conversation".to_string(), approx_tokens(&layer5));
    let total: usize = counts.values().sum();

    Ok(Json(AssembleResponse {
        layers: Layers {
            procedural: layer1,
            project_context: layer2,
            memories: layer3,
            document_chunks: layer4,
            recent_conversation: layer5,
        },
        token_counts: counts,
        total_tokens: total,
        mode: match mode {
            Mode::Answer => "answer".to_string(),
            Mode::Manager => "manager".to_string(),
        },
    }))
}

async fn render_procedural(
    pool: &PgPool,
    user_id: &str,
    project_id: Option<&str>,
) -> AppResult<String> {
    let rows: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1
          AND type = 'procedural'
          AND superseded_by IS NULL
          AND ($2::TEXT IS NULL OR project_id = $2 OR project_id IS NULL)
        ORDER BY importance DESC, last_accessed_at DESC
        LIMIT 10
        "#,
    )
    .bind(user_id)
    .bind(project_id)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(String::new());
    }

    let mut out = String::from("# Procedural patterns\n");
    for r in rows {
        out.push_str(&format!("- {}\n", r.content));
    }
    Ok(out)
}

async fn render_project_context(
    pool: &PgPool,
    user_id: &str,
    project_id: Option<&str>,
    session_id: Option<&str>,
) -> AppResult<String> {
    let mut out = String::new();

    let core: Vec<(Uuid, String)> = sqlx::query_as(
        r#"SELECT id, content FROM core_memory
           WHERE user_id = $1 ORDER BY importance DESC, pinned_at ASC"#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    if !core.is_empty() {
        out.push_str("# Core memory\n");
        for (_, c) in core {
            out.push_str(&format!("- {c}\n"));
        }
    }

    let states = fetch_active_state_objects(pool, user_id, project_id, session_id, 8).await?;
    if !states.is_empty() {
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str("# Active state\n");
        for s in &states {
            out.push_str(&s.content);
            out.push_str("\n\n");
        }
    }

    if project_id.is_some() {
        let semantic: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
            r#"
            SELECT id, type, content, embedding, importance, access_count, decay_class,
                   user_id, project_id, session_id, entities, superseded_by, supersedes,
                   source_path, chunk_index, content_hash, state_kind, state_key, state_data,
                   expires_at, created_at, updated_at, last_accessed_at
            FROM memories
            WHERE user_id = $1
              AND project_id = $2
              AND type = 'semantic'
              AND superseded_by IS NULL
            ORDER BY importance DESC, last_accessed_at DESC
            LIMIT 8
            "#,
        )
        .bind(user_id)
        .bind(project_id)
        .fetch_all(pool)
        .await?;

        if !semantic.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("# Project facts\n");
            for r in semantic {
                out.push_str(&format!("- {}\n", r.content));
            }
        }
    }

    Ok(out.trim_end().to_string())
}

async fn render_top_memories(
    app: &AppState,
    user_id: &str,
    req: &AssembleRequest,
    mode: Mode,
) -> AppResult<(String, Option<Vec<f32>>)> {
    let (embedding, entities) =
        if let Some(q) = req.query.as_deref().filter(|s| !s.trim().is_empty()) {
            let entities = app.nlp.extract_entities(q);
            let emb = app.nlp.embed_one(q).await?;
            (Some(emb), entities)
        } else {
            (None, Vec::new())
        };

    let allowed = vec!["episodic", "semantic", "working"];
    let params = SearchParams {
        user_id,
        project_id: req.project_id.as_deref(),
        session_id: req.session_id.as_deref(),
        query: req.query.as_deref(),
        query_embedding: embedding.as_deref(),
        query_entities: entities,
        memory_types: Some(allowed),
        since: None,
        until: None,
        mode,
        top_k: 10,
        include_superseded: false,
    };

    let scored = retrieval::search(&app.pool, params).await?;
    let ids: Vec<Uuid> = scored.iter().map(|s| s.view.id).collect();
    retrieval::touch_access(&app.pool, &ids).await?;

    if scored.is_empty() {
        return Ok((String::new(), embedding));
    }

    let mut out = String::from("# Retrieved memories\n");
    for s in scored {
        out.push_str(&format!("- ({:.2}) {}\n", s.score, s.view.content));
    }
    Ok((out, embedding))
}

async fn render_documents(
    app: &AppState,
    user_id: &str,
    project_id: Option<&str>,
    query: Option<&str>,
    query_embedding: Option<&[f32]>,
) -> AppResult<String> {
    let params = SearchParams {
        user_id,
        project_id,
        session_id: None,
        query,
        query_embedding,
        query_entities: Vec::new(),
        memory_types: Some(vec!["document"]),
        since: None,
        until: None,
        mode: Mode::Answer,
        top_k: 6,
        include_superseded: false,
    };
    let scored = retrieval::search(&app.pool, params).await?;
    if scored.is_empty() {
        return Ok(String::new());
    }
    let mut out = String::from("# Document chunks\n");
    for s in scored {
        let label = s.view.source_path.clone().unwrap_or_else(|| "doc".to_string());
        out.push_str(&format!("## {label}\n{}\n\n", s.view.content));
    }
    Ok(out.trim_end().to_string())
}

async fn render_recent_conversation(
    pool: &PgPool,
    user_id: &str,
    session_id: Option<&str>,
    _floor: usize,
) -> AppResult<String> {
    let Some(session_id) = session_id else {
        return Ok(String::new());
    };
    let rows: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1
          AND session_id = $2
          AND type IN ('working','episodic')
          AND superseded_by IS NULL
        ORDER BY created_at DESC
        LIMIT 25
        "#,
    )
    .bind(user_id)
    .bind(session_id)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(String::new());
    }

    let mut views: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();
    views.reverse();

    let mut out = String::from("# Recent conversation\n");
    for v in views {
        out.push_str(&v.content);
        out.push_str("\n\n");
    }
    Ok(out.trim_end().to_string())
}

fn truncate_to_budget(s: String, budget: usize) -> String {
    if approx_tokens(&s) <= budget {
        return s;
    }
    let mut blocks: Vec<&str> = s.split("\n\n").collect();
    while !blocks.is_empty() && approx_tokens(&blocks.join("\n\n")) > budget {
        blocks.pop();
    }
    blocks.join("\n\n")
}

fn truncate_protecting_floor(s: String, budget: usize, floor: usize) -> String {
    let blocks: Vec<String> = s.split("\n\n").map(|s| s.to_string()).collect();
    if blocks.is_empty() {
        return s;
    }
    if approx_tokens(&s) <= budget {
        return s;
    }
    let n = blocks.len();
    let protected_start = n.saturating_sub(floor);
    let protected: Vec<String> = blocks[protected_start..].to_vec();

    let mut working = protected.join("\n\n");
    if approx_tokens(&working) >= budget {
        return working;
    }

    for i in (0..protected_start).rev() {
        let candidate = format!("{}\n\n{}", blocks[i], working);
        if approx_tokens(&candidate) > budget {
            break;
        }
        working = candidate;
    }
    working
}
