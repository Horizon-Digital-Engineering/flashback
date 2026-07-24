//! state_object routes — the reference half of memory.
//!
//! POST   /state/:kind              create
//! GET    /state/:kind/:key         current value (terminal node)
//! PATCH  /state/:kind/:key         apply op → new terminal node
//! GET    /state/:kind/:key/history supersede chain
//!
//! user_id is always derived from the auth token; never accepted from request
//! bodies or query strings.

use axum::{
    extract::{Path, State},
    routing::get,
    Json, Router,
};
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
    state::StateKind,
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/{kind}", axum::routing::post(create))
        .route("/{kind}/{key}", get(get_current).patch(patch).put(replace))
        .route("/{kind}/{key}/history", get(history))
        .with_state(state)
}

#[derive(Debug)]
struct OwnedScope {
    user_id: String,
    project_id: Option<String>,
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    pub state_key: String,
    #[serde(default)]
    pub initial: Option<Value>,
    #[serde(default)]
    pub importance: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct StateView {
    pub id: Uuid,
    pub state_kind: String,
    pub state_key: String,
    pub user_id: String,
    pub project_id: Option<String>,
    pub session_id: Option<String>,
    pub state_data: Value,
    pub rendered: String,
    pub supersedes: Option<Uuid>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

async fn create(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path(kind_str): Path<String>,
    Json(req): Json<CreateRequest>,
) -> AppResult<Json<StateView>> {
    Ok(Json(
        create_state(&app.pool, &*app.nlp, &auth_user.user_id, &kind_str, req).await?,
    ))
}

pub(crate) async fn create_state(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    kind_str: &str,
    req: CreateRequest,
) -> AppResult<StateView> {
    let kind = StateKind::parse(kind_str)?;

    if req.state_key.trim().is_empty() {
        return Err(AppError::bad_request("state_key cannot be empty"));
    }

    let existing: Option<(Uuid,)> = sqlx::query_as(
        r#"
        SELECT id FROM memories
        WHERE user_id = $1 AND state_kind = $2 AND state_key = $3
          AND superseded_by IS NULL
        LIMIT 1
        "#,
    )
    .bind(user_id)
    .bind(kind.as_str())
    .bind(&req.state_key)
    .fetch_optional(pool)
    .await?;

    if let Some((id,)) = existing {
        return Err(AppError::Conflict(format!(
            "state object {}/{} already exists ({id})",
            kind.as_str(),
            req.state_key
        )));
    }

    let initial_data = match req.initial {
        Some(v) => kind.validate_initial(&v)?,
        None => kind.empty(),
    };

    let scope = OwnedScope {
        user_id: user_id.to_string(),
        project_id: req.project_id,
        session_id: req.session_id,
    };

    let new_id = insert_state(
        pool,
        nlp,
        kind,
        &req.state_key,
        &scope,
        &initial_data,
        req.importance,
        None,
    )
    .await?;

    let row = fetch_state(pool, new_id).await?;
    to_state_view(kind, row)
}

async fn get_current(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
) -> AppResult<Json<StateView>> {
    Ok(Json(
        get_state_current(&app.pool, &auth_user.user_id, &kind_str, &key).await?,
    ))
}

pub(crate) async fn get_state_current(
    pool: &PgPool,
    user_id: &str,
    kind_str: &str,
    key: &str,
) -> AppResult<StateView> {
    let kind = StateKind::parse(kind_str)?;
    let row = fetch_terminal(pool, user_id, kind, key).await?;
    to_state_view(kind, row)
}

#[derive(Debug, Deserialize)]
pub struct PatchRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(flatten)]
    pub op: Value,
}

async fn patch(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
    Json(req): Json<PatchRequest>,
) -> AppResult<Json<StateView>> {
    Ok(Json(
        patch_state(
            &app.pool,
            &*app.nlp,
            &auth_user.user_id,
            &kind_str,
            &key,
            req,
        )
        .await?,
    ))
}

pub(crate) async fn patch_state(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    kind_str: &str,
    key: &str,
    req: PatchRequest,
) -> AppResult<StateView> {
    let kind = StateKind::parse(kind_str)?;
    let current = fetch_terminal(pool, user_id, kind, key).await?;
    let current_data = current
        .state_data
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state row missing state_data")))?;

    let new_data = kind.apply(&current_data, &req.op)?;

    let scope = OwnedScope {
        user_id: user_id.to_string(),
        project_id: req.project_id.or(current.project_id.clone()),
        session_id: req.session_id.or(current.session_id.clone()),
    };

    let new_id = insert_state(
        pool,
        nlp,
        kind,
        key,
        &scope,
        &new_data,
        Some(current.importance),
        Some(current.id),
    )
    .await?;

    sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
        .bind(new_id)
        .bind(current.id)
        .execute(pool)
        .await?;

    let row = fetch_state(pool, new_id).await?;
    to_state_view(kind, row)
}

#[derive(Debug, Deserialize)]
pub struct ReplaceRequest {
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    pub state_data: Value,
    #[serde(default)]
    pub importance: Option<f32>,
}

async fn replace(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
    Json(req): Json<ReplaceRequest>,
) -> AppResult<Json<StateView>> {
    Ok(Json(
        replace_state(
            &app.pool,
            &*app.nlp,
            &auth_user.user_id,
            &kind_str,
            &key,
            req,
        )
        .await?,
    ))
}

pub(crate) async fn replace_state(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    kind_str: &str,
    key: &str,
    req: ReplaceRequest,
) -> AppResult<StateView> {
    let kind = StateKind::parse(kind_str)?;
    let current = fetch_terminal(pool, user_id, kind, key).await?;
    let validated = kind.validate_initial(&req.state_data)?;

    let scope = OwnedScope {
        user_id: user_id.to_string(),
        project_id: req.project_id.or(current.project_id.clone()),
        session_id: req.session_id.or(current.session_id.clone()),
    };

    let new_id = insert_state(
        pool,
        nlp,
        kind,
        key,
        &scope,
        &validated,
        req.importance.or(Some(current.importance)),
        Some(current.id),
    )
    .await?;

    sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
        .bind(new_id)
        .bind(current.id)
        .execute(pool)
        .await?;

    let row = fetch_state(pool, new_id).await?;
    to_state_view(kind, row)
}

async fn history(
    State(app): State<AppState>,
    auth_user: AuthUser,
    Path((kind_str, key)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    Ok(Json(
        state_history(&app.pool, &auth_user.user_id, &kind_str, &key).await?,
    ))
}

pub(crate) async fn state_history(
    pool: &PgPool,
    user_id: &str,
    kind_str: &str,
    key: &str,
) -> AppResult<Value> {
    let kind = StateKind::parse(kind_str)?;
    let rows: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1 AND state_kind = $2 AND state_key = $3
        ORDER BY created_at ASC
        "#,
    )
    .bind(user_id)
    .bind(kind.as_str())
    .bind(key)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Err(AppError::not_found(format!(
            "state object {}/{key}",
            kind.as_str()
        )));
    }

    let terminal = rows
        .iter()
        .find(|r| r.superseded_by.is_none())
        .map(|r| r.id);
    let chain: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();

    Ok(json!({
        "state_kind": kind.as_str(),
        "state_key": key,
        "terminal": terminal,
        "length": chain.len(),
        "chain": chain,
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async fn fetch_state(pool: &sqlx::PgPool, id: Uuid) -> AppResult<MemoryRow> {
    sqlx::query_as::<_, MemoryRow>(
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
    .ok_or_else(|| AppError::not_found(format!("memory {id}")))
}

async fn fetch_terminal(
    pool: &sqlx::PgPool,
    user_id: &str,
    kind: StateKind,
    key: &str,
) -> AppResult<MemoryRow> {
    sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1 AND state_kind = $2 AND state_key = $3
          AND superseded_by IS NULL
        LIMIT 1
        "#,
    )
    .bind(user_id)
    .bind(kind.as_str())
    .bind(key)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| AppError::not_found(format!("state object {}/{key}", kind.as_str())))
}

async fn insert_state(
    pool: &PgPool,
    nlp: &dyn NlpService,
    kind: StateKind,
    key: &str,
    scope: &OwnedScope,
    data: &Value,
    importance: Option<f32>,
    supersedes: Option<Uuid>,
) -> AppResult<Uuid> {
    let rendered = kind.render(data, key);
    let embedding = nlp.embed_one(&rendered).await?;
    let vector = Vector::from(embedding);

    let new_id = Uuid::new_v4();
    sqlx::query(
        r#"
        INSERT INTO memories (
            id, type, content, embedding, importance, decay_class,
            user_id, project_id, session_id, entities,
            state_kind, state_key, state_data,
            supersedes
        ) VALUES (
            $1, 'state_object', $2, $3, $4, 'none',
            $5, $6, $7, '{}',
            $8, $9, $10,
            $11
        )
        "#,
    )
    .bind(new_id)
    .bind(&rendered)
    .bind(vector)
    .bind(importance.unwrap_or(0.8).clamp(0.0, 1.0))
    .bind(&scope.user_id)
    .bind(&scope.project_id)
    .bind(&scope.session_id)
    .bind(kind.as_str())
    .bind(key)
    .bind(data)
    .bind(supersedes)
    .execute(pool)
    .await?;
    Ok(new_id)
}

fn to_state_view(kind: StateKind, row: MemoryRow) -> AppResult<StateView> {
    let data = row
        .state_data
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state row missing state_data")))?;
    let key = row
        .state_key
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state row missing state_key")))?;
    let rendered = kind.render(&data, &key);

    Ok(StateView {
        id: row.id,
        state_kind: kind.as_str().to_string(),
        state_key: key,
        user_id: row.user_id,
        project_id: row.project_id,
        session_id: row.session_id,
        state_data: data,
        rendered,
        supersedes: row.supersedes,
        created_at: row.created_at,
    })
}

// Used by /context/assemble Layer 2.
pub async fn fetch_active_state_objects(
    pool: &sqlx::PgPool,
    user_id: &str,
    project_id: Option<&str>,
    session_id: Option<&str>,
    limit: usize,
) -> AppResult<Vec<MemoryView>> {
    let rows: Vec<MemoryRow> = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1
          AND type = 'state_object'
          AND superseded_by IS NULL
          AND ($2::TEXT IS NULL OR project_id = $2 OR project_id IS NULL)
          AND ($3::TEXT IS NULL OR session_id = $3 OR session_id IS NULL)
        ORDER BY (CASE WHEN session_id = $3 THEN 0 ELSE 1 END),
                 (CASE WHEN project_id = $2 THEN 0 ELSE 1 END),
                 importance DESC,
                 last_accessed_at DESC
        LIMIT $4
        "#,
    )
    .bind(user_id)
    .bind(project_id)
    .bind(session_id)
    .bind(limit as i64)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(MemoryView::from).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};

    // Minimal NlpService stub — returns zero embeddings + empty extractions.
    // Used because state ops embed the rendered text on every insert.
    struct StubNlp;

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
            Ok(vec![0.0_f32; 384])
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| vec![0.0_f32; 384]).collect())
        }
        fn extract_entities(&self, _text: &str) -> Vec<String> {
            Vec::new()
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            _episodes: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("stub".into()))
        }
    }

    fn create_req(key: &str) -> CreateRequest {
        CreateRequest {
            state_key: key.to_string(),
            project_id: None,
            session_id: None,
            initial: None,
            importance: None,
        }
    }

    fn patch_op(op: serde_json::Value) -> PatchRequest {
        PatchRequest {
            project_id: None,
            session_id: None,
            op,
        }
    }

    // ---- create_state ----------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_state_inserts_new_state_object(pool: PgPool) {
        let view = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        assert_eq!(view.state_kind, "todo_list");
        assert_eq!(view.state_key, "today");
        assert_eq!(view.user_id, "alice");
        assert!(view.supersedes.is_none());
        // Default-empty body.
        assert_eq!(view.state_data, json!({ "items": [] }));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_state_rejects_unknown_kind(pool: PgPool) {
        let err = create_state(&pool, &StubNlp, "alice", "garbage", create_req("k"))
            .await
            .unwrap_err();
        let _ = err;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_state_rejects_empty_state_key(pool: PgPool) {
        let err = create_state(&pool, &StubNlp, "alice", "todo_list", create_req(""))
            .await
            .unwrap_err();
        let _ = err;
        let err2 = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("  "))
            .await
            .unwrap_err();
        let _ = err2;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_state_rejects_duplicate_active_key(pool: PgPool) {
        create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        let err = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap_err();
        let _ = err;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_state_scopes_uniqueness_to_user(pool: PgPool) {
        // Alice and Bob can both have a todo_list named "today" — they're
        // separate state objects under different user scopes.
        create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        create_state(&pool, &StubNlp, "bob", "todo_list", create_req("today"))
            .await
            .unwrap();
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_state_with_initial_data_validates(pool: PgPool) {
        let mut req = create_req("today");
        req.initial = Some(json!({ "items": [{ "text": "first" }] }));
        let view = create_state(&pool, &StubNlp, "alice", "todo_list", req)
            .await
            .unwrap();
        let items = view.state_data["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["text"], "first");
        // assign_missing_ids should have populated an id.
        assert!(items[0]["id"].as_str().unwrap().len() > 0);
    }

    // ---- get_state_current ----------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_state_current_returns_404_when_missing(pool: PgPool) {
        let err = get_state_current(&pool, "alice", "todo_list", "nope")
            .await
            .unwrap_err();
        let _ = err;
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_state_current_returns_terminal_node(pool: PgPool) {
        let created = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        let fetched = get_state_current(&pool, "alice", "todo_list", "today")
            .await
            .unwrap();
        assert_eq!(fetched.id, created.id);
        assert_eq!(fetched.state_key, "today");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_state_current_scopes_to_user(pool: PgPool) {
        create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        // Bob can't read alice's todo_list.
        assert!(get_state_current(&pool, "bob", "todo_list", "today")
            .await
            .is_err());
    }

    // ---- patch_state -----------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn patch_state_appends_via_add_op(pool: PgPool) {
        let created = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();

        let after = patch_state(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            patch_op(json!({ "op": "add", "text": "buy milk" })),
        )
        .await
        .unwrap();

        // Patch creates a new row that supersedes the previous terminal.
        assert_ne!(after.id, created.id);
        assert_eq!(after.supersedes, Some(created.id));
        let items = after.state_data["items"].as_array().unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["text"], "buy milk");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn patch_state_marks_old_row_superseded(pool: PgPool) {
        let created = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        let after = patch_state(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            patch_op(json!({ "op": "add", "text": "x" })),
        )
        .await
        .unwrap();

        // Old row's superseded_by should point at the new id.
        let row_superseded_by: Option<Uuid> =
            sqlx::query_scalar("SELECT superseded_by FROM memories WHERE id = $1")
                .bind(created.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(row_superseded_by, Some(after.id));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn patch_state_rejects_invalid_op(pool: PgPool) {
        create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        let err = patch_state(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            patch_op(json!({ "op": "garbage" })),
        )
        .await
        .unwrap_err();
        let _ = err;
    }

    // ---- replace_state ---------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn replace_state_swaps_full_data_and_supersedes(pool: PgPool) {
        let created = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();

        let replaced = replace_state(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            ReplaceRequest {
                project_id: None,
                session_id: None,
                state_data: json!({ "items": [{ "text": "new1" }, { "text": "new2" }] }),
                importance: None,
            },
        )
        .await
        .unwrap();

        assert_ne!(replaced.id, created.id);
        assert_eq!(replaced.supersedes, Some(created.id));
        assert_eq!(replaced.state_data["items"].as_array().unwrap().len(), 2);
    }

    // ---- state_history ---------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn state_history_returns_supersede_chain(pool: PgPool) {
        let v1 = create_state(&pool, &StubNlp, "alice", "todo_list", create_req("today"))
            .await
            .unwrap();
        patch_state(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            patch_op(json!({ "op": "add", "text": "x" })),
        )
        .await
        .unwrap();
        let v3 = patch_state(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            patch_op(json!({ "op": "add", "text": "y" })),
        )
        .await
        .unwrap();

        let history = state_history(&pool, "alice", "todo_list", "today")
            .await
            .unwrap();
        assert_eq!(history["state_kind"], "todo_list");
        assert_eq!(history["state_key"], "today");
        assert_eq!(history["length"], 3);
        assert_eq!(history["terminal"], serde_json::json!(v3.id));
        // Chain ordered by created_at ASC, so v1 first.
        let chain = history["chain"].as_array().unwrap();
        assert_eq!(chain[0]["id"], serde_json::json!(v1.id));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn state_history_404_when_no_rows(pool: PgPool) {
        let err = state_history(&pool, "alice", "todo_list", "missing")
            .await
            .unwrap_err();
        let _ = err;
    }
}
