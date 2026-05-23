//! Axum handlers for the admin UI. Hand-rolled HTML — no templating crate.

use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, StatusCode},
    response::{Html, IntoResponse, Redirect, Response},
    Form, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::{
    auth::{self, AuthUser},
    models::{MemoryRow, MemoryView},
    AppState,
};

use super::{
    projection::{build_graph, GraphInput},
    style::STYLE_CSS,
    views,
};
use crate::consolidation;

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

pub async fn style_css() -> Response {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLE_CSS,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Login / logout
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct LoginQuery {
    #[serde(default)]
    pub reason: Option<String>,
}

pub async fn login_form(Query(q): Query<LoginQuery>) -> Response {
    let err = match q.reason.as_deref() {
        Some("unauth") => Some("Your session expired or no token was set."),
        Some("bad-token") => Some("That token is invalid or revoked."),
        _ => None,
    };
    Html(super::views::login_page(err)).into_response()
}

#[derive(Deserialize)]
pub struct LoginForm {
    pub token: String,
}

pub async fn login_submit(State(state): State<AppState>, Form(form): Form<LoginForm>) -> Response {
    let token = form.token.trim();
    if token.is_empty() {
        return Redirect::to("/admin/login?reason=bad-token").into_response();
    }
    // Validate by attempting a token lookup. Same algorithm as the auth
    // middleware uses for Bearer headers.
    let hash = auth::sha256_hex(token);
    let row: Result<Option<(Uuid, String)>, _> = sqlx::query_as(
        r#"
        SELECT id, user_id FROM tokens
        WHERE token_hash = $1 AND revoked_at IS NULL
        LIMIT 1
        "#,
    )
    .bind(&hash)
    .fetch_optional(&state.pool)
    .await;

    let valid = matches!(row, Ok(Some(_)));
    if !valid {
        return Redirect::to("/admin/login?reason=bad-token").into_response();
    }

    let cookie =
        format!("flashback_token={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000");
    (
        StatusCode::SEE_OTHER,
        [
            (header::LOCATION, "/admin".to_string()),
            (header::SET_COOKIE, cookie),
        ],
    )
        .into_response()
}

pub async fn logout() -> Response {
    let cookie = "flashback_token=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0";
    (
        StatusCode::SEE_OTHER,
        [
            (header::LOCATION, "/admin/login"),
            (header::SET_COOKIE, cookie),
        ],
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

pub async fn dashboard(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let memories_total: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM memories WHERE user_id = $1")
            .bind(&user.user_id)
            .fetch_one(&state.pool)
            .await?;
    let memories_terminal: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND superseded_by IS NULL",
    )
    .bind(&user.user_id)
    .fetch_one(&state.pool)
    .await?;
    let state_objects: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND type = 'state_object' AND superseded_by IS NULL",
    )
    .bind(&user.user_id)
    .fetch_one(&state.pool)
    .await?;
    let tokens_active: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM tokens WHERE user_id = $1 AND revoked_at IS NULL")
            .bind(&user.user_id)
            .fetch_one(&state.pool)
            .await?;

    let recent_rows =
        fetch_memories(&state.pool, &user.user_id, &MemoryQuery::default(), 10, 0).await?;

    let stats = views::DashboardStats {
        memories_total,
        memories_terminal,
        state_objects,
        tokens_active,
        provider: state.nlp.provider_name().to_string(),
        embedder_model: state.nlp.embedder_model_name().to_string(),
        embedder_dim: state.nlp.embedder_dimension(),
    };
    let recent: Vec<MemoryView> = recent_rows.into_iter().map(MemoryView::from).collect();
    Ok(Html(views::dashboard(&user.user_id, stats, &recent)))
}

// ---------------------------------------------------------------------------
// Memories list
// ---------------------------------------------------------------------------

#[derive(Default, Deserialize)]
pub struct MemoryQuery {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub include_superseded: Option<String>,
}

impl MemoryQuery {
    fn include_super(&self) -> bool {
        matches!(
            self.include_superseded.as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    }
    fn clean(&self) -> Self {
        Self {
            r#type: self.r#type.as_ref().and_then(empty_to_none),
            project_id: self.project_id.as_ref().and_then(empty_to_none),
            session_id: self.session_id.as_ref().and_then(empty_to_none),
            include_superseded: if self.include_super() {
                Some("1".to_string())
            } else {
                None
            },
        }
    }
}

fn empty_to_none(s: &String) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

pub async fn memories_list(
    State(state): State<AppState>,
    user: AuthUser,
    Query(q): Query<MemoryQuery>,
) -> Result<Html<String>, super::Error> {
    let q = q.clean();
    let rows = fetch_memories(&state.pool, &user.user_id, &q, 200, 0).await?;
    let total = count_memories(&state.pool, &user.user_id, &q).await?;
    let memories: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();

    let filter = views::MemoriesFilter {
        r#type: q.r#type.clone(),
        project_id: q.project_id.clone(),
        session_id: q.session_id.clone(),
        include_superseded: q.include_super(),
    };
    Ok(Html(views::memories_list(
        &user.user_id,
        &filter,
        &memories,
        total,
    )))
}

async fn fetch_memories(
    pool: &sqlx::PgPool,
    user_id: &str,
    q: &MemoryQuery,
    limit: i64,
    offset: i64,
) -> Result<Vec<MemoryRow>, sqlx::Error> {
    sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1
          AND ($2::TEXT IS NULL OR type = $2)
          AND ($3::TEXT IS NULL OR project_id = $3)
          AND ($4::TEXT IS NULL OR session_id = $4)
          AND ($5::BOOLEAN OR superseded_by IS NULL)
        ORDER BY created_at DESC
        LIMIT $6 OFFSET $7
        "#,
    )
    .bind(user_id)
    .bind(&q.r#type)
    .bind(&q.project_id)
    .bind(&q.session_id)
    .bind(q.include_super())
    .bind(limit)
    .bind(offset)
    .fetch_all(pool)
    .await
}

async fn count_memories(
    pool: &sqlx::PgPool,
    user_id: &str,
    q: &MemoryQuery,
) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar(
        r#"
        SELECT COUNT(*) FROM memories
        WHERE user_id = $1
          AND ($2::TEXT IS NULL OR type = $2)
          AND ($3::TEXT IS NULL OR project_id = $3)
          AND ($4::TEXT IS NULL OR session_id = $4)
          AND ($5::BOOLEAN OR superseded_by IS NULL)
        "#,
    )
    .bind(user_id)
    .bind(&q.r#type)
    .bind(&q.project_id)
    .bind(&q.session_id)
    .bind(q.include_super())
    .fetch_one(pool)
    .await
}

// ---------------------------------------------------------------------------
// Memory detail
// ---------------------------------------------------------------------------

pub async fn memory_detail(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Html<String>, super::Error> {
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
    .fetch_optional(&state.pool)
    .await?
    .ok_or(super::Error::NotFound)?;

    if row.user_id != user.user_id {
        return Err(super::Error::NotFound);
    }

    // Lineage walk.
    let chain_rows = sqlx::query_as::<_, MemoryRow>(
        r#"
        WITH RECURSIVE
        back AS (
            SELECT m.*, 0 AS depth FROM memories m WHERE m.id = $1
            UNION ALL
            SELECT prev.*, b.depth - 1 FROM memories prev JOIN back b ON prev.id = b.supersedes
        ),
        forward AS (
            SELECT m.*, 0 AS depth FROM memories m WHERE m.id = $1
            UNION ALL
            SELECT nxt.*, f.depth + 1 FROM memories nxt JOIN forward f ON nxt.supersedes = f.id
        )
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM (SELECT id, type, content, embedding, importance, access_count, decay_class,
                     user_id, project_id, session_id, entities, superseded_by, supersedes,
                     source_path, chunk_index, content_hash, state_kind, state_key, state_data,
                     expires_at, created_at, updated_at, last_accessed_at, depth FROM back
              UNION
              SELECT id, type, content, embedding, importance, access_count, decay_class,
                     user_id, project_id, session_id, entities, superseded_by, supersedes,
                     source_path, chunk_index, content_hash, state_kind, state_key, state_data,
                     expires_at, created_at, updated_at, last_accessed_at, depth FROM forward) AS u
        ORDER BY created_at ASC
        "#,
    )
    .bind(id)
    .fetch_all(&state.pool)
    .await?;

    let extraction: Option<Value> =
        sqlx::query_scalar("SELECT extraction FROM memories WHERE id = $1")
            .bind(id)
            .fetch_optional(&state.pool)
            .await?
            .flatten();

    let chain: Vec<MemoryView> = chain_rows.into_iter().map(MemoryView::from).collect();
    let view = MemoryView::from(row);
    Ok(Html(views::memory_detail(
        &user.user_id,
        &view,
        &chain,
        extraction.as_ref(),
    )))
}

pub async fn memory_delete(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Response, super::Error> {
    delete_memory_owned_by(&state.pool, &user.user_id, id).await?;
    Ok(Redirect::to("/admin/memories").into_response())
}

/// Hard-delete a memory after verifying it belongs to `user_id`. Nulls out
/// inbound supersede pointers so the DELETE doesn't violate FK constraints.
pub(crate) async fn delete_memory_owned_by(
    pool: &sqlx::PgPool,
    user_id: &str,
    id: Uuid,
) -> Result<(), super::Error> {
    let owner: Option<String> = sqlx::query_scalar("SELECT user_id FROM memories WHERE id = $1")
        .bind(id)
        .fetch_optional(pool)
        .await?;
    if owner.as_deref() != Some(user_id) {
        return Err(super::Error::NotFound);
    }
    sqlx::query("UPDATE memories SET superseded_by = NULL WHERE superseded_by = $1")
        .bind(id)
        .execute(pool)
        .await?;
    sqlx::query("UPDATE memories SET supersedes = NULL WHERE supersedes = $1")
        .bind(id)
        .execute(pool)
        .await?;
    sqlx::query("DELETE FROM memories WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// State objects
// ---------------------------------------------------------------------------

pub async fn state_list(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let rows = sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1 AND type = 'state_object' AND superseded_by IS NULL
        ORDER BY last_accessed_at DESC
        "#,
    )
    .bind(&user.user_id)
    .fetch_all(&state.pool)
    .await?;
    let views: Vec<MemoryView> = rows.into_iter().map(MemoryView::from).collect();
    Ok(Html(views::state_list(&user.user_id, &views)))
}

// ---------------------------------------------------------------------------
// Tokens
// ---------------------------------------------------------------------------

pub async fn tokens_list(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    #[derive(sqlx::FromRow)]
    struct Row {
        id: Uuid,
        token_prefix: String,
        user_id: String,
        name: Option<String>,
        created_at: DateTime<Utc>,
        last_used_at: Option<DateTime<Utc>>,
        revoked_at: Option<DateTime<Utc>>,
    }
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT id, token_prefix, user_id, name, created_at, last_used_at, revoked_at
        FROM tokens WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(&user.user_id)
    .fetch_all(&state.pool)
    .await?;
    let views: Vec<views::TokenView> = rows
        .into_iter()
        .map(|r| views::TokenView {
            id: r.id,
            prefix: r.token_prefix,
            user_id: r.user_id,
            name: r.name,
            created_at: r.created_at,
            last_used_at: r.last_used_at,
            revoked_at: r.revoked_at,
        })
        .collect();
    Ok(Html(views::tokens_list(&user.user_id, &views)))
}

pub async fn token_revoke(
    State(state): State<AppState>,
    user: AuthUser,
    Path(id): Path<Uuid>,
) -> Result<Response, super::Error> {
    revoke_token_owned_by(&state.pool, &user.user_id, id).await?;
    Ok(Redirect::to("/admin/tokens").into_response())
}

/// Revoke a token (if currently active) belonging to `user_id`. Returns Ok
/// even if the row is missing or already revoked — the admin UI just refreshes
/// the list on success.
pub(crate) async fn revoke_token_owned_by(
    pool: &sqlx::PgPool,
    user_id: &str,
    id: Uuid,
) -> Result<(), super::Error> {
    sqlx::query(
        "UPDATE tokens SET revoked_at = NOW() WHERE id = $1 AND user_id = $2 AND revoked_at IS NULL",
    )
    .bind(id)
    .bind(user_id)
    .execute(pool)
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Map
// ---------------------------------------------------------------------------

pub async fn map_view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let (nodes, edges) = compute_graph(&state.pool, &user.user_id).await?;
    Ok(Html(views::map_view(&user.user_id, nodes.len(), edges)))
}

pub async fn map_data(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Json<Value>, super::Error> {
    let (nodes, _) = compute_graph(&state.pool, &user.user_id).await?;
    // Re-run to get the structured edge list — separated so map_view can
    // count without doing the full work twice.
    let rows = fetch_for_graph(&state.pool, &user.user_id).await?;
    let inputs: Vec<GraphInput> = rows
        .iter()
        .map(|r| GraphInput {
            id: r.id,
            embedding: r.embedding.as_ref().map(|v| v.to_vec()).unwrap_or_default(),
            entities: r.entities.clone(),
            session_id: r.session_id.clone(),
            supersedes: r.supersedes,
        })
        .collect();
    let layout = build_graph(&inputs);

    let json_nodes: Vec<Value> = nodes
        .into_iter()
        .map(|n| {
            json!({
                "id": n.id,
                "type": n.type_,
                "label": n.label,
                "content": n.content,
                "importance": n.importance,
                "superseded": n.superseded,
                "x": n.x,
                "y": n.y,
                "x3": n.x3,
                "y3": n.y3,
                "z3": n.z3,
            })
        })
        .collect();

    let json_edges: Vec<Value> = layout
        .edges
        .into_iter()
        .map(|e| {
            json!({
                "source": e.source,
                "target": e.target,
                "kind": e.kind,
                "weight": e.weight,
            })
        })
        .collect();

    Ok(Json(json!({
        "nodes": json_nodes,
        "edges": json_edges,
    })))
}

// ---------------------------------------------------------------------------
// Consolidation
// ---------------------------------------------------------------------------

#[derive(Debug, sqlx::FromRow, Serialize)]
struct ConsolidationRunRow {
    id: Uuid,
    kind: String,
    trigger: String,
    started_at: DateTime<Utc>,
    finished_at: Option<DateTime<Utc>>,
    promoted_count: i32,
    expired_count: i32,
    distilled_count: i32,
    clusters_seen: i32,
    user_id: Option<String>,
    error: Option<String>,
}

pub async fn consolidate_view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let rows: Vec<ConsolidationRunRow> = sqlx::query_as(
        r#"
        SELECT id, kind, trigger, started_at, finished_at, promoted_count,
               expired_count, distilled_count, clusters_seen, user_id, error
        FROM consolidation_runs
        ORDER BY started_at DESC
        LIMIT 50
        "#,
    )
    .fetch_all(&state.pool)
    .await?;
    let provider_can_distill = state.nlp.provider_can_distill();
    Ok(Html(views::consolidate_view(
        &user.user_id,
        &rows,
        provider_can_distill,
        state.nlp.provider_name(),
    )))
}

pub async fn consolidate_trigger(
    State(state): State<AppState>,
    user: AuthUser,
    Path(kind): Path<String>,
) -> Result<Response, super::Error> {
    let stats = match kind.as_str() {
        "daily" => consolidation::run_daily(&state.pool, &user.user_id).await,
        "weekly" => consolidation::run_weekly(&state.pool, &state.nlp, &user.user_id).await,
        _ => {
            return Ok((StatusCode::BAD_REQUEST, "kind must be daily|weekly").into_response());
        }
    };
    tracing::info!(?stats, "manual consolidation run complete");
    Ok(Redirect::to("/admin/consolidate").into_response())
}

struct MapNode {
    id: Uuid,
    type_: String,
    label: String,
    content: String,
    importance: f32,
    superseded: bool,
    x: f32,
    y: f32,
    x3: f32,
    y3: f32,
    z3: f32,
}

async fn compute_graph(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<(Vec<MapNode>, usize), super::Error> {
    let rows = fetch_for_graph(pool, user_id).await?;
    let inputs: Vec<GraphInput> = rows
        .iter()
        .map(|r| GraphInput {
            id: r.id,
            embedding: r.embedding.as_ref().map(|v| v.to_vec()).unwrap_or_default(),
            entities: r.entities.clone(),
            session_id: r.session_id.clone(),
            supersedes: r.supersedes,
        })
        .collect();
    let layout = build_graph(&inputs);

    let nodes = rows
        .into_iter()
        .map(|r| {
            let (x, y) = layout.coords.get(&r.id).copied().unwrap_or((0.0, 0.0));
            let (x3, y3, z3) = layout
                .coords_3d
                .get(&r.id)
                .copied()
                .unwrap_or((0.0, 0.0, 0.0));
            let label = node_label(&r);
            let preview = preview(&r.content, 220);
            MapNode {
                id: r.id,
                type_: r.r#type,
                label,
                content: preview,
                importance: r.importance,
                superseded: r.superseded_by.is_some(),
                x,
                y,
                x3,
                y3,
                z3,
            }
        })
        .collect::<Vec<_>>();
    let edge_count = layout.edges.len();
    Ok((nodes, edge_count))
}

async fn fetch_for_graph(
    pool: &sqlx::PgPool,
    user_id: &str,
) -> Result<Vec<MemoryRow>, sqlx::Error> {
    // Cap at 500 to keep the O(N²) Jaccard scan + browser-side force sim happy.
    sqlx::query_as::<_, MemoryRow>(
        r#"
        SELECT id, type, content, embedding, importance, access_count, decay_class,
               user_id, project_id, session_id, entities, superseded_by, supersedes,
               source_path, chunk_index, content_hash, state_kind, state_key, state_data,
               expires_at, created_at, updated_at, last_accessed_at
        FROM memories
        WHERE user_id = $1
        ORDER BY last_accessed_at DESC
        LIMIT 500
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
}

fn node_label(r: &MemoryRow) -> String {
    // Prefer extracted topic if present in `extraction.topic`.
    // For state_object, prefer state_key.
    if let Some(key) = &r.state_key {
        return key.clone();
    }
    // We don't deserialize the full extraction here for cost reasons; just
    // pull the topic via a quick JSON string lookup.
    // (For now we rely on entities — Phase 2d will use the topic column.)
    if let Some(first_entity) = r.entities.iter().find(|e| e.contains(' ')) {
        return first_entity.clone();
    }
    if let Some(first) = r.entities.first() {
        return first.clone();
    }
    preview(&r.content, 28)
}

fn preview(s: &str, n: usize) -> String {
    let cleaned: String = s
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" · ");
    if cleaned.chars().count() <= n {
        cleaned
    } else {
        let mut out: String = cleaned.chars().take(n).collect();
        out.push('…');
        out
    }
}

// Suppress unused-headers warning on imports we keep for future use.
#[allow(dead_code)]
fn _hdr(_: HeaderMap) {}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    async fn insert_memory_for(pool: &PgPool, user_id: &str) -> Uuid {
        let id = Uuid::new_v4();
        let emb = pgvector::Vector::from(vec![0.0_f32; 384]);
        sqlx::query(
            "INSERT INTO memories (id, type, content, embedding, importance, decay_class, user_id, entities)
             VALUES ($1, 'episodic', 'admin test', $2, 0.5, 'medium', $3, '{}')",
        )
        .bind(id)
        .bind(emb)
        .bind(user_id)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    // ---- delete_memory_owned_by ------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_removes_row(pool: PgPool) {
        let id = insert_memory_for(&pool, "alice").await;
        delete_memory_owned_by(&pool, "alice", id).await.unwrap();
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories WHERE id = $1")
            .bind(id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_rejects_wrong_user(pool: PgPool) {
        let id = insert_memory_for(&pool, "alice").await;
        let err = delete_memory_owned_by(&pool, "bob", id).await.unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
        // Row still exists.
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories WHERE id = $1")
            .bind(id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_404_when_missing(pool: PgPool) {
        let err = delete_memory_owned_by(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap_err();
        assert!(matches!(err, super::super::Error::NotFound));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn delete_memory_owned_by_nulls_supersede_pointers(pool: PgPool) {
        let old = insert_memory_for(&pool, "alice").await;
        let new = insert_memory_for(&pool, "alice").await;
        // Wire supersede chain: new supersedes old, old.superseded_by = new.
        sqlx::query("UPDATE memories SET supersedes = $1 WHERE id = $2")
            .bind(old)
            .bind(new)
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new)
            .bind(old)
            .execute(&pool)
            .await
            .unwrap();

        // Deleting `old` should null the inbound pointer on `new` and not FK-violate.
        delete_memory_owned_by(&pool, "alice", old).await.unwrap();

        let supersedes_on_new: Option<Uuid> =
            sqlx::query_scalar("SELECT supersedes FROM memories WHERE id = $1")
                .bind(new)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(supersedes_on_new.is_none());
    }

    // ---- revoke_token_owned_by -------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoke_token_owned_by_marks_revoked(pool: PgPool) {
        let minted = crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        revoke_token_owned_by(&pool, "alice", minted.id)
            .await
            .unwrap();
        let revoked_at: Option<chrono::DateTime<chrono::Utc>> =
            sqlx::query_scalar("SELECT revoked_at FROM tokens WHERE id = $1")
                .bind(minted.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(revoked_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoke_token_owned_by_doesnt_affect_other_users_tokens(pool: PgPool) {
        let alice_token = crate::auth::mint_token(&pool, "alice", None).await.unwrap();
        // Bob tries to revoke alice's token — should silently no-op.
        revoke_token_owned_by(&pool, "bob", alice_token.id)
            .await
            .unwrap();
        let revoked_at: Option<chrono::DateTime<chrono::Utc>> =
            sqlx::query_scalar("SELECT revoked_at FROM tokens WHERE id = $1")
                .bind(alice_token.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(
            revoked_at.is_none(),
            "bob's call should not revoke alice's token"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoke_token_owned_by_unknown_id_is_silent_ok(pool: PgPool) {
        // No row → no error. This matches the handler's UX contract
        // (refresh the tokens page either way).
        revoke_token_owned_by(&pool, "alice", Uuid::new_v4())
            .await
            .unwrap();
    }

    // ---- pure helpers ----------------------------------------------------

    #[test]
    fn empty_to_none_collapses_whitespace() {
        assert_eq!(empty_to_none(&"".to_string()), None);
        assert_eq!(empty_to_none(&"   ".to_string()), None);
        assert_eq!(empty_to_none(&"abc".to_string()), Some("abc".to_string()));
        // Trim is applied.
        assert_eq!(
            empty_to_none(&"  trimmed  ".to_string()),
            Some("trimmed".to_string())
        );
    }

    #[test]
    fn memory_query_include_super_truthy_values() {
        for v in ["1", "true", "on"] {
            let q = MemoryQuery {
                include_superseded: Some(v.to_string()),
                ..MemoryQuery::default()
            };
            assert!(q.include_super(), "{v:?} should be truthy");
        }
    }

    #[test]
    fn memory_query_include_super_falsy_and_unset() {
        let unset = MemoryQuery::default();
        assert!(!unset.include_super());

        for v in ["", "0", "false", "anything-else"] {
            let q = MemoryQuery {
                include_superseded: Some(v.to_string()),
                ..MemoryQuery::default()
            };
            assert!(!q.include_super(), "{v:?} should be falsy");
        }
    }

    #[test]
    fn memory_query_clean_collapses_empties() {
        let raw = MemoryQuery {
            r#type: Some("  ".to_string()),
            project_id: Some("real-project".to_string()),
            session_id: Some("".to_string()),
            include_superseded: Some("0".to_string()),
        };
        let cleaned = raw.clean();
        assert_eq!(cleaned.r#type, None);
        assert_eq!(cleaned.project_id.as_deref(), Some("real-project"));
        assert_eq!(cleaned.session_id, None);
        // include_superseded normalized to None when falsy.
        assert_eq!(cleaned.include_superseded, None);
    }

    #[test]
    fn memory_query_clean_preserves_truthy_include_super() {
        let raw = MemoryQuery {
            include_superseded: Some("true".to_string()),
            ..MemoryQuery::default()
        };
        let cleaned = raw.clean();
        // Normalized to canonical "1".
        assert_eq!(cleaned.include_superseded.as_deref(), Some("1"));
    }
}
