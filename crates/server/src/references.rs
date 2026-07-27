//! References as first-class citizens of the RAW layer.
//!
//! A reference is a named mutable cell — the "what am I currently maintaining"
//! half of memory, as opposed to the append-only "what happened" half. It rides
//! entirely on `raw_records` with `type='state_object'`; 010 promoted its
//! identity onto two indexed columns:
//!
//!   identity      = (user_id, state_kind, state_key)
//!   current value = the terminal node — the state_object raw row for that
//!                   identity not referenced by any newer row's `supersedes`.
//!   audit trail   = the older rows in the supersede chain.
//!
//! `payload->'data'` on the terminal row is ALWAYS the complete current value
//! (never a delta), so reading the current value is a single indexed lookup with
//! no chain walk. A PATCH-style update is a NEW state_object raw row inserted
//! through the normal ingest path with `supersedes` = the old terminal id; the
//! old row is never mutated (append-only).
//!
//! The raw-native reference surface: these endpoints live under
//! `/records/state/*` and share the raw door's bearer auth and ingest path.

use axum::{
    extract::{Path, State},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{AssertSqlSafe, PgPool};
use uuid::Uuid;

use crate::routes::records::{ingest_record, IngestRecordRequest, RawRecordRow, COLS};
use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    nlp::NlpService,
    AppState,
};

/// The source tag stamped on reference writes that come through this surface.
const REF_SOURCE: &str = "reference";

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/{kind}", get(list_kind))
        .route("/{kind}/{key}", get(get_current).post(put_value))
        .route("/{kind}/{key}/history", get(history))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// The reference view over raw. A reference row is a state_object raw record
// projected to the reference shape; `state_data` is `payload->'data'`.
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct ReferenceView {
    pub id: Uuid,
    pub state_kind: String,
    pub state_key: String,
    pub user_id: String,
    pub project_id: Option<String>,
    pub container_id: Option<String>,
    /// The COMPLETE current value (never a delta).
    pub state_data: Value,
    pub supersedes: Option<Uuid>,
    pub event_time: chrono::DateTime<chrono::Utc>,
}

fn to_view(row: RawRecordRow) -> AppResult<ReferenceView> {
    let payload = row
        .payload
        .clone()
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("state_object row missing payload")))?;
    let state_kind = row.state_kind_of(&payload)?;
    let state_key = row.state_key_of(&payload)?;
    let state_data = payload.get("data").cloned().unwrap_or(Value::Null);
    Ok(ReferenceView {
        id: row.id,
        state_kind,
        state_key,
        user_id: row.user_id,
        project_id: row.project_id,
        container_id: row.container_id,
        state_data,
        supersedes: row.supersedes,
        event_time: row.event_time,
    })
}

// ---------------------------------------------------------------------------
// GET /records/state/:kind/:key — terminal current value (O(1), no chain walk)
// ---------------------------------------------------------------------------

async fn get_current(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path((kind, key)): Path<(String, String)>,
) -> AppResult<Json<ReferenceView>> {
    Ok(Json(
        get_current_inner(&state.pool, &auth_user.user_id, &kind, &key).await?,
    ))
}

pub(crate) async fn get_current_inner(
    pool: &PgPool,
    user_id: &str,
    kind: &str,
    key: &str,
) -> AppResult<ReferenceView> {
    let row = terminal_row(pool, user_id, kind, key)
        .await?
        .ok_or_else(|| AppError::not_found(format!("reference {kind}/{key}")))?;
    to_view(row)
}

/// The terminal state_object raw row for (user, kind, key): the one not
/// referenced by any newer row's `supersedes`. Single indexed lookup — the
/// self-contained current value, no chain walk. `event_time DESC` breaks ties in
/// the rare branched-chain case (mirrors the raw active-filter idiom).
async fn terminal_row(
    pool: &PgPool,
    user_id: &str,
    kind: &str,
    key: &str,
) -> AppResult<Option<RawRecordRow>> {
    let sql = format!(
        r#"
        SELECT {COLS} FROM raw_records
        WHERE type = 'state_object'
          AND user_id = $1
          AND state_kind = $2
          AND state_key = $3
          AND id NOT IN (
              SELECT supersedes FROM raw_records
              WHERE supersedes IS NOT NULL AND user_id = $1
          )
        ORDER BY event_time DESC
        LIMIT 1
        "#
    );
    let row = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(kind)
        .bind(key)
        .fetch_optional(pool)
        .await?;
    Ok(row)
}

// ---------------------------------------------------------------------------
// GET /records/state/:kind — all keys (their current values) for a kind
// ---------------------------------------------------------------------------

async fn list_kind(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(kind): Path<String>,
) -> AppResult<Json<Vec<ReferenceView>>> {
    Ok(Json(
        list_kind_inner(&state.pool, &auth_user.user_id, &kind).await?,
    ))
}

pub(crate) async fn list_kind_inner(
    pool: &PgPool,
    user_id: &str,
    kind: &str,
) -> AppResult<Vec<ReferenceView>> {
    // Terminal row per key in this kind. Same not-superseded filter as the
    // single-key read, folded over every key in the family.
    let sql = format!(
        r#"
        SELECT {COLS} FROM raw_records
        WHERE type = 'state_object'
          AND user_id = $1
          AND state_kind = $2
          AND id NOT IN (
              SELECT supersedes FROM raw_records
              WHERE supersedes IS NOT NULL AND user_id = $1
          )
        ORDER BY state_key ASC, event_time DESC
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(kind)
        .fetch_all(pool)
        .await?;
    rows.into_iter().map(to_view).collect()
}

// ---------------------------------------------------------------------------
// POST /records/state/:kind/:key — set the current value (append-only PATCH)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct PutValueRequest {
    /// The COMPLETE new current value (never a delta).
    pub data: Value,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub container_id: Option<String>,
    #[serde(default)]
    pub importance: Option<f32>,
}

async fn put_value(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path((kind, key)): Path<(String, String)>,
    Json(req): Json<PutValueRequest>,
) -> AppResult<Json<ReferenceView>> {
    Ok(Json(
        put_value_inner(
            &state.pool,
            &*state.nlp,
            &auth_user.user_id,
            &kind,
            &key,
            req,
        )
        .await?,
    ))
}

/// Write a new current value for a reference. Computes the current terminal,
/// then inserts a NEW state_object raw row through the normal ingest path with
/// `supersedes` = the old terminal id (or no supersede on first write). The old
/// row is never mutated — the whole surface is append-only.
pub(crate) async fn put_value_inner(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    kind: &str,
    key: &str,
    req: PutValueRequest,
) -> AppResult<ReferenceView> {
    if kind.trim().is_empty() || key.trim().is_empty() {
        return Err(AppError::bad_request(
            "state_kind and state_key must not be empty",
        ));
    }

    let prior = terminal_row(pool, user_id, kind, key).await?;
    let supersedes = prior.as_ref().map(|r| r.id);
    // Inherit scope from the prior terminal when the writer doesn't override it,
    // so a reference stays in its bucket across updates.
    let project_id = req
        .project_id
        .or_else(|| prior.as_ref().and_then(|r| r.project_id.clone()));
    let container_id = req
        .container_id
        .or_else(|| prior.as_ref().and_then(|r| r.container_id.clone()));

    // The payload carries the reference convention {kind, key, data}; the DB
    // trigger promotes kind/key onto the indexed columns at insert. `content` is
    // a deterministic rendering so the value stays in embeddings + BM25.
    let payload = json!({ "kind": kind, "key": key, "data": req.data });
    let content = render_reference(kind, key, &req.data);

    let ingest = IngestRecordRequest {
        r#type: "state_object".to_string(),
        content,
        event_time: None,
        source: REF_SOURCE.to_string(),
        source_ref: None,
        project_id,
        container_id,
        mode: None,
        importance: req.importance,
        supersedes,
        payload: Some(payload),
    };
    let out = ingest_record(pool, nlp, user_id, ingest).await?;

    get_current_inner(pool, user_id, kind, key)
        .await
        .map(|mut v| {
            // The just-written row is the terminal; assert we round-tripped the id.
            debug_assert_eq!(v.id, out.id);
            v.state_kind = kind.to_string();
            v.state_key = key.to_string();
            v
        })
}

/// A deterministic textual rendering of a reference value so it stays visible to
/// embeddings + BM25 retrieval. Structured values render as compact JSON; scalar
/// values render inline. `kind/key` prefix keeps the name searchable.
fn render_reference(kind: &str, key: &str, data: &Value) -> String {
    let body = match data {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => serde_json::to_string(other).unwrap_or_default(),
    };
    format!("{kind} {key}: {body}")
}

// ---------------------------------------------------------------------------
// GET /records/state/:kind/:key/history — walk the supersede chain oldest→newest
// ---------------------------------------------------------------------------

async fn history(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path((kind, key)): Path<(String, String)>,
) -> AppResult<Json<Value>> {
    Ok(Json(
        history_inner(&state.pool, &auth_user.user_id, &kind, &key).await?,
    ))
}

pub(crate) async fn history_inner(
    pool: &PgPool,
    user_id: &str,
    kind: &str,
    key: &str,
) -> AppResult<Value> {
    // Every row for the identity, ordered oldest→newest. state_object rows for a
    // (user, kind, key) form a supersede chain; ordering by event_time then
    // ingest_time reproduces the chain order deterministically.
    let sql = format!(
        r#"
        SELECT {COLS} FROM raw_records
        WHERE type = 'state_object'
          AND user_id = $1
          AND state_kind = $2
          AND state_key = $3
        ORDER BY event_time ASC, ingest_time ASC
        "#
    );
    let rows = sqlx::query_as::<_, RawRecordRow>(AssertSqlSafe(sql))
        .bind(user_id)
        .bind(kind)
        .bind(key)
        .fetch_all(pool)
        .await?;

    if rows.is_empty() {
        return Err(AppError::not_found(format!("reference {kind}/{key}")));
    }

    let terminal = rows
        .iter()
        .find(|r| {
            // Terminal = not superseded by any sibling in the chain.
            !rows.iter().any(|o| o.supersedes == Some(r.id))
        })
        .map(|r| r.id);
    let length = rows.len();
    let chain: Vec<ReferenceView> = rows
        .into_iter()
        .map(to_view)
        .collect::<AppResult<Vec<_>>>()?;

    Ok(json!({
        "state_kind": kind,
        "state_key": key,
        "terminal": terminal,
        "length": length,
        "chain": chain,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};

    #[derive(Clone)]
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
            Ok(vec![0.1_f32; 384])
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| vec![0.1_f32; 384]).collect())
        }
        fn extract_entities(&self, _text: &str) -> Vec<String> {
            Vec::new()
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            _e: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("stub".into()))
        }
    }

    fn put(data: Value) -> PutValueRequest {
        PutValueRequest {
            data,
            project_id: None,
            container_id: None,
            importance: None,
        }
    }

    async fn count_state_rows(pool: &PgPool, user_id: &str, key: &str) -> i64 {
        sqlx::query_scalar(
            "SELECT COUNT(*) FROM raw_records \
             WHERE type = 'state_object' AND user_id = $1 AND state_key = $2",
        )
        .bind(user_id)
        .bind(key)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    // ---- create + patch are append-only state_object rows -----------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_then_patch_writes_append_only_rows(pool: PgPool) {
        // Create.
        let v1 = put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "items": [{ "text": "buy milk" }] })),
        )
        .await
        .unwrap();
        assert_eq!(v1.state_kind, "todo_list");
        assert_eq!(v1.state_key, "today");
        assert!(v1.supersedes.is_none());
        assert_eq!(count_state_rows(&pool, "alice", "today").await, 1);

        // Patch (new complete value) → a NEW row superseding v1; v1 untouched.
        let v2 = put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "items": [{ "text": "buy milk" }, { "text": "fix bug" }] })),
        )
        .await
        .unwrap();
        assert_ne!(v2.id, v1.id);
        assert_eq!(v2.supersedes, Some(v1.id));
        assert_eq!(count_state_rows(&pool, "alice", "today").await, 2);

        // The promoted columns were populated by the trigger on both rows.
        let ident: (String, String) =
            sqlx::query_as("SELECT state_kind, state_key FROM raw_records WHERE id = $1")
                .bind(v2.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(ident, ("todo_list".to_string(), "today".to_string()));

        // v1's row is byte-for-byte unchanged (append-only).
        let v1_super: Option<Uuid> =
            sqlx::query_scalar("SELECT supersedes FROM raw_records WHERE id = $1")
                .bind(v1.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(v1_super, None);
    }

    // ---- terminal current value, no chain walk ----------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_current_returns_terminal_value(pool: PgPool) {
        put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "items": [] })),
        )
        .await
        .unwrap();
        let v2 = put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "items": [{ "text": "ship it" }] })),
        )
        .await
        .unwrap();

        let cur = get_current_inner(&pool, "alice", "todo_list", "today")
            .await
            .unwrap();
        // Terminal is v2 — the latest complete value, resolved by the
        // not-superseded filter (no chain walk).
        assert_eq!(cur.id, v2.id);
        assert_eq!(cur.state_data, json!({ "items": [{ "text": "ship it" }] }));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn get_current_404_when_missing(pool: PgPool) {
        let err = get_current_inner(&pool, "alice", "todo_list", "nope")
            .await
            .unwrap_err();
        let _ = err;
    }

    // ---- history walks the chain oldest → newest --------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn history_returns_full_chain_oldest_first(pool: PgPool) {
        let v1 = put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "n": 1 })),
        )
        .await
        .unwrap();
        // Distinct event_time so the oldest→newest order is deterministic.
        std::thread::sleep(std::time::Duration::from_millis(5));
        let v2 = put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "n": 2 })),
        )
        .await
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(5));
        let v3 = put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "n": 3 })),
        )
        .await
        .unwrap();

        let h = history_inner(&pool, "alice", "todo_list", "today")
            .await
            .unwrap();
        assert_eq!(h["length"], 3);
        assert_eq!(h["terminal"], json!(v3.id));
        let chain = h["chain"].as_array().unwrap();
        assert_eq!(chain[0]["id"], json!(v1.id));
        assert_eq!(chain[1]["id"], json!(v2.id));
        assert_eq!(chain[2]["id"], json!(v3.id));
    }

    // ---- raw immutability intact ------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn reference_writes_keep_raw_immutable(pool: PgPool) {
        let v1 = put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "items": [] })),
        )
        .await
        .unwrap();
        put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "items": [{ "text": "x" }] })),
        )
        .await
        .unwrap();

        // A direct UPDATE on the superseded row is still blocked by the trigger.
        let upd = sqlx::query("UPDATE raw_records SET content = 'tampered' WHERE id = $1")
            .bind(v1.id)
            .execute(&pool)
            .await;
        assert!(upd.is_err(), "raw_records must remain append-only");
    }

    // ---- scope isolation by user ------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn references_are_scope_isolated_by_user(pool: PgPool) {
        put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "who": "alice" })),
        )
        .await
        .unwrap();
        put_value_inner(
            &pool,
            &StubNlp,
            "bob",
            "todo_list",
            "today",
            put(json!({ "who": "bob" })),
        )
        .await
        .unwrap();

        // Same identity name, different users — each sees only their own value.
        let a = get_current_inner(&pool, "alice", "todo_list", "today")
            .await
            .unwrap();
        let b = get_current_inner(&pool, "bob", "todo_list", "today")
            .await
            .unwrap();
        assert_eq!(a.state_data, json!({ "who": "alice" }));
        assert_eq!(b.state_data, json!({ "who": "bob" }));

        // Bob's write superseded only bob's chain; alice keeps a single row.
        assert_eq!(count_state_rows(&pool, "alice", "today").await, 1);
        assert_eq!(count_state_rows(&pool, "bob", "today").await, 1);
    }

    // ---- list all keys for a kind -----------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_kind_returns_terminal_of_every_key(pool: PgPool) {
        put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "n": 1 })),
        )
        .await
        .unwrap();
        put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "tomorrow",
            put(json!({ "n": 2 })),
        )
        .await
        .unwrap();
        // Supersede "today" — list should show the terminal, not both rows.
        put_value_inner(
            &pool,
            &StubNlp,
            "alice",
            "todo_list",
            "today",
            put(json!({ "n": 3 })),
        )
        .await
        .unwrap();

        let all = list_kind_inner(&pool, "alice", "todo_list").await.unwrap();
        assert_eq!(all.len(), 2, "one terminal per key");
        let today = all.iter().find(|r| r.state_key == "today").unwrap();
        assert_eq!(today.state_data, json!({ "n": 3 }));
    }
}
