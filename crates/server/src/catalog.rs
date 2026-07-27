//! The catalog — the store-registry over the lake.
//!
//! A catalog is the lake's answer to "is my data organized, and can I see it?".
//! Every store the lake knows about is a row in `catalog_stores`:
//!
//!   * The two built-in layers, `raw` and `curated`, AUTO-REGISTER on first read
//!     with a live schema + record-count summary. They are computed, not
//!     user-managed: their `access` is `{interface:'internal'}` and their counts
//!     come from the actual tables.
//!   * `operational`/`external` stores are user-managed. They declare a schema +
//!     an access descriptor and publish slices into the lake via
//!     `catalog_published_facts`; a sync pulls those slices into `raw_records`
//!     through the normal idempotent import path.
//!
//! Endpoints (nested under /catalog):
//!   GET    /catalog                    stores grouped by kind, each with a
//!                                       schema + current record count + lineage
//!   POST   /catalog/stores             register an operational/external store
//!   GET    /catalog/stores/:id         one store
//!   PUT    /catalog/stores/:id         update a store
//!   DELETE /catalog/stores/:id         remove a store
//!   POST   /catalog/stores/:id/sync    pull the store's published slices into raw

use axum::{
    extract::{Path, State},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::routes::records::{import_records_inner, ImportRecord, ImportRequest};
use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    nlp::NlpService,
    AppState,
};

/// The reserved built-in store names. They map to the two lake layers and can
/// never be registered/updated/deleted by a user — they are computed on read.
const RAW_STORE: &str = "raw_records";
const CURATED_STORE: &str = "curated_nodes";

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/", get(list_catalog))
        .route("/stores", post(create_store))
        .route(
            "/stores/{id}",
            get(get_store).put(update_store).delete(delete_store),
        )
        .route("/stores/{id}/sync", post(sync_store))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Row + view shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct StoreRow {
    pub id: Uuid,
    pub user_id: String,
    pub name: String,
    pub kind: String,
    pub schema: Option<Value>,
    pub access: Option<Value>,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_synced_at: Option<DateTime<Utc>>,
}

/// A store enriched with its current record count + a lineage note. This is the
/// shape the store map renders — a registry row plus the live "how full is it".
#[derive(Debug, Serialize)]
pub struct StoreView {
    #[serde(flatten)]
    pub store: StoreRow,
    /// Live record count in the store (from the real table for built-ins, or the
    /// staged published-fact count for a registered store).
    pub record_count: i64,
    /// A short human note on where this store's data comes from.
    pub lineage: String,
}

/// The catalog response — stores grouped by kind.
#[derive(Debug, Serialize)]
pub struct CatalogView {
    pub raw: Vec<StoreView>,
    pub curated: Vec<StoreView>,
    pub operational: Vec<StoreView>,
    pub external: Vec<StoreView>,
}

// ---------------------------------------------------------------------------
// GET /catalog — the store map
// ---------------------------------------------------------------------------

async fn list_catalog(
    State(state): State<AppState>,
    auth_user: AuthUser,
) -> AppResult<Json<CatalogView>> {
    Ok(Json(
        list_catalog_inner(&state.pool, &auth_user.user_id).await?,
    ))
}

pub(crate) async fn list_catalog_inner(pool: &PgPool, user_id: &str) -> AppResult<CatalogView> {
    ensure_builtin_stores(pool, user_id).await?;

    let stores: Vec<StoreRow> = sqlx::query_as::<_, StoreRow>(
        r#"
        SELECT id, user_id, name, kind, schema, access, description,
               created_at, updated_at, last_synced_at
        FROM catalog_stores
        WHERE ($1 = '*' OR user_id = $1)
        ORDER BY kind, name
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    let mut view = CatalogView {
        raw: Vec::new(),
        curated: Vec::new(),
        operational: Vec::new(),
        external: Vec::new(),
    };
    for store in stores {
        let record_count = store_record_count(pool, user_id, &store).await?;
        let lineage = lineage_note(&store);
        let enriched = StoreView {
            record_count,
            lineage,
            store,
        };
        match enriched.store.kind.as_str() {
            "raw" => view.raw.push(enriched),
            "curated" => view.curated.push(enriched),
            "operational" => view.operational.push(enriched),
            "external" => view.external.push(enriched),
            _ => {}
        }
    }
    Ok(view)
}

/// Idempotently register the two built-in layers as `raw`/`curated` stores and
/// refresh their live schema. Runs on every catalog read so a store map is
/// always current even for a brand-new user. Never touches user-registered rows.
async fn ensure_builtin_stores(pool: &PgPool, user_id: &str) -> AppResult<()> {
    upsert_builtin(
        pool,
        user_id,
        RAW_STORE,
        "raw",
        raw_schema(),
        "Immutable append-only raw layer — every typed record every consumer writes.",
    )
    .await?;
    upsert_builtin(
        pool,
        user_id,
        CURATED_STORE,
        "curated",
        curated_schema(),
        "Derived summary/fact layer, rebuildable from raw.",
    )
    .await?;
    Ok(())
}

async fn upsert_builtin(
    pool: &PgPool,
    user_id: &str,
    name: &str,
    kind: &str,
    schema: Value,
    description: &str,
) -> AppResult<()> {
    let access = json!({ "interface": "internal" });
    sqlx::query(
        r#"
        INSERT INTO catalog_stores (user_id, name, kind, schema, access, description)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (user_id, name) DO UPDATE
            SET schema = EXCLUDED.schema,
                access = EXCLUDED.access,
                description = EXCLUDED.description,
                updated_at = NOW()
        "#,
    )
    .bind(user_id)
    .bind(name)
    .bind(kind)
    .bind(&schema)
    .bind(&access)
    .bind(description)
    .execute(pool)
    .await?;
    Ok(())
}

/// The live schema of the raw layer — the promoted, queryable columns.
fn raw_schema() -> Value {
    json!({
        "table": "raw_records",
        "columns": [
            {"name": "id", "type": "uuid"},
            {"name": "type", "type": "text"},
            {"name": "content", "type": "text"},
            {"name": "event_time", "type": "timestamptz"},
            {"name": "ingest_time", "type": "timestamptz"},
            {"name": "source", "type": "text"},
            {"name": "source_ref", "type": "text"},
            {"name": "importance", "type": "real"},
            {"name": "supersedes", "type": "uuid"},
            {"name": "payload", "type": "jsonb"}
        ]
    })
}

/// The live schema of the curated layer.
fn curated_schema() -> Value {
    json!({
        "table": "curated_nodes",
        "columns": [
            {"name": "id", "type": "uuid"},
            {"name": "kind", "type": "text"},
            {"name": "content", "type": "text"},
            {"name": "level", "type": "int"},
            {"name": "importance", "type": "real"},
            {"name": "event_time", "type": "timestamptz"},
            {"name": "created_at", "type": "timestamptz"}
        ]
    })
}

/// The current record count for a store: the real table for the two built-ins,
/// the staged published-fact count for a user-registered store.
async fn store_record_count(pool: &PgPool, user_id: &str, store: &StoreRow) -> AppResult<i64> {
    let count = match (store.kind.as_str(), store.name.as_str()) {
        ("raw", RAW_STORE) => {
            sqlx::query_scalar("SELECT COUNT(*) FROM raw_records WHERE ($1 = '*' OR user_id = $1)")
                .bind(user_id)
                .fetch_one(pool)
                .await?
        }
        ("curated", CURATED_STORE) => {
            sqlx::query_scalar(
                "SELECT COUNT(*) FROM curated_nodes WHERE ($1 = '*' OR user_id = $1)",
            )
            .bind(user_id)
            .fetch_one(pool)
            .await?
        }
        _ => {
            sqlx::query_scalar("SELECT COUNT(*) FROM catalog_published_facts WHERE store_id = $1")
                .bind(store.id)
                .fetch_one(pool)
                .await?
        }
    };
    Ok(count)
}

/// A short human lineage note per store. Raw is the source of truth; curated is
/// derived from it; a registered store publishes slices INTO raw on sync.
fn lineage_note(store: &StoreRow) -> String {
    match store.kind.as_str() {
        "raw" => "source of truth — immutable, append-only".to_string(),
        "curated" => "derived from raw (raw ← curated)".to_string(),
        "operational" | "external" => "publishes slices into raw on sync (store → raw)".to_string(),
        _ => String::new(),
    }
}

// ---------------------------------------------------------------------------
// POST /catalog/stores — register an operational/external store
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateStoreRequest {
    pub name: String,
    /// 'operational' | 'external'. The built-in 'raw'/'curated' kinds cannot be
    /// registered by a user — they auto-register on read.
    pub kind: String,
    #[serde(default)]
    pub schema: Option<Value>,
    #[serde(default)]
    pub access: Option<Value>,
    #[serde(default)]
    pub description: Option<String>,
}

async fn create_store(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<CreateStoreRequest>,
) -> AppResult<Json<StoreRow>> {
    Ok(Json(
        create_store_inner(&state.pool, &auth_user.user_id, req).await?,
    ))
}

pub(crate) async fn create_store_inner(
    pool: &PgPool,
    user_id: &str,
    req: CreateStoreRequest,
) -> AppResult<StoreRow> {
    let name = req.name.trim();
    if name.is_empty() {
        return Err(AppError::bad_request("store name must not be empty"));
    }
    // Only user-managed kinds are registrable; raw/curated are computed layers.
    if !matches!(req.kind.as_str(), "operational" | "external") {
        return Err(AppError::bad_request(
            "kind must be 'operational' or 'external' (raw/curated auto-register)",
        ));
    }
    if matches!(name, RAW_STORE | CURATED_STORE) {
        return Err(AppError::Conflict(format!(
            "'{name}' is a reserved built-in store name"
        )));
    }

    let row = sqlx::query_as::<_, StoreRow>(
        r#"
        INSERT INTO catalog_stores (user_id, name, kind, schema, access, description)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, user_id, name, kind, schema, access, description,
                  created_at, updated_at, last_synced_at
        "#,
    )
    .bind(user_id)
    .bind(name)
    .bind(&req.kind)
    .bind(&req.schema)
    .bind(&req.access)
    .bind(&req.description)
    .fetch_one(pool)
    .await
    .map_err(dup_name_as_conflict)?;
    Ok(row)
}

/// Map the unique-index violation on (user_id, name) to a 409 Conflict.
fn dup_name_as_conflict(e: sqlx::Error) -> AppError {
    if let sqlx::Error::Database(db) = &e {
        if db.is_unique_violation() {
            return AppError::Conflict("a store with that name already exists".to_string());
        }
    }
    AppError::Database(e)
}

// ---------------------------------------------------------------------------
// GET /catalog/stores/:id
// ---------------------------------------------------------------------------

async fn get_store(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<StoreRow>> {
    Ok(Json(
        fetch_store_owned(&state.pool, &auth_user.user_id, id).await?,
    ))
}

/// Fetch a store by id, 404ing anything the caller doesn't own.
async fn fetch_store_owned(pool: &PgPool, user_id: &str, id: Uuid) -> AppResult<StoreRow> {
    let row = sqlx::query_as::<_, StoreRow>(
        r#"
        SELECT id, user_id, name, kind, schema, access, description,
               created_at, updated_at, last_synced_at
        FROM catalog_stores WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| AppError::not_found(format!("store {id}")))?;
    if row.user_id != user_id {
        return Err(AppError::not_found(format!("store {id}")));
    }
    Ok(row)
}

// ---------------------------------------------------------------------------
// PUT /catalog/stores/:id — update a user-registered store
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct UpdateStoreRequest {
    #[serde(default)]
    pub schema: Option<Value>,
    #[serde(default)]
    pub access: Option<Value>,
    #[serde(default)]
    pub description: Option<String>,
}

async fn update_store(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateStoreRequest>,
) -> AppResult<Json<StoreRow>> {
    Ok(Json(
        update_store_inner(&state.pool, &auth_user.user_id, id, req).await?,
    ))
}

pub(crate) async fn update_store_inner(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
    req: UpdateStoreRequest,
) -> AppResult<StoreRow> {
    let existing = fetch_store_owned(pool, user_id, id).await?;
    if matches!(existing.kind.as_str(), "raw" | "curated") {
        return Err(AppError::bad_request(
            "built-in raw/curated stores are computed and cannot be edited",
        ));
    }
    // COALESCE keeps unspecified fields; a caller updates only what it sends.
    let row = sqlx::query_as::<_, StoreRow>(
        r#"
        UPDATE catalog_stores
           SET schema = COALESCE($2, schema),
               access = COALESCE($3, access),
               description = COALESCE($4, description),
               updated_at = NOW()
         WHERE id = $1
        RETURNING id, user_id, name, kind, schema, access, description,
                  created_at, updated_at, last_synced_at
        "#,
    )
    .bind(id)
    .bind(&req.schema)
    .bind(&req.access)
    .bind(&req.description)
    .fetch_one(pool)
    .await?;
    Ok(row)
}

// ---------------------------------------------------------------------------
// DELETE /catalog/stores/:id
// ---------------------------------------------------------------------------

async fn delete_store(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<Value>> {
    delete_store_inner(&state.pool, &auth_user.user_id, id).await?;
    Ok(Json(json!({ "deleted": id })))
}

pub(crate) async fn delete_store_inner(pool: &PgPool, user_id: &str, id: Uuid) -> AppResult<()> {
    let existing = fetch_store_owned(pool, user_id, id).await?;
    if matches!(existing.kind.as_str(), "raw" | "curated") {
        return Err(AppError::bad_request(
            "built-in raw/curated stores cannot be removed",
        ));
    }
    // CASCADE on the FK cleans up this store's staged published facts.
    sqlx::query("DELETE FROM catalog_stores WHERE id = $1")
        .bind(id)
        .execute(pool)
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// POST /catalog/stores/:id/sync — pull published slices into raw
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct SyncResponse {
    pub imported: usize,
    pub skipped: usize,
}

async fn sync_store(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<SyncResponse>> {
    Ok(Json(
        sync_store_inner(&state.pool, &*state.nlp, &auth_user.user_id, id).await?,
    ))
}

/// Pull a store's `catalog_published_facts` slices into `raw_records` through the
/// existing idempotent import path. Each fact becomes one raw record with
/// `source` = the store name and `source_ref` = the published-fact id, so a
/// re-sync dedups on `(user_id, source, source_ref)` and never duplicates. A
/// fact with a JSON payload lands as `state_object`; a plain fact as `semantic`.
///
/// Live query-through for `access.interface = 'sql' | 'http'` is not wired yet;
/// only the staged-slice path is implemented (see the TODO below).
pub(crate) async fn sync_store_inner(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
    id: Uuid,
) -> AppResult<SyncResponse> {
    let store = fetch_store_owned(pool, user_id, id).await?;
    if matches!(store.kind.as_str(), "raw" | "curated") {
        return Err(AppError::bad_request(
            "built-in raw/curated stores are the lake itself and are not synced",
        ));
    }

    // TODO: when access.interface is 'sql' or 'http', query the external store
    // live and buffer the returned rows for import. For now sync only drains the
    // buffered `catalog_published_facts` slices the store has published.

    #[derive(FromRow)]
    struct Fact {
        id: Uuid,
        fact: String,
        event_time: Option<DateTime<Utc>>,
        payload: Option<Value>,
    }
    let facts: Vec<Fact> = sqlx::query_as::<_, Fact>(
        r#"
        SELECT id, fact, event_time, payload
        FROM catalog_published_facts
        WHERE store_id = $1
        ORDER BY created_at ASC
        "#,
    )
    .bind(store.id)
    .fetch_all(pool)
    .await?;

    let records: Vec<ImportRecord> = facts
        .into_iter()
        .map(|f| ImportRecord {
            // A fact carrying a payload is a structured slice (state_object);
            // a bare fact is text the store published, so it lands as a
            // document — evidence arriving from outside, not a tier we derived.
            r#type: if f.payload.is_some() {
                "state_object".to_string()
            } else {
                "document".to_string()
            },
            content: f.fact,
            event_time: f.event_time,
            source: store.name.clone(),
            source_ref: Some(f.id.to_string()),
            project_id: None,
            container_id: None,
            mode: None,
            importance: None,
            payload: f.payload,
        })
        .collect();

    let out = import_records_inner(pool, nlp, user_id, ImportRequest { records }).await?;

    sqlx::query("UPDATE catalog_stores SET last_synced_at = NOW() WHERE id = $1")
        .bind(store.id)
        .execute(pool)
        .await?;

    Ok(SyncResponse {
        imported: out.imported,
        skipped: out.skipped,
    })
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

    async fn ingest_raw(pool: &PgPool, user_id: &str, content: &str) -> Uuid {
        crate::routes::records::ingest_record(
            pool,
            &StubNlp,
            user_id,
            crate::routes::records::IngestRecordRequest {
                r#type: "document".into(),
                content: content.into(),
                event_time: None,
                source: "test".into(),
                source_ref: None,
                project_id: None,
                container_id: None,
                mode: None,
                importance: None,
                supersedes: None,
                payload: None,
            },
        )
        .await
        .unwrap()
        .id
    }

    async fn publish_fact(
        pool: &PgPool,
        store_id: Uuid,
        fact: &str,
        payload: Option<Value>,
    ) -> Uuid {
        sqlx::query_scalar(
            "INSERT INTO catalog_published_facts (store_id, fact, event_time, payload) \
             VALUES ($1, $2, NOW(), $3) RETURNING id",
        )
        .bind(store_id)
        .bind(fact)
        .bind(payload)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    fn op_store(name: &str) -> CreateStoreRequest {
        CreateStoreRequest {
            name: name.into(),
            kind: "operational".into(),
            schema: Some(json!({ "fields": ["amount", "merchant"] })),
            access: Some(json!({ "interface": "sql", "url": "postgres://…" })),
            description: Some("a bank feed".into()),
        }
    }

    // ---- GET /catalog auto-registers raw + curated -----------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn catalog_auto_registers_builtins_with_counts(pool: PgPool) {
        // Two raw records + one curated node for alice.
        ingest_raw(&pool, "alice", "took 5mg lisinopril").await;
        ingest_raw(&pool, "alice", "weight 180").await;
        sqlx::query(
            "INSERT INTO curated_nodes (kind, content, level, user_id) \
             VALUES ('semantic', 'blood pressure meds', 1, 'alice')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let cat = list_catalog_inner(&pool, "alice").await.unwrap();

        assert_eq!(cat.raw.len(), 1, "raw layer auto-registers as one store");
        assert_eq!(cat.curated.len(), 1, "curated layer auto-registers");
        let raw = &cat.raw[0];
        assert_eq!(raw.store.name, "raw_records");
        assert_eq!(raw.store.kind, "raw");
        assert_eq!(raw.record_count, 2);
        assert!(
            raw.store.schema.is_some(),
            "raw store carries a live schema"
        );
        assert!(raw.lineage.contains("source of truth"));

        let cur = &cat.curated[0];
        assert_eq!(cur.record_count, 1);
        assert!(cur.lineage.contains("raw ← curated"));

        // Idempotent: a second read doesn't duplicate the built-ins.
        let cat2 = list_catalog_inner(&pool, "alice").await.unwrap();
        assert_eq!(cat2.raw.len(), 1);
        assert_eq!(cat2.curated.len(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn catalog_lists_registered_operational_store(pool: PgPool) {
        create_store_inner(&pool, "alice", op_store("bank_feed"))
            .await
            .unwrap();
        let cat = list_catalog_inner(&pool, "alice").await.unwrap();
        assert_eq!(cat.operational.len(), 1);
        assert_eq!(cat.operational[0].store.name, "bank_feed");
        assert!(cat.operational[0].lineage.contains("store → raw"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_store_rejects_builtin_kind_and_reserved_name(pool: PgPool) {
        let mut bad_kind = op_store("x");
        bad_kind.kind = "raw".into();
        assert!(create_store_inner(&pool, "alice", bad_kind).await.is_err());

        let mut reserved = op_store("raw_records");
        reserved.kind = "operational".into();
        assert!(create_store_inner(&pool, "alice", reserved).await.is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_store_rejects_duplicate_name(pool: PgPool) {
        create_store_inner(&pool, "alice", op_store("dupe"))
            .await
            .unwrap();
        let err = create_store_inner(&pool, "alice", op_store("dupe"))
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::Conflict(_)));
    }

    // ---- sync pulls published slices idempotently ------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn sync_ingests_published_facts_idempotently(pool: PgPool) {
        let store = create_store_inner(&pool, "alice", op_store("feed"))
            .await
            .unwrap();
        publish_fact(&pool, store.id, "spent 12.50 at cafe", None).await;
        publish_fact(
            &pool,
            store.id,
            "balance snapshot",
            Some(json!({ "kind": "balance", "key": "checking", "data": { "usd": 1000 } })),
        )
        .await;

        let out = sync_store_inner(&pool, &StubNlp, "alice", store.id)
            .await
            .unwrap();
        assert_eq!((out.imported, out.skipped), (2, 0));

        // Both slices landed in raw with source = the store name.
        let n: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM raw_records WHERE user_id = 'alice' AND source = 'feed'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(n, 2);
        // The payload-bearing slice became a state_object.
        let so: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM raw_records WHERE source = 'feed' AND type = 'state_object'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(so, 1);

        // Re-sync: same published-fact ids → dedup on (user, source, source_ref).
        let out2 = sync_store_inner(&pool, &StubNlp, "alice", store.id)
            .await
            .unwrap();
        assert_eq!((out2.imported, out2.skipped), (0, 2));
        let n2: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM raw_records WHERE user_id = 'alice' AND source = 'feed'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(n2, 2, "re-sync must not duplicate");

        // last_synced_at was stamped.
        let synced: Option<DateTime<Utc>> =
            sqlx::query_scalar("SELECT last_synced_at FROM catalog_stores WHERE id = $1")
                .bind(store.id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(synced.is_some());
    }

    // ---- update / delete + scope isolation --------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn update_and_delete_store(pool: PgPool) {
        let store = create_store_inner(&pool, "alice", op_store("feed"))
            .await
            .unwrap();
        let updated = update_store_inner(
            &pool,
            "alice",
            store.id,
            UpdateStoreRequest {
                schema: None,
                access: None,
                description: Some("renamed feed".into()),
            },
        )
        .await
        .unwrap();
        assert_eq!(updated.description.as_deref(), Some("renamed feed"));

        delete_store_inner(&pool, "alice", store.id).await.unwrap();
        assert!(fetch_store_owned(&pool, "alice", store.id).await.is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn stores_are_scope_isolated_by_user(pool: PgPool) {
        let alices = create_store_inner(&pool, "alice", op_store("feed"))
            .await
            .unwrap();
        // Bob can register the same name (scoped per user) and can't see alice's.
        create_store_inner(&pool, "bob", op_store("feed"))
            .await
            .unwrap();
        assert!(fetch_store_owned(&pool, "bob", alices.id).await.is_err());

        let bob_cat = list_catalog_inner(&pool, "bob").await.unwrap();
        assert!(bob_cat.operational.iter().all(|s| s.store.user_id == "bob"));
    }
}
