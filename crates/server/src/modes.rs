//! Modes (cognitive registers) — the mode axis of the memory model.
//!
//! A mode pins an embedder + vector dimension, so a record ingested in one mode
//! is embedded in that mode's geometry and only ever compared against records in
//! the same geometry. Retrieval and curation both scope by `(user_id, mode)`
//! and never cross a mode boundary.
//!
//! Four built-in registers ship out of the box (general/code/journal/research),
//! seeded under the reserved `'*'` template user by migration 014. The first
//! time a real user touches modes, `ensure_builtin_modes` clones the template
//! into that user's rows (the same idiom the catalog uses to auto-register its
//! built-in stores). Users can then declare their own registers on top; the
//! built-ins are protected from deletion.
//!
//! Mode precedence on ingest, high to low, falling back to `general` when
//! nothing resolves:
//!
//! 1. caller override (an explicit `mode` on the ingest call)
//! 2. LLM auto-classification (the `mode` field the AiProvider extracts)
//! 3. project default (the user's default mode)

use serde::{Deserialize, Serialize};
use sqlx::{FromRow, PgPool};

use crate::error::{AppError, AppResult};

/// The reserved template user whose rows hold the canonical built-in registers.
pub const TEMPLATE_USER: &str = "*";

/// The built-in register names — protected from deletion, cloned per-user.
pub const BUILTIN_MODES: &[&str] = &["general", "code", "journal", "research"];

/// The default register every fallback lands in.
pub const DEFAULT_MODE: &str = "general";

#[derive(Debug, Clone, Serialize, FromRow)]
pub struct Mode {
    pub user_id: String,
    pub name: String,
    pub embedder: String,
    pub embedding_dim: i32,
    pub description: Option<String>,
    pub default_decay: Option<String>,
    pub prompt_overrides: Option<serde_json::Value>,
    pub is_default: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Idempotently clone the built-in registers from the `'*'` template into a
/// user's own rows. Cheap and safe to call on every mode-touching path — a
/// row that already exists is left untouched (so a user who renamed/re-pointed
/// a built-in keeps their edit). Mirrors the catalog's `ensure_builtin_stores`.
pub async fn ensure_builtin_modes(pool: &PgPool, user_id: &str) -> AppResult<()> {
    if user_id == TEMPLATE_USER {
        return Ok(());
    }
    sqlx::query(
        r#"
        INSERT INTO modes
            (user_id, name, embedder, embedding_dim, description, default_decay, prompt_overrides, is_default)
        SELECT $1, name, embedder, embedding_dim, description, default_decay, prompt_overrides, is_default
        FROM modes
        WHERE user_id = $2
        ON CONFLICT (user_id, name) DO NOTHING
        "#,
    )
    .bind(user_id)
    .bind(TEMPLATE_USER)
    .execute(pool)
    .await?;
    Ok(())
}

/// List a user's modes (built-ins ensured first), ordered default-first then by
/// name.
pub async fn list_modes(pool: &PgPool, user_id: &str) -> AppResult<Vec<Mode>> {
    ensure_builtin_modes(pool, user_id).await?;
    let rows = sqlx::query_as::<_, Mode>(
        r#"
        SELECT user_id, name, embedder, embedding_dim, description, default_decay,
               prompt_overrides, is_default, created_at
        FROM modes
        WHERE user_id = $1
        ORDER BY is_default DESC, name ASC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Fetch one mode by name (built-ins ensured first). `None` if the user has no
/// such register.
pub async fn get_mode(pool: &PgPool, user_id: &str, name: &str) -> AppResult<Option<Mode>> {
    ensure_builtin_modes(pool, user_id).await?;
    let row = sqlx::query_as::<_, Mode>(
        r#"
        SELECT user_id, name, embedder, embedding_dim, description, default_decay,
               prompt_overrides, is_default, created_at
        FROM modes
        WHERE user_id = $1 AND name = $2
        "#,
    )
    .bind(user_id)
    .bind(name)
    .fetch_optional(pool)
    .await?;
    Ok(row)
}

/// The user's default register (built-ins ensured first). Falls back to a
/// synthesized `general` if — against the invariant — no default exists.
pub async fn default_mode(pool: &PgPool, user_id: &str) -> AppResult<Mode> {
    ensure_builtin_modes(pool, user_id).await?;
    let row = sqlx::query_as::<_, Mode>(
        r#"
        SELECT user_id, name, embedder, embedding_dim, description, default_decay,
               prompt_overrides, is_default, created_at
        FROM modes
        WHERE user_id = $1 AND is_default
        LIMIT 1
        "#,
    )
    .bind(user_id)
    .fetch_optional(pool)
    .await?;
    match row {
        Some(m) => Ok(m),
        None => get_mode(pool, user_id, DEFAULT_MODE)
            .await?
            .ok_or_else(|| AppError::Internal(anyhow::anyhow!("no default mode for {user_id}"))),
    }
}

/// Resolve the mode a record lands in, applying the precedence chain:
/// caller override → LLM auto-classification → project default → `general`.
/// Only a name known to the user is honored at each level; an unknown name
/// degrades to the next signal. Returns the resolved [`Mode`] (embedder + dim
/// included) so the caller can embed with it directly.
pub async fn resolve_mode(
    pool: &PgPool,
    user_id: &str,
    caller_override: Option<&str>,
    llm_classified: Option<&str>,
) -> AppResult<Mode> {
    ensure_builtin_modes(pool, user_id).await?;

    // 1. caller override (explicit, wins).
    if let Some(name) = caller_override.map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(m) = get_mode(pool, user_id, name).await? {
            return Ok(m);
        }
    }
    // 2. LLM auto-classification.
    if let Some(name) = llm_classified.map(str::trim).filter(|s| !s.is_empty()) {
        if let Some(m) = get_mode(pool, user_id, name).await? {
            return Ok(m);
        }
    }
    // 3. project (user) default → general.
    default_mode(pool, user_id).await
}

// ---------------------------------------------------------------------------
// CRUD (user-defined registers). Built-ins are protected from deletion.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct UpsertModeRequest {
    pub embedder: String,
    #[serde(default)]
    pub embedding_dim: Option<i32>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub default_decay: Option<String>,
    #[serde(default)]
    pub prompt_overrides: Option<serde_json::Value>,
    #[serde(default)]
    pub is_default: Option<bool>,
}

/// Create a new register. Rejects a name that already exists (use update) and
/// an unsupported embedder. When `embedding_dim` is omitted it's derived from
/// the embedder; when supplied it must match the embedder's real dimension.
pub async fn create_mode(
    pool: &PgPool,
    user_id: &str,
    name: &str,
    req: UpsertModeRequest,
) -> AppResult<Mode> {
    ensure_builtin_modes(pool, user_id).await?;
    let name = name.trim();
    if name.is_empty() {
        return Err(AppError::bad_request("mode name must not be empty"));
    }
    if get_mode(pool, user_id, name).await?.is_some() {
        return Err(AppError::bad_request(format!("mode {name} already exists")));
    }
    let dim = resolve_dim(&req.embedder, req.embedding_dim)?;
    let is_default = req.is_default.unwrap_or(false);

    let mut tx = pool.begin().await?;
    if is_default {
        clear_default(&mut tx, user_id).await?;
    }
    sqlx::query(
        r#"
        INSERT INTO modes
            (user_id, name, embedder, embedding_dim, description, default_decay, prompt_overrides, is_default)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        "#,
    )
    .bind(user_id)
    .bind(name)
    .bind(&req.embedder)
    .bind(dim)
    .bind(&req.description)
    .bind(&req.default_decay)
    .bind(&req.prompt_overrides)
    .bind(is_default)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;

    get_mode(pool, user_id, name)
        .await?
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("mode vanished after insert")))
}

/// Update an existing register (built-in or user-defined — its embedder/decay/
/// default can all be re-pointed). 404s an unknown name.
pub async fn update_mode(
    pool: &PgPool,
    user_id: &str,
    name: &str,
    req: UpsertModeRequest,
) -> AppResult<Mode> {
    let name = name.trim();
    let existing = get_mode(pool, user_id, name)
        .await?
        .ok_or_else(|| AppError::not_found(format!("mode {name}")))?;
    let dim = resolve_dim(&req.embedder, req.embedding_dim)?;
    let is_default = req.is_default.unwrap_or(existing.is_default);

    let mut tx = pool.begin().await?;
    if is_default {
        clear_default(&mut tx, user_id).await?;
    }
    sqlx::query(
        r#"
        UPDATE modes
        SET embedder = $3, embedding_dim = $4, description = $5,
            default_decay = $6, prompt_overrides = $7, is_default = $8
        WHERE user_id = $1 AND name = $2
        "#,
    )
    .bind(user_id)
    .bind(name)
    .bind(&req.embedder)
    .bind(dim)
    .bind(&req.description)
    .bind(&req.default_decay)
    .bind(&req.prompt_overrides)
    .bind(is_default)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;

    get_mode(pool, user_id, name)
        .await?
        .ok_or_else(|| AppError::Internal(anyhow::anyhow!("mode vanished after update")))
}

/// Delete a user-defined register. Built-ins are protected (400). 404s an
/// unknown name.
pub async fn delete_mode(pool: &PgPool, user_id: &str, name: &str) -> AppResult<()> {
    let name = name.trim();
    if BUILTIN_MODES.contains(&name) {
        return Err(AppError::bad_request(format!(
            "'{name}' is a built-in mode and cannot be deleted"
        )));
    }
    let existing = get_mode(pool, user_id, name)
        .await?
        .ok_or_else(|| AppError::not_found(format!("mode {name}")))?;
    if existing.is_default {
        return Err(AppError::bad_request(
            "cannot delete the default mode; set another default first",
        ));
    }
    sqlx::query("DELETE FROM modes WHERE user_id = $1 AND name = $2")
        .bind(user_id)
        .bind(name)
        .execute(pool)
        .await?;
    Ok(())
}

/// Validate + resolve an embedder key to its dimension. Rejects an unsupported
/// key and a supplied dim that disagrees with the embedder's real dimension.
fn resolve_dim(embedder: &str, supplied: Option<i32>) -> AppResult<i32> {
    let (_, dim) = flashback_nlp::model_for_key(embedder).ok_or_else(|| {
        AppError::bad_request(format!(
            "unsupported embedder '{embedder}' (must be a fastembed model key)"
        ))
    })?;
    let dim = dim as i32;
    match supplied {
        Some(s) if s != dim => Err(AppError::bad_request(format!(
            "embedding_dim {s} does not match embedder '{embedder}' (dim {dim})"
        ))),
        _ => Ok(dim),
    }
}

/// Clear the current default flag for a user inside a transaction — the partial
/// unique index allows only one default per user, so a new default must unset
/// the old one first.
async fn clear_default(
    tx: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    user_id: &str,
) -> AppResult<()> {
    sqlx::query("UPDATE modes SET is_default = false WHERE user_id = $1 AND is_default")
        .bind(user_id)
        .execute(&mut **tx)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    #[sqlx::test(migrations = "../../migrations")]
    async fn builtins_are_seeded_per_user(pool: PgPool) {
        let modes = list_modes(&pool, "alice").await.unwrap();
        let names: Vec<&str> = modes.iter().map(|m| m.name.as_str()).collect();
        for b in BUILTIN_MODES {
            assert!(names.contains(b), "missing built-in {b}");
        }
        // general is the default and pins the 384-dim MiniLM model.
        let general = modes.iter().find(|m| m.name == "general").unwrap();
        assert!(general.is_default);
        assert_eq!(general.embedding_dim, 384);
        // code pins the 768-dim jina-code model; research the 1024-dim bge-large.
        assert_eq!(
            modes
                .iter()
                .find(|m| m.name == "code")
                .unwrap()
                .embedding_dim,
            768
        );
        assert_eq!(
            modes
                .iter()
                .find(|m| m.name == "research")
                .unwrap()
                .embedding_dim,
            1024
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn builtins_are_scoped_per_user(pool: PgPool) {
        list_modes(&pool, "alice").await.unwrap();
        list_modes(&pool, "bob").await.unwrap();
        let n: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM modes WHERE user_id = 'alice' AND name = 'code'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(n, 1);
        // Bob has his own copy — deleting is per-user (tested via CRUD below).
        let bob: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM modes WHERE user_id = 'bob'")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(bob as usize, BUILTIN_MODES.len());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn crud_creates_updates_and_deletes_user_modes(pool: PgPool) {
        let created = create_mode(
            &pool,
            "alice",
            "rust",
            UpsertModeRequest {
                embedder: "jina-code".into(),
                embedding_dim: None,
                description: Some("rust code".into()),
                default_decay: None,
                prompt_overrides: None,
                is_default: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(created.embedding_dim, 768);
        assert_eq!(created.embedder, "jina-code");

        // Update re-points the embedder to a 1024-dim model.
        let updated = update_mode(
            &pool,
            "alice",
            "rust",
            UpsertModeRequest {
                embedder: "BAAI/bge-large-en-v1.5".into(),
                embedding_dim: None,
                description: None,
                default_decay: None,
                prompt_overrides: None,
                is_default: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(updated.embedding_dim, 1024);

        // Delete the user mode.
        delete_mode(&pool, "alice", "rust").await.unwrap();
        assert!(get_mode(&pool, "alice", "rust").await.unwrap().is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn builtins_are_protected_from_delete(pool: PgPool) {
        for b in BUILTIN_MODES {
            let err = delete_mode(&pool, "alice", b).await.unwrap_err();
            assert!(
                err.to_string().contains("built-in"),
                "{b} should be protected"
            );
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_rejects_bad_embedder_and_dim_mismatch(pool: PgPool) {
        let bad = create_mode(
            &pool,
            "alice",
            "x",
            UpsertModeRequest {
                embedder: "not-a-model".into(),
                embedding_dim: None,
                description: None,
                default_decay: None,
                prompt_overrides: None,
                is_default: None,
            },
        )
        .await
        .unwrap_err();
        assert!(bad.to_string().contains("unsupported embedder"));

        let mismatch = create_mode(
            &pool,
            "alice",
            "y",
            UpsertModeRequest {
                embedder: "jina-code".into(),
                embedding_dim: Some(384),
                description: None,
                default_decay: None,
                prompt_overrides: None,
                is_default: None,
            },
        )
        .await
        .unwrap_err();
        assert!(mismatch.to_string().contains("does not match"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn resolve_mode_precedence(pool: PgPool) {
        // caller override wins over LLM classification.
        let m = resolve_mode(&pool, "alice", Some("code"), Some("journal"))
            .await
            .unwrap();
        assert_eq!(m.name, "code");
        // LLM classification wins when no override.
        let m = resolve_mode(&pool, "alice", None, Some("journal"))
            .await
            .unwrap();
        assert_eq!(m.name, "journal");
        // Falls back to the default (general) when neither resolves.
        let m = resolve_mode(&pool, "alice", None, None).await.unwrap();
        assert_eq!(m.name, "general");
        // An unknown override degrades to the next signal.
        let m = resolve_mode(&pool, "alice", Some("nope"), Some("code"))
            .await
            .unwrap();
        assert_eq!(m.name, "code");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn only_one_default_after_create_default(pool: PgPool) {
        create_mode(
            &pool,
            "alice",
            "mine",
            UpsertModeRequest {
                embedder: "BAAI/bge-base-en-v1.5".into(),
                embedding_dim: None,
                description: None,
                default_decay: None,
                prompt_overrides: None,
                is_default: Some(true),
            },
        )
        .await
        .unwrap();
        let n: i64 =
            sqlx::query_scalar("SELECT COUNT(*) FROM modes WHERE user_id = 'alice' AND is_default")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(n, 1);
        let def = default_mode(&pool, "alice").await.unwrap();
        assert_eq!(def.name, "mine");
    }
}
