//! `flashback doctor` — read-only deployment diagnostics.
//!
//! Checks the things that actually break installs, in dependency order:
//! config, database, pgvector, migrations, embedding cache, AI provider,
//! and whether a server is already running. Prints a human-readable report;
//! exits non-zero when any check fails so scripts can gate on it. Never
//! writes to the database or mutates anything.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;

use crate::config::{Config, ProviderKind, RemoteProviderConfig};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Ok,
    Info,
    Warn,
    Fail,
}

/// Collects check results and renders them as aligned report lines.
struct Report {
    warnings: usize,
    failures: usize,
}

impl Report {
    fn new() -> Self {
        Self {
            warnings: 0,
            failures: 0,
        }
    }

    fn line(&mut self, level: Level, check: &str, detail: &str) {
        let tag = match level {
            Level::Ok => " ok ",
            Level::Info => "info",
            Level::Warn => "WARN",
            Level::Fail => "FAIL",
        };
        match level {
            Level::Warn => self.warnings += 1,
            Level::Fail => self.failures += 1,
            _ => {}
        }
        println!("  [{tag}] {check:<12} {detail}");
    }
}

pub async fn run() -> Result<()> {
    println!("flashback doctor\n");
    let mut r = Report::new();

    let cfg = match Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            r.line(Level::Fail, "config", &format!("{e:#}"));
            println!("\n1 check(s) failed. Nothing else can run without configuration.");
            std::process::exit(1);
        }
    };
    r.line(
        Level::Ok,
        "config",
        &format!(
            "listen {}, database {}",
            cfg.listen_addr(),
            cfg.database_url_safe()
        ),
    );
    if cfg.dev_mode {
        r.line(
            Level::Warn,
            "dev-mode",
            "auth is BYPASSED — never expose this server beyond localhost",
        );
    }

    // --- Database ---------------------------------------------------------
    let pool = match connect(&cfg.database_url).await {
        Ok(pool) => {
            let version = server_version(&pool)
                .await
                .unwrap_or_else(|_| "unknown".to_string());
            r.line(
                Level::Ok,
                "database",
                &format!("connected (Postgres {version})"),
            );
            Some(pool)
        }
        Err(e) => {
            r.line(Level::Fail, "database", &format!("{e:#}"));
            None
        }
    };

    if let Some(pool) = &pool {
        match pgvector_state(pool).await {
            Ok(PgVector::Installed) => {
                r.line(Level::Ok, "pgvector", "extension installed");
            }
            Ok(PgVector::Available) => r.line(
                Level::Warn,
                "pgvector",
                "available but not installed — migrations run CREATE EXTENSION vector on first apply",
            ),
            Ok(PgVector::Missing) => r.line(
                Level::Fail,
                "pgvector",
                "not available on this Postgres server — install the pgvector package matching your Postgres major",
            ),
            Err(e) => r.line(Level::Fail, "pgvector", &format!("{e:#}")),
        }

        match migration_status(pool).await {
            Ok((applied, bundled)) => {
                let (level, msg) = migration_verdict(applied, bundled, cfg.auto_migrate);
                r.line(level, "migrations", &msg);
            }
            Err(e) => r.line(Level::Fail, "migrations", &format!("{e:#}")),
        }
    }

    // --- Embedding model cache --------------------------------------------
    match &cfg.fastembed_cache_dir {
        None => r.line(
            Level::Info,
            "embeddings",
            "FLASHBACK_FASTEMBED_CACHE unset — the model downloads to the default cache on first start",
        ),
        Some(dir) if dir_is_populated(dir) => r.line(
            Level::Ok,
            "embeddings",
            &format!("model cache present at {}", dir.display()),
        ),
        Some(dir) => r.line(
            Level::Warn,
            "embeddings",
            &format!(
                "cache {} is empty — first start downloads the model (or run flashback-nlp-prefetch)",
                dir.display()
            ),
        ),
    }

    // --- AI provider ------------------------------------------------------
    match cfg.provider.kind {
        ProviderKind::Heuristic => r.line(
            Level::Ok,
            "provider",
            "heuristic — in-process, nothing to reach",
        ),
        ProviderKind::Remote => {
            let remote = &cfg.provider.remote;
            if remote.api_key.is_empty() {
                if remote.api_base.is_some() {
                    r.line(
                        Level::Warn,
                        "provider",
                        "remote with an empty API key — fine for most self-hosted endpoints; set PROVIDER_REMOTE_API_KEY if yours requires one",
                    );
                } else {
                    r.line(
                        Level::Fail,
                        "provider",
                        &format!(
                            "remote backend '{}' has no API key — set PROVIDER_REMOTE_API_KEY (or the backend-specific variable)",
                            remote.backend
                        ),
                    );
                }
            }
            let base = resolve_remote_base(remote);
            // Unauthenticated reachability probe: any HTTP status counts as
            // reachable, and the API key is deliberately not sent.
            match probe(&base).await {
                Ok(status) => r.line(
                    Level::Ok,
                    "provider",
                    &format!(
                        "remote endpoint {base} reachable (HTTP {status}); extract={}, distill={}",
                        remote.extract_model, remote.distill_model
                    ),
                ),
                Err(e) => r.line(
                    Level::Fail,
                    "provider",
                    &format!("remote endpoint {base} unreachable: {e}"),
                ),
            }
        }
        ProviderKind::Embedded => {
            if flashback_nlp::EMBEDDED_LLM_COMPILED {
                r.line(
                    Level::Ok,
                    "provider",
                    &format!("embedded — model {}", cfg.provider.embedded.model),
                );
            } else {
                r.line(
                    Level::Warn,
                    "provider",
                    "PROVIDER=embedded but this binary lacks the embedded-llm feature — the server falls back to heuristic; rebuild with --features flashback-nlp/embedded-llm",
                );
            }
        }
    }

    // --- Is a server already running on the configured port? --------------
    match probe(&format!("http://127.0.0.1:{}/health", cfg.port)).await {
        Ok(_) => r.line(
            Level::Info,
            "server",
            &format!("already running on port {}", cfg.port),
        ),
        Err(_) => r.line(
            Level::Info,
            "server",
            &format!(
                "not running on port {} (fine if you're about to start it)",
                cfg.port
            ),
        ),
    }

    println!();
    if r.failures > 0 {
        println!(
            "{} check(s) failed, {} warning(s). Fix the failures above and re-run.",
            r.failures, r.warnings
        );
        // The report above is the whole message — exit non-zero without the
        // anyhow error chain a returned Err would print on top of it.
        std::process::exit(1);
    } else if r.warnings > 0 {
        println!("{} warning(s), no failures.", r.warnings);
    } else {
        println!("All checks passed.");
    }
    Ok(())
}

// --- Helpers (pool-only / pure, so they unit-test without AppState) --------

async fn connect(database_url: &str) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .acquire_timeout(CONNECT_TIMEOUT)
        .connect(database_url)
        .await?;
    Ok(pool)
}

async fn server_version(pool: &PgPool) -> Result<String> {
    let v: String = sqlx::query_scalar("SHOW server_version")
        .fetch_one(pool)
        .await?;
    Ok(v)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PgVector {
    /// CREATE EXTENSION has run in this database.
    Installed,
    /// The server ships the extension but no database has created it yet.
    Available,
    /// The pgvector package is not installed on the server at all.
    Missing,
}

async fn pgvector_state(pool: &PgPool) -> Result<PgVector> {
    let installed: bool =
        sqlx::query_scalar("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            .fetch_one(pool)
            .await?;
    if installed {
        return Ok(PgVector::Installed);
    }
    let available: bool = sqlx::query_scalar(
        "SELECT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'vector')",
    )
    .fetch_one(pool)
    .await?;
    Ok(if available {
        PgVector::Available
    } else {
        PgVector::Missing
    })
}

/// (applied, bundled) — how many migrations the database has vs how many
/// this binary carries. `_sqlx_migrations` not existing counts as zero
/// applied (a fresh database).
async fn migration_status(pool: &PgPool) -> Result<(i64, i64)> {
    let bundled = sqlx::migrate!("../../migrations").migrations.len() as i64;
    let table_exists: bool =
        sqlx::query_scalar("SELECT to_regclass('public._sqlx_migrations') IS NOT NULL")
            .fetch_one(pool)
            .await?;
    let applied: i64 = if table_exists {
        sqlx::query_scalar("SELECT COUNT(*) FROM _sqlx_migrations WHERE success")
            .fetch_one(pool)
            .await?
    } else {
        0
    };
    Ok((applied, bundled))
}

fn migration_verdict(applied: i64, bundled: i64, auto_migrate: bool) -> (Level, String) {
    let pending = bundled - applied;
    if pending == 0 {
        (Level::Ok, format!("all {bundled} applied"))
    } else if pending > 0 && auto_migrate {
        (
            Level::Warn,
            format!("{pending} pending — applied automatically on next start (AUTO_MIGRATE=1)"),
        )
    } else if pending > 0 {
        (
            Level::Fail,
            format!("{pending} pending and AUTO_MIGRATE is off — run: flashback migrate"),
        )
    } else {
        (
            Level::Warn,
            format!(
                "database has {applied} migrations but this binary bundles {bundled} — binary older than the database?"
            ),
        )
    }
}

fn dir_is_populated(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false)
}

/// Mirrors the remote provider's base-URL defaults so the doctor probes the
/// same endpoint the server would call.
fn resolve_remote_base(remote: &RemoteProviderConfig) -> String {
    remote.api_base.clone().unwrap_or_else(|| {
        match remote.backend.as_str() {
            "anthropic" => "https://api.anthropic.com",
            "openai" => "https://api.openai.com/v1",
            _ => "https://openrouter.ai/api/v1", // openrouter default
        }
        .to_string()
    })
}

/// GET the URL; any HTTP status (even 401/404) proves reachability.
async fn probe(url: &str) -> Result<u16> {
    let client = reqwest::Client::builder()
        .timeout(PROBE_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()?;
    let resp = client.get(url).send().await?;
    Ok(resp.status().as_u16())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;

    fn remote_cfg(backend: &str, api_base: Option<&str>) -> RemoteProviderConfig {
        // Reuse the env-default construction, then override the fields under
        // test, so this stays in sync with the real config shape.
        let mut remote = ProviderConfig::from_env().remote;
        remote.backend = backend.to_string();
        remote.api_base = api_base.map(str::to_string);
        remote
    }

    // ---- resolve_remote_base ---------------------------------------------

    #[test]
    fn resolve_remote_base_uses_backend_defaults() {
        assert_eq!(
            resolve_remote_base(&remote_cfg("anthropic", None)),
            "https://api.anthropic.com"
        );
        assert_eq!(
            resolve_remote_base(&remote_cfg("openai", None)),
            "https://api.openai.com/v1"
        );
        assert_eq!(
            resolve_remote_base(&remote_cfg("openrouter", None)),
            "https://openrouter.ai/api/v1"
        );
        // Unknown backends fall through to the openrouter default.
        assert_eq!(
            resolve_remote_base(&remote_cfg("garbage", None)),
            "https://openrouter.ai/api/v1"
        );
    }

    #[test]
    fn resolve_remote_base_prefers_explicit_api_base() {
        assert_eq!(
            resolve_remote_base(&remote_cfg("openai", Some("http://127.0.0.1:11434/v1"))),
            "http://127.0.0.1:11434/v1"
        );
    }

    // ---- migration_verdict -----------------------------------------------

    #[test]
    fn migration_verdict_all_applied_is_ok() {
        let (level, msg) = migration_verdict(12, 12, false);
        assert_eq!(level, Level::Ok);
        assert!(msg.contains("all 12"));
    }

    #[test]
    fn migration_verdict_pending_with_auto_migrate_warns() {
        let (level, msg) = migration_verdict(10, 12, true);
        assert_eq!(level, Level::Warn);
        assert!(msg.contains("2 pending"));
    }

    #[test]
    fn migration_verdict_pending_without_auto_migrate_fails() {
        let (level, msg) = migration_verdict(0, 12, false);
        assert_eq!(level, Level::Fail);
        assert!(msg.contains("flashback migrate"));
    }

    #[test]
    fn migration_verdict_database_ahead_of_binary_warns() {
        let (level, msg) = migration_verdict(13, 12, true);
        assert_eq!(level, Level::Warn);
        assert!(msg.contains("older"));
    }

    // ---- dir_is_populated ------------------------------------------------

    #[test]
    fn dir_is_populated_distinguishes_missing_empty_and_full() {
        let base = std::env::temp_dir().join(format!("doctor-test-{}", uuid::Uuid::new_v4()));
        assert!(!dir_is_populated(&base)); // missing

        std::fs::create_dir_all(&base).unwrap();
        assert!(!dir_is_populated(&base)); // empty

        std::fs::write(base.join("model.onnx"), b"x").unwrap();
        assert!(dir_is_populated(&base)); // populated

        std::fs::remove_dir_all(&base).unwrap();
    }

    // ---- database-backed checks ------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn pgvector_state_reports_installed_after_migrate(pool: PgPool) {
        assert_eq!(pgvector_state(&pool).await.unwrap(), PgVector::Installed);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn migration_status_matches_bundled_after_migrate(pool: PgPool) {
        let (applied, bundled) = migration_status(&pool).await.unwrap();
        assert_eq!(applied, bundled);
        assert!(bundled > 0);
    }

    #[sqlx::test]
    async fn migration_status_zero_applied_on_fresh_database(pool: PgPool) {
        let (applied, bundled) = migration_status(&pool).await.unwrap();
        assert_eq!(applied, 0);
        assert!(bundled > 0);
    }
}
