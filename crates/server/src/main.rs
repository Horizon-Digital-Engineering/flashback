mod auth;
mod build_info;
mod catalog;
mod config;
mod curation;
mod db;
mod decay;
mod doctor;
mod error;
mod modes;
mod nlp;
mod proposals;
mod references;
mod routes;
mod settings;
mod summaries;

use std::sync::Arc;

use anyhow::{anyhow, Result};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Clone)]
pub struct AppState {
    pub pool: sqlx::PgPool,
    pub nlp: nlp::SharedNlp,
    pub cfg: Arc<config::Config>,
}

/// Ceiling on a single request body. A bulk import of a few hundred
/// conversations is comfortably inside this; unbounded is what lets one
/// request exhaust the process.
const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Headers that cost nothing and remove whole bug classes. The admin UI is
/// server-rendered with inline styles and inline scripts of our own, so the CSP
/// permits those and nothing else — notably no remote script origins, so an
/// injected `<script src>` has nowhere to load from.
async fn security_headers(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::header::{HeaderName, HeaderValue};
    let mut res = next.run(req).await;
    let h = res.headers_mut();
    for (name, value) in [
        ("x-content-type-options", "nosniff"),
        ("x-frame-options", "DENY"),
        ("referrer-policy", "no-referrer"),
        (
            "content-security-policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; \
             style-src 'self' 'unsafe-inline'; img-src 'self' data:; \
             connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; \
             form-action 'self'",
        ),
    ] {
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            h.insert(n, v);
        }
    }
    res
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "flashback=info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let mut args = std::env::args().skip(1);
    let subcommand = args.next();
    let rest: Vec<String> = args.collect();

    match subcommand.as_deref() {
        Some("migrate") => run_migrate().await,
        Some("doctor") => doctor::run().await,
        Some("token") => run_token(rest).await,
        Some("serve") | None => run_serve().await,
        Some("version") | Some("--version") | Some("-V") => {
            println!("{}", build_info::summary());
            Ok(())
        }
        Some(other) => Err(anyhow!(
            "Unknown subcommand: {other}. Try: serve | migrate | doctor | version | token mint|list|revoke"
        )),
    }
}

async fn run_migrate() -> Result<()> {
    let cfg = config::Config::from_env()?;
    tracing::info!("Running migrations against {}", cfg.database_url_safe());
    let pool = db::create_pool(&cfg.database_url).await?;
    db::migrate(&pool).await?;
    tracing::info!("Migrations complete.");
    Ok(())
}

async fn run_serve() -> Result<()> {
    let cfg = config::Config::from_env()?;

    if cfg.dev_mode {
        // Belt-and-braces loud warning. Repeated in /health output and
        // the admin UI banner.
        tracing::warn!("================================================================");
        tracing::warn!("⚠  FLASHBACK DEV MODE — AUTH BYPASSED");
        tracing::warn!("   Every request is treated as `user_id=dev`.");
        tracing::warn!("   Do NOT expose this server to the public internet.");
        tracing::warn!("   Disable by removing --dev / unsetting FLASHBACK_DEV_MODE.");
        tracing::warn!("================================================================");
    }
    routes::admin::views::set_dev_mode(cfg.dev_mode);

    tracing::info!("Connecting to database...");
    let pool = db::create_pool(&cfg.database_url).await?;

    if cfg.auto_migrate {
        tracing::info!("AUTO_MIGRATE=1 — running migrations on startup");
        db::migrate(&pool).await?;
    }

    // The environment seeds the provider config; the settings row — written
    // from the admin settings page — wins over it once one exists.
    let provider_cfg = settings::resolve_from_db(&pool, &cfg.provider).await;

    tracing::info!("Loading embedding model (this can take a few seconds on first run)...");
    let nlp_cfg = nlp::Config {
        cache_dir: cfg.fastembed_cache_dir.clone(),
    };
    let nlp = Arc::new(nlp::Nlp::new(nlp_cfg, &provider_cfg).await?);

    // Verify the configured models exist at the endpoint — in the background,
    // because a model server that boots slower than this service must not
    // wedge the memory store. Misses log at error level and show in /health
    // as the provider failing, not as silence.
    tokio::spawn(async move { settings::verify_models_at_startup(&provider_cfg).await });

    let cfg_arc = Arc::new(cfg);

    let state = AppState {
        pool,
        nlp,
        cfg: cfg_arc.clone(),
    };

    // Curation scheduler — rebuilds the curated layer from raw_records.
    // FLASHBACK_DISABLE_CURATION=1 turns it off (useful for tests).
    let disable_curation = matches!(
        std::env::var("FLASHBACK_DISABLE_CURATION").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes")
    );
    if !disable_curation {
        spawn_curation_scheduler(state.clone());
    } else {
        tracing::warn!("FLASHBACK_DISABLE_CURATION=1 — curation scheduler not spawned");
    }

    let app = routes::router(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_bearer,
        ))
        .layer(axum::middleware::from_fn(security_headers))
        // No CORS layer. Every browser client of this server — the admin UI,
        // the playground — is same-origin, and hosts that call the API (ritsu,
        // the MCP wrapper) are server-side, where CORS does not apply. A
        // permissive layer here only ever widened what a foreign page could do.
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = cfg_arc.listen_addr();
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    if addr.starts_with("0.0.0.0") && !cfg_arc.dev_mode {
        tracing::warn!(
            "listening on all interfaces ({addr}) — every host that can route here \
             reaches the admin UI and API. Bind a specific address (HOST=...) unless \
             that is intended."
        );
    }
    tracing::info!("Listening on {addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_token(args: Vec<String>) -> Result<()> {
    let cfg = config::Config::from_env()?;
    let pool = db::create_pool(&cfg.database_url).await?;

    // Ensure schema exists — common case: someone tries to mint before serve has run.
    db::migrate(&pool).await?;

    let output = token_dispatch(&pool, &args).await?;
    print!("{output}");
    Ok(())
}

/// Dispatches a `flashback token <mint|list|revoke>` subcommand against the
/// given pool. Returns the formatted human-readable output as a String so
/// the CLI prints it; tests assert on the returned string.
pub(crate) async fn token_dispatch(pool: &sqlx::PgPool, args: &[String]) -> Result<String> {
    let op = args
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("usage: flashback token <mint|list|revoke> [args]"))?;
    let rest = &args[1..];
    match op.as_str() {
        "mint" => token_mint_cli(pool, rest).await,
        "list" => token_list_cli(pool, rest).await,
        "revoke" => token_revoke_cli(pool, rest).await,
        other => Err(anyhow!("unknown token subcommand: {other}")),
    }
}

async fn token_mint_cli(pool: &sqlx::PgPool, args: &[String]) -> Result<String> {
    let user = take_flag(args, "--user").ok_or_else(|| {
        anyhow!("usage: flashback token mint --user=<user_id> [--name=<label>] [--role=service|operator]")
    })?;
    let name = take_flag(args, "--name");
    let role = match take_flag(args, "--role").as_deref() {
        None => auth::TokenRole::Service,
        Some(r) => auth::TokenRole::parse(r).ok_or_else(|| {
            anyhow!("--role must be 'service' (API/MCP) or 'operator' (admin UI)")
        })?,
    };
    let minted = auth::mint_token(pool, &user, name.as_deref(), role).await?;

    // Print exactly once. Never log this. Never store this plaintext.
    let mut out = String::new();
    out.push('\n');
    out.push_str(&format!(
        "  Token minted for user={user} role={}\n",
        role.as_str()
    ));
    if let Some(n) = name {
        out.push_str(&format!("  Name:   {n}\n"));
    }
    out.push_str(&format!("  ID:     {}\n", minted.id));
    out.push_str(&format!("  Prefix: {}\n", minted.prefix));
    out.push_str(&format!("  TOKEN:  {}\n", minted.plaintext));
    out.push('\n');
    out.push_str("  Save this token now. It will not be shown again.\n\n");
    Ok(out)
}

async fn token_list_cli(pool: &sqlx::PgPool, args: &[String]) -> Result<String> {
    let user = take_flag(args, "--user");
    let rows = auth::list_tokens(pool, user.as_deref()).await?;
    if rows.is_empty() {
        return Ok("(no tokens)\n".to_string());
    }
    let mut out = String::new();
    out.push_str(&format!(
        "{:<36}  {:<11}  {:<20}  {:<25}  {}\n",
        "id", "prefix", "user", "name", "status"
    ));
    for row in rows {
        let status = match (row.revoked_at, row.last_used_at) {
            (Some(_), _) => "revoked".to_string(),
            (None, Some(t)) => format!("used {}", t.format("%Y-%m-%d")),
            (None, None) => "unused".to_string(),
        };
        out.push_str(&format!(
            "{:<36}  {:<11}  {:<20}  {:<25}  {}\n",
            row.id,
            row.token_prefix,
            row.user_id,
            row.name.unwrap_or_default(),
            status
        ));
    }
    Ok(out)
}

async fn token_revoke_cli(pool: &sqlx::PgPool, args: &[String]) -> Result<String> {
    let id_str = args
        .first()
        .ok_or_else(|| anyhow!("usage: flashback token revoke <id>"))?;
    let id: uuid::Uuid = id_str
        .parse()
        .map_err(|_| anyhow!("not a valid UUID: {id_str}"))?;
    let revoked = auth::revoke_token(pool, id).await?;
    if revoked {
        Ok(format!("revoked {id}\n"))
    } else {
        Ok(format!(
            "nothing to revoke ({id} not found or already revoked)\n"
        ))
    }
}

/// Spawn a background task that rebuilds the curated layer (promote + distill)
/// for every user on an interval. It lists users from `raw_records` and only
/// ever INSERTs into `curated_*`. Failures are logged, never fatal.
fn spawn_curation_scheduler(state: AppState) {
    let interval_hours = std::env::var("FLASHBACK_CURATION_HOURS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(24);

    tokio::spawn(async move {
        // Settle after startup so first-boot logs don't interleave.
        tokio::time::sleep(std::time::Duration::from_secs(180)).await;
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(interval_hours * 3600));
        loop {
            interval.tick().await;
            tracing::info!("curation: scheduled tick");
            // Incremental: promote/refresh/distill only what changed since the
            // last tick. The wipe-and-re-derive `rebuild` stays available as
            // the recovery tool behind the explicit rebuild endpoint — running
            // it on a schedule re-billed the whole corpus every tick.
            let _ = curation::curate_all_users(&state.pool, &*state.nlp).await;
        }
    });

    tracing::info!(interval_h = interval_hours, "curation scheduler armed");
}

fn take_flag(args: &[String], name: &str) -> Option<String> {
    for (i, a) in args.iter().enumerate() {
        if let Some(rest) = a.strip_prefix(&format!("{name}=")) {
            return Some(rest.to_string());
        }
        if a == name {
            return args.get(i + 1).cloned();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    fn args(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    // ---- take_flag (pure) -----------------------------------------------

    #[test]
    fn take_flag_equals_form() {
        let a = args(&["other", "--user=alice", "--name=test"]);
        assert_eq!(take_flag(&a, "--user"), Some("alice".to_string()));
        assert_eq!(take_flag(&a, "--name"), Some("test".to_string()));
    }

    #[test]
    fn take_flag_space_separated_form() {
        let a = args(&["--user", "alice", "--name", "test"]);
        assert_eq!(take_flag(&a, "--user"), Some("alice".to_string()));
        assert_eq!(take_flag(&a, "--name"), Some("test".to_string()));
    }

    #[test]
    fn take_flag_returns_none_when_missing() {
        let a = args(&["--user=alice"]);
        assert_eq!(take_flag(&a, "--name"), None);
    }

    #[test]
    fn take_flag_returns_none_when_empty() {
        let a: Vec<String> = Vec::new();
        assert_eq!(take_flag(&a, "--user"), None);
    }

    #[test]
    fn take_flag_space_form_no_value_returns_none() {
        // `--user` at end with nothing after → None.
        let a = args(&["--user"]);
        assert_eq!(take_flag(&a, "--user"), None);
    }

    #[test]
    fn take_flag_first_match_wins() {
        let a = args(&["--user=alice", "--user=bob"]);
        assert_eq!(take_flag(&a, "--user"), Some("alice".to_string()));
    }

    // ---- token_dispatch / mint / list / revoke (integration) -----------

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_dispatch_empty_args_errors(pool: PgPool) {
        let err = token_dispatch(&pool, &[]).await.unwrap_err();
        assert!(err.to_string().contains("usage"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_dispatch_unknown_subcommand_errors(pool: PgPool) {
        let err = token_dispatch(&pool, &args(&["frobnicate"]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown token subcommand"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_mint_requires_user_flag(pool: PgPool) {
        let err = token_dispatch(&pool, &args(&["mint"])).await.unwrap_err();
        assert!(err.to_string().contains("--user"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_mint_succeeds_and_prints_token_block(pool: PgPool) {
        let out = token_dispatch(&pool, &args(&["mint", "--user=alice", "--name=test"]))
            .await
            .unwrap();
        assert!(out.contains("Token minted for user=alice"));
        assert!(out.contains("Name:   test"));
        assert!(out.contains("TOKEN:"));
        assert!(out.contains("fb_"));
        assert!(out.contains("Save this token now"));

        // DB confirms.
        let listed = auth::list_tokens(&pool, Some("alice")).await.unwrap();
        assert_eq!(listed.len(), 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_mint_works_without_name_flag(pool: PgPool) {
        let out = token_dispatch(&pool, &args(&["mint", "--user=alice"]))
            .await
            .unwrap();
        assert!(out.contains("Token minted for user=alice"));
        // No "Name:" line in output when --name omitted.
        assert!(!out.contains("Name:"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_list_empty_prints_no_tokens_marker(pool: PgPool) {
        let out = token_dispatch(&pool, &args(&["list"])).await.unwrap();
        assert!(out.contains("(no tokens)"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_list_shows_minted_tokens(pool: PgPool) {
        auth::mint_token(&pool, "alice", Some("primary"), auth::TokenRole::Service)
            .await
            .unwrap();
        auth::mint_token(&pool, "bob", None, auth::TokenRole::Service)
            .await
            .unwrap();

        let all = token_dispatch(&pool, &args(&["list"])).await.unwrap();
        assert!(all.contains("alice"));
        assert!(all.contains("bob"));
        assert!(all.contains("primary"));
        assert!(all.contains("unused")); // status column for never-used tokens

        let alice_only = token_dispatch(&pool, &args(&["list", "--user=alice"]))
            .await
            .unwrap();
        assert!(alice_only.contains("alice"));
        assert!(!alice_only.contains("bob"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_revoke_requires_id(pool: PgPool) {
        let err = token_dispatch(&pool, &args(&["revoke"])).await.unwrap_err();
        assert!(err.to_string().contains("usage"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_revoke_rejects_non_uuid(pool: PgPool) {
        let err = token_dispatch(&pool, &args(&["revoke", "not-a-uuid"]))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a valid UUID"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_revoke_unknown_id_is_friendly(pool: PgPool) {
        let fresh = uuid::Uuid::new_v4();
        let out = token_dispatch(&pool, &args(&["revoke", &fresh.to_string()]))
            .await
            .unwrap();
        assert!(out.contains("nothing to revoke"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn token_revoke_marks_existing_token(pool: PgPool) {
        let minted = auth::mint_token(&pool, "alice", None, auth::TokenRole::Service)
            .await
            .unwrap();
        let out = token_dispatch(&pool, &args(&["revoke", &minted.id.to_string()]))
            .await
            .unwrap();
        assert!(out.contains(&format!("revoked {}", minted.id)));

        // Confirm via DB.
        let listed = auth::list_tokens(&pool, Some("alice")).await.unwrap();
        assert!(listed[0].revoked_at.is_some());
    }
}
