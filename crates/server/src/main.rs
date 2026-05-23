mod auth;
mod chunking;
mod config;
mod consolidation;
mod db;
mod error;
mod models;
mod nlp;
mod retrieval;
mod routes;
mod state;

use std::sync::Arc;

use anyhow::{anyhow, Result};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Clone)]
pub struct AppState {
    pub pool: sqlx::PgPool,
    pub nlp: nlp::SharedNlp,
    pub cfg: Arc<config::Config>,
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
        Some("token") => run_token(rest).await,
        Some("serve") | None => run_serve().await,
        Some(other) => Err(anyhow!(
            "Unknown subcommand: {other}. Try: serve | migrate | token mint|list|revoke"
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

    tracing::info!("Loading embedding model (this can take a few seconds on first run)...");
    let nlp_cfg = nlp::Config {
        cache_dir: cfg.fastembed_cache_dir.clone(),
    };
    let nlp = Arc::new(nlp::Nlp::new(nlp_cfg, &cfg.provider).await?);

    let cfg_arc = Arc::new(cfg);

    let state = AppState {
        pool,
        nlp,
        cfg: cfg_arc.clone(),
    };

    // Consolidation scheduler — daily promote + weekly distill.
    // FLASHBACK_DISABLE_CONSOLIDATION=1 turns it off (useful for tests).
    let disable_consolidation = matches!(
        std::env::var("FLASHBACK_DISABLE_CONSOLIDATION").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes")
    );
    if !disable_consolidation {
        spawn_consolidation_scheduler(state.clone());
    } else {
        tracing::warn!("FLASHBACK_DISABLE_CONSOLIDATION=1 — scheduler not spawned");
    }

    let app = routes::router(state.clone())
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth::require_bearer,
        ))
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr = cfg_arc.listen_addr();
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("Listening on {addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

async fn run_token(args: Vec<String>) -> Result<()> {
    let cfg = config::Config::from_env()?;
    let pool = db::create_pool(&cfg.database_url).await?;

    // Ensure schema exists — common case: someone tries to mint before serve has run.
    db::migrate(&pool).await?;

    let op = args
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("usage: flashback token <mint|list|revoke> [args]"))?;
    let rest = &args[1..];

    match op.as_str() {
        "mint" => {
            let user = take_flag(rest, "--user").ok_or_else(|| {
                anyhow!("usage: flashback token mint --user=<user_id> [--name=<label>]")
            })?;
            let name = take_flag(rest, "--name");
            let minted = auth::mint_token(&pool, &user, name.as_deref()).await?;

            // Print exactly once. Never log this. Never store this plaintext.
            println!();
            println!("  Token minted for user={user}");
            if let Some(n) = name {
                println!("  Name:   {n}");
            }
            println!("  ID:     {}", minted.id);
            println!("  Prefix: {}", minted.prefix);
            println!("  TOKEN:  {}", minted.plaintext);
            println!();
            println!("  Save this token now. It will not be shown again.");
            println!();
        }
        "list" => {
            let user = take_flag(rest, "--user");
            let rows = auth::list_tokens(&pool, user.as_deref()).await?;
            if rows.is_empty() {
                println!("(no tokens)");
            } else {
                println!(
                    "{:<36}  {:<11}  {:<20}  {:<25}  {}",
                    "id", "prefix", "user", "name", "status"
                );
                for row in rows {
                    let status = match (row.revoked_at, row.last_used_at) {
                        (Some(_), _) => "revoked".to_string(),
                        (None, Some(t)) => format!("used {}", t.format("%Y-%m-%d")),
                        (None, None) => "unused".to_string(),
                    };
                    println!(
                        "{:<36}  {:<11}  {:<20}  {:<25}  {}",
                        row.id,
                        row.token_prefix,
                        row.user_id,
                        row.name.unwrap_or_default(),
                        status
                    );
                }
            }
        }
        "revoke" => {
            let id_str = rest
                .first()
                .ok_or_else(|| anyhow!("usage: flashback token revoke <id>"))?;
            let id: uuid::Uuid = id_str
                .parse()
                .map_err(|_| anyhow!("not a valid UUID: {id_str}"))?;
            let revoked = auth::revoke_token(&pool, id).await?;
            if revoked {
                println!("revoked {id}");
            } else {
                println!("nothing to revoke ({id} not found or already revoked)");
            }
        }
        other => return Err(anyhow!("unknown token subcommand: {other}")),
    }
    Ok(())
}

/// Spawn a background tokio task that runs the consolidation jobs on a
/// schedule. The first tick fires 60s after startup (gives the server time
/// to settle); subsequent ticks fire on the configured intervals.
///
/// Failures are logged, not panicked — a single failed consolidation should
/// never bring down the server.
fn spawn_consolidation_scheduler(state: AppState) {
    let daily_interval = std::env::var("FLASHBACK_CONSOLIDATION_DAILY_HOURS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(24);
    let weekly_interval = std::env::var("FLASHBACK_CONSOLIDATION_WEEKLY_HOURS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(24 * 7);

    let s_daily = state.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(60)).await;
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(daily_interval * 3600));
        // First tick fires immediately; that's fine since we already slept 60s.
        loop {
            interval.tick().await;
            tracing::info!("consolidation: daily tick");
            let _ = consolidation::run_daily_all_users(&s_daily.pool).await;
        }
    });

    let s_weekly = state;
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_secs(120)).await;
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(weekly_interval * 3600));
        loop {
            interval.tick().await;
            tracing::info!("consolidation: weekly tick");
            let _ = consolidation::run_weekly_all_users(&s_weekly.pool, &s_weekly.nlp).await;
        }
    });

    tracing::info!(
        daily_h = daily_interval,
        weekly_h = weekly_interval,
        "consolidation scheduler armed"
    );
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
