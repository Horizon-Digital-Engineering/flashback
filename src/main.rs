mod config;
mod db;
mod error;
mod models;
mod routes;

use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env if present
    let _ = dotenvy::dotenv();

    // Tracing
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "flashback=info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cfg = config::Config::from_env()?;
    tracing::info!("Connecting to database...");

    let pool = db::create_pool(&cfg.database_url).await?;
    tracing::info!("Database connected.");

    let app = routes::router(pool)
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let listener = tokio::net::TcpListener::bind(&cfg.listen_addr()).await?;
    tracing::info!("Listening on {}", cfg.listen_addr());

    axum::serve(listener, app).await?;
    Ok(())
}
