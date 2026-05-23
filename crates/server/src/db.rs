use anyhow::{Context, Result};
use sqlx::{postgres::PgPoolOptions, PgPool};

pub async fn create_pool(database_url: &str) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await
        .context("connecting to Postgres")?;
    Ok(pool)
}

pub async fn migrate(pool: &PgPool) -> Result<()> {
    sqlx::migrate!("../../migrations")
        .run(pool)
        .await
        .context("running migrations")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[sqlx::test]
    async fn migrate_is_idempotent(pool: PgPool) {
        // First run applies all migrations.
        migrate(&pool).await.unwrap();
        // Second run should be a no-op (no errors).
        migrate(&pool).await.unwrap();

        // Sanity: the memories table exists after migration.
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM memories")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn migrate_creates_expected_tables(pool: PgPool) {
        // Cross-check the table set we depend on actually got created.
        for table in ["memories", "tokens", "core_memory", "consolidation_runs"] {
            let exists: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
            )
            .bind(table)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert!(exists, "table {table} should exist after migrate");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn migrate_creates_pgvector_extension(pool: PgPool) {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists, "pgvector extension should be enabled");
    }
}
