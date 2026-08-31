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

/// Same database, but connections default to the `playground` schema — so the
/// sandbox reuses every query unchanged and cannot reach real memory.
pub async fn create_playground_pool(database_url: &str) -> Result<PgPool> {
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .after_connect(|conn, _meta| {
            Box::pin(async move {
                sqlx::query("SET search_path TO playground, public")
                    .execute(conn)
                    .await?;
                Ok(())
            })
        })
        .connect(database_url)
        .await
        .context("connecting to Postgres (playground schema)")?;
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

        // Sanity: the raw layer exists after migration.
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM raw_records")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn migrate_creates_expected_tables(pool: PgPool) {
        // Cross-check the canonical table set actually got created.
        for table in [
            "raw_records",
            "raw_embeddings",
            "curated_nodes",
            "proposals",
            "modes",
            "tokens",
        ] {
            let exists: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
            )
            .bind(table)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert!(exists, "table {table} should exist after migrate");
        }

        // The pre-raw world is never created; confirm those tables are absent.
        for table in ["memories", "core_memory", "consolidation_runs"] {
            let exists: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = $1)",
            )
            .bind(table)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert!(!exists, "table {table} should not exist");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn playground_schema_is_a_separate_store(pool: PgPool) {
        sqlx::query(
            "INSERT INTO public.raw_records (type, content, event_time, source, user_id) \
             VALUES ('document','real memory',NOW(),'test','alice')",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO playground.raw_records (type, content, event_time, source, user_id) \
             VALUES ('document','scratch',NOW(),'test','alice')",
        )
        .execute(&pool)
        .await
        .unwrap();

        let real: Vec<String> =
            sqlx::query_scalar("SELECT content FROM public.raw_records WHERE user_id='alice'")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(
            real,
            vec!["real memory".to_string()],
            "scratch must not appear in public"
        );

        let scratch: Vec<String> =
            sqlx::query_scalar("SELECT content FROM playground.raw_records WHERE user_id='alice'")
                .fetch_all(&pool)
                .await
                .unwrap();
        assert_eq!(scratch, vec!["scratch".to_string()]);

        let chained: Option<String> = sqlx::query_scalar(
            "SELECT record_hash FROM playground.raw_records WHERE user_id='alice'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            chained.is_some_and(|h| h.len() == 64),
            "hash trigger carried over"
        );

        // LIKE does not copy triggers, so check them rather than assume. The
        // sandbox must have the SAME storage semantics as production or it stops
        // being a rehearsal: an app-level UPDATE would pass here and fail in prod.
        let triggers: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM pg_trigger t
             JOIN pg_class c ON c.oid = t.tgrelid
             JOIN pg_namespace n ON n.oid = c.relnamespace
             WHERE n.nspname = 'playground' AND c.relname = 'raw_records'
               AND NOT t.tgisinternal",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(triggers, 3, "playground needs every raw_records trigger");

        assert!(
            sqlx::query("UPDATE playground.raw_records SET content = 'rewritten'")
                .execute(&pool)
                .await
                .is_err(),
            "the sandbox must be append-only too"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn playground_has_a_twin_for_every_table_it_writes(pool: PgPool) {
        // search_path falls through to public for anything missing here, so a
        // table the sandbox writes without a twin silently mutates real state.
        for table in [
            "raw_records",
            "raw_embeddings",
            "curated_nodes",
            "curated_edges",
            "curated_embeddings",
            "entity_index",
            "ref_weights",
            "modes",
        ] {
            let exists: bool = sqlx::query_scalar(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables
                 WHERE table_schema = 'playground' AND table_name = $1)",
            )
            .bind(table)
            .fetch_one(&pool)
            .await
            .unwrap();
            assert!(
                exists,
                "playground.{table} missing — writes would hit public"
            );
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
