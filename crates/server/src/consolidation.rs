//! Consolidation worker — the daily/weekly job that turns raw conversational
//! turns into long-lived structured memory.
//!
//! Three timescales, two are implemented in Phase 4:
//!
//!   - **Daily** (`run_daily`): sweeps `working` memories that are about to
//!     expire. Promotes high-importance, multi-accessed ones to `episodic`
//!     (with the medium decay class); lets the rest expire normally.
//!
//!   - **Weekly** (`run_weekly`): finds `episodic` memories ≥7 days old,
//!     groups them into clusters by entity overlap, and asks the configured
//!     AI provider to distill each cluster into a small number of
//!     `semantic` facts. Source episodes get pointed at the new semantic
//!     row via the existing `supersedes` mechanism so the trail is walkable.
//!
//!   - Monthly is Phase 5 (procedural-pattern extraction).
//!
//! Both jobs are idempotent-enough: re-running daily within the same window
//! finds nothing to promote (all candidates already moved); re-running
//! weekly distills any newly-eligible clusters. Audit rows live in
//! `consolidation_runs`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use pgvector::Vector;
use sqlx::PgPool;
use uuid::Uuid;

use flashback_nlp::{AiProvider, EpisodeRef};

use crate::nlp::Nlp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Daily,
    Weekly,
}

impl Kind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Kind::Daily => "daily",
            Kind::Weekly => "weekly",
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct RunStats {
    pub kind: &'static str,
    pub promoted_count: i32,
    pub expired_count: i32,
    pub distilled_count: i32,
    pub clusters_seen: i32,
    pub error: Option<String>,
}

/// Run the daily job for a single user. No-op if there's nothing to promote.
pub async fn run_daily(pool: &PgPool, user_id: &str) -> RunStats {
    let started = Instant::now();
    let run_id = open_run(pool, "daily", "scheduled", Some(user_id)).await;
    let mut stats = RunStats {
        kind: "daily",
        ..Default::default()
    };

    // Working memories with an expires_at in the next 6 hours OR already
    // past — promote if they look high-signal, otherwise let TTL expire.
    let rows: Vec<(Uuid, f32, i32, Vec<String>)> =
        match sqlx::query_as::<_, (Uuid, f32, i32, Vec<String>)>(
            r#"
        SELECT id, importance, access_count, entities
        FROM memories
        WHERE user_id = $1
          AND type = 'working'
          AND superseded_by IS NULL
          AND expires_at IS NOT NULL
          AND expires_at < NOW() + INTERVAL '6 hours'
        ORDER BY expires_at ASC
        LIMIT 500
        "#,
        )
        .bind(user_id)
        .fetch_all(pool)
        .await
        {
            Ok(r) => r,
            Err(e) => {
                stats.error = Some(format!("query candidates: {e}"));
                close_run(pool, run_id, &stats, started).await;
                return stats;
            }
        };

    for (id, importance, access_count, entities) in rows {
        // Promotion rule: importance ≥ 0.6 OR (importance ≥ 0.4 AND been accessed ≥ 1)
        // OR has ≥ 2 entities (signal of "meaningful content").
        let promote =
            importance >= 0.6 || (importance >= 0.4 && access_count >= 1) || entities.len() >= 2;

        if promote {
            let result = sqlx::query(
                r#"
                UPDATE memories
                SET type = 'episodic',
                    decay_class = 'medium',
                    expires_at = NULL
                WHERE id = $1
                "#,
            )
            .bind(id)
            .execute(pool)
            .await;
            if result.is_ok() {
                stats.promoted_count += 1;
            }
        } else {
            // Expire immediately — TTL would have done this anyway.
            let _ = sqlx::query(
                "UPDATE memories SET expires_at = NOW() - INTERVAL '1 second' WHERE id = $1",
            )
            .bind(id)
            .execute(pool)
            .await;
            stats.expired_count += 1;
        }
    }

    close_run(pool, run_id, &stats, started).await;
    stats
}

/// Run the weekly job for a single user. Requires an AiProvider with the
/// fact_distillation capability; falls through to a no-op (with a logged
/// warning) when only heuristic is available.
pub async fn run_weekly(pool: &PgPool, nlp: &Arc<Nlp>, user_id: &str) -> RunStats {
    let started = Instant::now();
    let run_id = open_run(pool, "weekly", "scheduled", Some(user_id)).await;
    let mut stats = RunStats {
        kind: "weekly",
        ..Default::default()
    };

    if !nlp.provider_can_distill() {
        let note = format!(
            "skipped: provider `{}` does not implement fact_distillation",
            nlp.provider_name()
        );
        tracing::warn!("{note}");
        stats.error = Some(note);
        close_run(pool, run_id, &stats, started).await;
        return stats;
    }

    // Pull episodic candidates ≥ 7d old that haven't been consolidated yet.
    // "haven't been consolidated" = `superseded_by IS NULL` (a distilled
    // semantic memory will mark these as superseded by it).
    let candidates: Vec<EpisodeRow> = match sqlx::query_as::<_, EpisodeRow>(
        r#"
        SELECT id, content, entities,
               (extraction ->> 'topic') AS topic,
               project_id, session_id
        FROM memories
        WHERE user_id = $1
          AND type = 'episodic'
          AND superseded_by IS NULL
          AND created_at < NOW() - INTERVAL '7 days'
        ORDER BY created_at ASC
        LIMIT 500
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await
    {
        Ok(r) => r,
        Err(e) => {
            stats.error = Some(format!("query candidates: {e}"));
            close_run(pool, run_id, &stats, started).await;
            return stats;
        }
    };

    if candidates.len() < 2 {
        // Nothing to cluster.
        close_run(pool, run_id, &stats, started).await;
        return stats;
    }

    let clusters = cluster_by_topic_and_entities(&candidates);
    stats.clusters_seen = clusters.len() as i32;

    for cluster in clusters.iter().filter(|c| c.episodes.len() >= 2) {
        let episodes: Vec<EpisodeRef> = cluster
            .episodes
            .iter()
            .map(|e| EpisodeRef {
                id: e.id,
                content: e.content.clone(),
                topic: e.topic.clone(),
                entities: e.entities.clone(),
            })
            .collect();

        let facts = match nlp.provider().distill_facts(&episodes).await {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(
                    "distillation failed on cluster (topic={:?}): {e}",
                    cluster.dominant_topic
                );
                continue;
            }
        };

        for fact in facts {
            // Embed the fact's content for retrieval.
            let embedding = match nlp.embed_one(&fact.content).await {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("embedding distilled fact failed: {e}");
                    continue;
                }
            };
            let source_ids: Vec<Uuid> = fact
                .source_episode_ids
                .into_iter()
                .filter(|id| cluster.episodes.iter().any(|e| e.id == *id))
                .collect();
            if source_ids.is_empty() {
                continue;
            }

            let new_id = Uuid::new_v4();
            let topic_jsonb = serde_json::json!({
                "topic": fact.topic,
                "intent": "update",
                "operation": null,
                "entities": cluster.shared_entities,
                "action_target": null,
                "contradicts_hint": null,
                "confidence": fact.confidence,
                "_distilled_from": source_ids,
            });

            // Insert as semantic memory.
            let project_id = cluster.episodes.iter().find_map(|e| e.project_id.clone());
            let session_id: Option<String> = None; // semantic facts aren't session-scoped

            let insert = sqlx::query(
                r#"
                INSERT INTO memories (
                    id, type, content, embedding, importance, decay_class,
                    user_id, project_id, session_id, entities,
                    extraction
                ) VALUES (
                    $1, 'semantic', $2, $3, 0.7, 'slow',
                    $4, $5, $6, $7,
                    $8
                )
                "#,
            )
            .bind(new_id)
            .bind(&fact.content)
            .bind(Vector::from(embedding))
            .bind(user_id)
            .bind(&project_id)
            .bind(&session_id)
            .bind(&cluster.shared_entities)
            .bind(&topic_jsonb)
            .execute(pool)
            .await;

            if let Err(e) = insert {
                tracing::warn!("insert distilled fact failed: {e}");
                continue;
            }

            // Mark source episodes as superseded by the new semantic fact.
            // Each episode points its supersedes pointer at the previous one in the
            // chain isn't right here — we want a fan-in. Use superseded_by on each
            // source row pointing at the new semantic id. (`supersedes` stays NULL
            // because there are multiple sources; lineage walks via the JSONB
            // `_distilled_from` array.)
            for sid in &source_ids {
                let _ = sqlx::query(
                    "UPDATE memories SET superseded_by = $1 WHERE id = $2 AND superseded_by IS NULL",
                )
                .bind(new_id)
                .bind(sid)
                .execute(pool)
                .await;
            }

            stats.distilled_count += 1;
        }
    }

    close_run(pool, run_id, &stats, started).await;
    stats
}

// ---------------------------------------------------------------------------
// Clustering — entity-Jaccard + shared-topic grouping. Cheap, O(N²) at most
// N=500 candidates. Phase 5 may swap this for embedding-based clustering.
// ---------------------------------------------------------------------------

#[derive(Debug, sqlx::FromRow)]
struct EpisodeRow {
    id: Uuid,
    content: String,
    entities: Vec<String>,
    topic: Option<String>,
    project_id: Option<String>,
    #[allow(dead_code)]
    session_id: Option<String>,
}

#[derive(Debug)]
struct Cluster {
    episodes: Vec<EpisodeRow>,
    dominant_topic: Option<String>,
    shared_entities: Vec<String>,
}

fn cluster_by_topic_and_entities(rows: &[EpisodeRow]) -> Vec<Cluster> {
    // First pass: group by exact-match topic (cheap, high precision).
    let mut by_topic: HashMap<String, Vec<usize>> = HashMap::new();
    let mut leftover: Vec<usize> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        match r.topic.as_deref().map(|t| t.trim().to_lowercase()) {
            Some(t) if !t.is_empty() => by_topic.entry(t).or_default().push(i),
            _ => leftover.push(i),
        }
    }

    let mut clusters: Vec<Cluster> = Vec::new();
    for (topic, idxs) in by_topic {
        if idxs.len() >= 2 {
            let episodes: Vec<EpisodeRow> = idxs.iter().map(|&i| clone_row(&rows[i])).collect();
            let shared = shared_entities(&episodes);
            clusters.push(Cluster {
                episodes,
                dominant_topic: Some(topic),
                shared_entities: shared,
            });
        } else {
            // singleton topic-bucket — try to merge into leftover entity-Jaccard step
            leftover.extend(idxs);
        }
    }

    // Second pass: entity-Jaccard greedy clustering on leftovers.
    // For each unassigned row, find the row with highest Jaccard ≥ 0.4 and merge.
    let mut assigned = vec![false; rows.len()];
    for &i in &leftover {
        if assigned[i] {
            continue;
        }
        let mut group: Vec<usize> = vec![i];
        assigned[i] = true;
        for &j in &leftover {
            if assigned[j] || i == j {
                continue;
            }
            if jaccard_str(&rows[i].entities, &rows[j].entities) >= 0.4 {
                group.push(j);
                assigned[j] = true;
            }
        }
        if group.len() >= 2 {
            let episodes: Vec<EpisodeRow> = group.iter().map(|&k| clone_row(&rows[k])).collect();
            let shared = shared_entities(&episodes);
            clusters.push(Cluster {
                episodes,
                dominant_topic: None,
                shared_entities: shared,
            });
        }
    }

    clusters
}

fn clone_row(r: &EpisodeRow) -> EpisodeRow {
    EpisodeRow {
        id: r.id,
        content: r.content.clone(),
        entities: r.entities.clone(),
        topic: r.topic.clone(),
        project_id: r.project_id.clone(),
        session_id: r.session_id.clone(),
    }
}

fn shared_entities(rows: &[EpisodeRow]) -> Vec<String> {
    if rows.is_empty() {
        return Vec::new();
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for r in rows {
        let unique: std::collections::HashSet<&str> =
            r.entities.iter().map(|s| s.as_str()).collect();
        for e in unique {
            *counts.entry(e.to_string()).or_insert(0) += 1;
        }
    }
    // Keep entities that appear in ≥ half the rows.
    let threshold = (rows.len() as f32 * 0.5).ceil() as usize;
    counts
        .into_iter()
        .filter(|(_, c)| *c >= threshold)
        .map(|(s, _)| s)
        .collect()
}

fn jaccard_str(a: &[String], b: &[String]) -> f32 {
    let sa: std::collections::HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let sb: std::collections::HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

// ---------------------------------------------------------------------------
// Run audit log
// ---------------------------------------------------------------------------

async fn open_run(pool: &PgPool, kind: &str, trigger: &str, user_id: Option<&str>) -> Option<Uuid> {
    let id = Uuid::new_v4();
    let result = sqlx::query(
        r#"INSERT INTO consolidation_runs (id, kind, trigger, user_id) VALUES ($1, $2, $3, $4)"#,
    )
    .bind(id)
    .bind(kind)
    .bind(trigger)
    .bind(user_id)
    .execute(pool)
    .await;
    match result {
        Ok(_) => Some(id),
        Err(e) => {
            tracing::warn!("open_run insert failed: {e}");
            None
        }
    }
}

async fn close_run(pool: &PgPool, id: Option<Uuid>, stats: &RunStats, started: Instant) {
    let Some(id) = id else { return };
    let _ = sqlx::query(
        r#"
        UPDATE consolidation_runs
        SET finished_at = NOW(),
            promoted_count = $2,
            expired_count = $3,
            distilled_count = $4,
            clusters_seen = $5,
            error = $6
        WHERE id = $1
        "#,
    )
    .bind(id)
    .bind(stats.promoted_count)
    .bind(stats.expired_count)
    .bind(stats.distilled_count)
    .bind(stats.clusters_seen)
    .bind(&stats.error)
    .execute(pool)
    .await;
    tracing::info!(
        kind = stats.kind,
        promoted = stats.promoted_count,
        expired = stats.expired_count,
        distilled = stats.distilled_count,
        clusters = stats.clusters_seen,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "consolidation run complete"
    );
}

// ---------------------------------------------------------------------------
// Cross-user runner — iterates each distinct user_id in `tokens` and runs
// the job for each. Background scheduler uses this; the manual trigger
// scopes to a specific user from the request.
// ---------------------------------------------------------------------------

pub async fn run_daily_all_users(pool: &PgPool) -> Vec<RunStats> {
    let users = list_users(pool).await;
    let mut out = Vec::new();
    for u in users {
        out.push(run_daily(pool, &u).await);
    }
    out
}

pub async fn run_weekly_all_users(pool: &PgPool, nlp: &Arc<Nlp>) -> Vec<RunStats> {
    let users = list_users(pool).await;
    let mut out = Vec::new();
    for u in users {
        out.push(run_weekly(pool, nlp, &u).await);
    }
    out
}

async fn list_users(pool: &PgPool) -> Vec<String> {
    sqlx::query_scalar::<_, String>("SELECT DISTINCT user_id FROM memories ORDER BY user_id")
        .fetch_all(pool)
        .await
        .unwrap_or_default()
}

// Avoid the unused `Utc` warning until we add scheduling timestamp fields.
#[allow(dead_code)]
fn _utc() -> chrono::DateTime<chrono::Utc> {
    Utc::now()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(topic: Option<&str>, entities: &[&str]) -> EpisodeRow {
        EpisodeRow {
            id: Uuid::new_v4(),
            content: String::new(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            topic: topic.map(|s| s.to_string()),
            project_id: None,
            session_id: None,
        }
    }

    // ---- Kind --------------------------------------------------------------

    #[test]
    fn kind_as_str_round_trip() {
        assert_eq!(Kind::Daily.as_str(), "daily");
        assert_eq!(Kind::Weekly.as_str(), "weekly");
    }

    // ---- jaccard_str -------------------------------------------------------

    #[test]
    fn jaccard_disjoint_sets_is_zero() {
        let a = ["x".into(), "y".into()];
        let b = ["p".into(), "q".into()];
        assert_eq!(jaccard_str(&a, &b), 0.0);
    }

    #[test]
    fn jaccard_identical_sets_is_one() {
        let a = ["x".into(), "y".into()];
        assert!((jaccard_str(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_both_empty_is_zero() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(jaccard_str(&empty, &empty), 0.0);
    }

    #[test]
    fn jaccard_partial_overlap_known_value() {
        // A ∩ B = {y}, A ∪ B = {x, y, z} → 1/3.
        let a = ["x".into(), "y".into()];
        let b = ["y".into(), "z".into()];
        assert!((jaccard_str(&a, &b) - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn jaccard_dedupes_within_input() {
        // Same string twice in `a` shouldn't inflate either side.
        let a = ["x".into(), "x".into(), "y".into()];
        let b = ["x".into()];
        // sa={x,y}, sb={x} → intersection={x}, union={x,y} → 1/2
        assert!((jaccard_str(&a, &b) - 0.5).abs() < 1e-6);
    }

    // ---- shared_entities ---------------------------------------------------

    #[test]
    fn shared_entities_empty_input_returns_empty() {
        let out = shared_entities(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn shared_entities_majority_threshold() {
        // 3 rows, threshold = ceil(3 * 0.5) = 2.
        // "common" appears in rows 0 and 1 (count=2 ≥ 2) — keep.
        // "rare" appears in row 0 only (count=1 < 2) — drop.
        let rows = vec![
            row(None, &["common", "rare"]),
            row(None, &["common"]),
            row(None, &["other"]),
        ];
        let out = shared_entities(&rows);
        assert!(out.contains(&"common".to_string()));
        assert!(!out.contains(&"rare".to_string()));
        assert!(!out.contains(&"other".to_string()));
    }

    #[test]
    fn shared_entities_dedupes_within_row() {
        // Same entity twice in one row should count once for that row.
        let rows = vec![row(None, &["x", "x", "x"]), row(None, &["y"])];
        let out = shared_entities(&rows);
        // x appears in 1 of 2 rows; threshold = ceil(2 * 0.5) = 1 → kept.
        assert!(out.contains(&"x".to_string()));
        assert!(out.contains(&"y".to_string()));
    }

    // ---- clone_row ---------------------------------------------------------

    #[test]
    fn clone_row_preserves_all_fields() {
        let original = row(Some("dep target"), &["postgres", "deploy"]);
        let cloned = clone_row(&original);
        assert_eq!(cloned.id, original.id);
        assert_eq!(cloned.content, original.content);
        assert_eq!(cloned.entities, original.entities);
        assert_eq!(cloned.topic, original.topic);
        assert_eq!(cloned.project_id, original.project_id);
        assert_eq!(cloned.session_id, original.session_id);
    }

    // ---- cluster_by_topic_and_entities ------------------------------------

    #[test]
    fn clustering_empty_input_returns_no_clusters() {
        assert!(cluster_by_topic_and_entities(&[]).is_empty());
    }

    #[test]
    fn clustering_groups_by_exact_topic_match() {
        // Three rows: two share topic "deploy", one has a different topic.
        // The two-row topic forms one cluster; the singleton doesn't.
        let rows = vec![
            row(Some("deploy"), &["postgres"]),
            row(Some("deploy"), &["docker"]),
            row(Some("auth"), &["bcrypt"]),
        ];
        let clusters = cluster_by_topic_and_entities(&rows);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].dominant_topic.as_deref(), Some("deploy"));
        assert_eq!(clusters[0].episodes.len(), 2);
    }

    #[test]
    fn clustering_normalizes_topic_case() {
        // Topics differ only by case + whitespace; should still merge.
        let rows = vec![
            row(Some("Deploy"), &[]),
            row(Some("  deploy "), &[]),
            row(Some("DEPLOY"), &[]),
        ];
        let clusters = cluster_by_topic_and_entities(&rows);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].episodes.len(), 3);
    }

    #[test]
    fn clustering_falls_back_to_entity_jaccard_for_topicless_rows() {
        // No topics, but shared entities pass jaccard ≥ 0.4.
        // Two rows with [a, b] and [a, b, c] → jaccard = 2/3 ≈ 0.67 → cluster.
        let rows = vec![row(None, &["a", "b"]), row(None, &["a", "b", "c"])];
        let clusters = cluster_by_topic_and_entities(&rows);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].dominant_topic.is_none());
        assert_eq!(clusters[0].episodes.len(), 2);
    }

    #[test]
    fn clustering_singleton_topics_dont_become_clusters() {
        // Each row has a unique topic and no entity overlap → no clusters.
        let rows = vec![
            row(Some("auth"), &["a"]),
            row(Some("deploy"), &["b"]),
            row(Some("ingest"), &["c"]),
        ];
        let clusters = cluster_by_topic_and_entities(&rows);
        assert!(clusters.is_empty());
    }

    #[test]
    fn clustering_singleton_topics_can_join_entity_clusters() {
        // Two rows with unique topics but high entity overlap should still
        // form an entity-Jaccard cluster after the singleton-topic fallback.
        let rows = vec![
            row(Some("topicA"), &["x", "y", "z"]),
            row(Some("topicB"), &["x", "y", "z"]),
        ];
        let clusters = cluster_by_topic_and_entities(&rows);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].dominant_topic.is_none()); // entity-jaccard branch
    }

    // ---- integration tests: run_daily against a real DB ------------------

    use chrono::Duration;

    /// Insert a working memory with a custom expires_at + entity count + importance,
    /// returning its id. Embedding is a zero vector since run_daily doesn't read it.
    async fn insert_working(
        pool: &sqlx::PgPool,
        user_id: &str,
        importance: f32,
        access_count: i32,
        entities: &[&str],
        expires_in_hours: i64,
    ) -> Uuid {
        let id = Uuid::new_v4();
        let ents: Vec<String> = entities.iter().map(|s| s.to_string()).collect();
        let emb = pgvector::Vector::from(vec![0.0_f32; 384]);
        let expires_at = chrono::Utc::now() + Duration::hours(expires_in_hours);
        sqlx::query(
            r#"INSERT INTO memories
               (id, type, content, embedding, importance, access_count, decay_class,
                user_id, entities, expires_at)
               VALUES ($1, 'working', $2, $3, $4, $5, 'fast', $6, $7, $8)"#,
        )
        .bind(id)
        .bind(format!("working memory {id}"))
        .bind(emb)
        .bind(importance)
        .bind(access_count)
        .bind(user_id)
        .bind(&ents)
        .bind(expires_at)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    async fn fetch_type(pool: &sqlx::PgPool, id: Uuid) -> String {
        sqlx::query_scalar("SELECT type FROM memories WHERE id = $1")
            .bind(id)
            .fetch_one(pool)
            .await
            .unwrap()
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_daily_no_op_when_nothing_near_expiry(pool: sqlx::PgPool) {
        // Inserted far past the 6-hour window — should be ignored.
        insert_working(&pool, "alice", 0.9, 5, &["a", "b"], 48).await;

        let stats = run_daily(&pool, "alice").await;
        assert_eq!(stats.promoted_count, 0);
        assert_eq!(stats.expired_count, 0);
        assert!(stats.error.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_daily_promotes_high_importance_to_episodic(pool: sqlx::PgPool) {
        // importance >= 0.6 → promote.
        let id = insert_working(&pool, "alice", 0.8, 0, &["important"], 1).await;
        let stats = run_daily(&pool, "alice").await;
        assert_eq!(stats.promoted_count, 1);
        assert_eq!(stats.expired_count, 0);
        assert_eq!(fetch_type(&pool, id).await, "episodic");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_daily_promotes_accessed_medium_importance(pool: sqlx::PgPool) {
        // importance >= 0.4 AND access_count >= 1 → promote.
        let id = insert_working(&pool, "alice", 0.5, 2, &["a"], 1).await;
        let stats = run_daily(&pool, "alice").await;
        assert_eq!(stats.promoted_count, 1);
        assert_eq!(fetch_type(&pool, id).await, "episodic");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_daily_promotes_when_two_or_more_entities(pool: sqlx::PgPool) {
        // Below importance/access thresholds, but >= 2 entities still triggers.
        let id = insert_working(&pool, "alice", 0.1, 0, &["x", "y", "z"], 1).await;
        let stats = run_daily(&pool, "alice").await;
        assert_eq!(stats.promoted_count, 1);
        assert_eq!(fetch_type(&pool, id).await, "episodic");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_daily_expires_low_signal_memories(pool: sqlx::PgPool) {
        // Low importance, no accesses, fewer than 2 entities → expire.
        let id = insert_working(&pool, "alice", 0.1, 0, &["just_one"], 1).await;
        let stats = run_daily(&pool, "alice").await;
        assert_eq!(stats.promoted_count, 0);
        assert_eq!(stats.expired_count, 1);
        // Type stayed working but expires_at moved into the past.
        assert_eq!(fetch_type(&pool, id).await, "working");
        let expires_at: chrono::DateTime<chrono::Utc> =
            sqlx::query_scalar("SELECT expires_at FROM memories WHERE id = $1")
                .bind(id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(expires_at < chrono::Utc::now());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_daily_scopes_to_user_id(pool: sqlx::PgPool) {
        let alice = insert_working(&pool, "alice", 0.9, 0, &[], 1).await;
        let bob = insert_working(&pool, "bob", 0.9, 0, &[], 1).await;

        let stats = run_daily(&pool, "alice").await;
        assert_eq!(stats.promoted_count, 1);
        // Bob's memory untouched.
        assert_eq!(fetch_type(&pool, alice).await, "episodic");
        assert_eq!(fetch_type(&pool, bob).await, "working");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn run_daily_writes_audit_row(pool: sqlx::PgPool) {
        insert_working(&pool, "alice", 0.9, 0, &[], 1).await;
        run_daily(&pool, "alice").await;

        let count: i64 = sqlx::query_scalar(
            r#"SELECT COUNT(*) FROM consolidation_runs
               WHERE kind = 'daily' AND user_id = $1 AND finished_at IS NOT NULL"#,
        )
        .bind("alice")
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(count, 1);
    }
}
