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
    let rows: Vec<(Uuid, f32, i32, Vec<String>)> = match sqlx::query_as::<_, (Uuid, f32, i32, Vec<String>)>(
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
        let promote = importance >= 0.6
            || (importance >= 0.4 && access_count >= 1)
            || entities.len() >= 2;

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
            let project_id = cluster
                .episodes
                .iter()
                .find_map(|e| e.project_id.clone());
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
            let episodes: Vec<EpisodeRow> = idxs
                .iter()
                .map(|&i| clone_row(&rows[i]))
                .collect();
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
        let unique: std::collections::HashSet<&str> = r.entities.iter().map(|s| s.as_str()).collect();
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
    sqlx::query_scalar::<_, String>(
        "SELECT DISTINCT user_id FROM memories ORDER BY user_id",
    )
    .fetch_all(pool)
    .await
    .unwrap_or_default()
}

// Avoid the unused `Utc` warning until we add scheduling timestamp fields.
#[allow(dead_code)]
fn _utc() -> chrono::DateTime<chrono::Utc> {
    Utc::now()
}
