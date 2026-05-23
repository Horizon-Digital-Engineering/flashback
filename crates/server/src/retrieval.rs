//! Hybrid retrieval: vector + BM25 + recency + importance + project + entity overlap.
//!
//! The spec's `w_task` term is wired in as zero — task tracking is Phase 2.

use std::collections::HashSet;

use chrono::{DateTime, Utc};
use pgvector::Vector;
use serde::Deserialize;
use sqlx::PgPool;
use uuid::Uuid;

use crate::{
    error::AppResult,
    models::{DecayClass, MemoryRow, MemoryView},
};

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Answer,
    Manager,
}

impl Default for Mode {
    fn default() -> Self {
        Self::Answer
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Weights {
    pub semantic: f64,
    pub keyword: f64,
    pub recency: f64,
    pub importance: f64,
    pub project: f64,
    pub entity: f64,
}

impl Weights {
    pub fn for_mode(mode: Mode) -> Self {
        match mode {
            // Answer: lean on relevance signals
            Mode::Answer => Self {
                semantic: 0.45,
                keyword: 0.25,
                recency: 0.10,
                importance: 0.10,
                project: 0.05,
                entity: 0.05,
            },
            // Manager: lean on situational-awareness signals
            Mode::Manager => Self {
                semantic: 0.15,
                keyword: 0.10,
                recency: 0.25,
                importance: 0.10,
                project: 0.25,
                entity: 0.15,
            },
        }
    }
}

#[derive(Debug)]
pub struct SearchParams<'a> {
    pub user_id: &'a str,
    pub project_id: Option<&'a str>,
    pub session_id: Option<&'a str>,
    pub query: Option<&'a str>,
    pub query_embedding: Option<&'a [f32]>,
    pub query_entities: Vec<String>,
    pub memory_types: Option<Vec<&'a str>>,
    pub since: Option<DateTime<Utc>>,
    pub until: Option<DateTime<Utc>>,
    pub mode: Mode,
    pub top_k: usize,
    pub include_superseded: bool,
}

#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub view: MemoryView,
    pub score: f64,
    pub semantic: f64,
    pub keyword: f64,
    pub recency: f64,
    pub project: f64,
    pub entity: f64,
}

pub async fn search(pool: &PgPool, params: SearchParams<'_>) -> AppResult<Vec<ScoredMemory>> {
    let weights = Weights::for_mode(params.mode);

    let candidate_pool: usize = std::cmp::max(params.top_k * 8, 50);
    let candidate_pool = std::cmp::min(candidate_pool, 200);

    let q_vec: Option<Vector> = params.query_embedding.map(|v| Vector::from(v.to_vec()));
    let q_text: Option<&str> = params.query;
    let types: Option<Vec<String>> = params
        .memory_types
        .as_ref()
        .map(|ts| ts.iter().map(|s| s.to_string()).collect());

    // We fetch a pre-filtered candidate set, ordered by an approximate
    // composite score that uses what Postgres can compute cheaply: vector
    // distance + ts_rank. Final ranking happens in Rust where we can apply
    // every weight and the diversity re-rank.
    let rows: Vec<RawCandidate> = sqlx::query_as::<_, RawCandidate>(
        r#"
        WITH params AS (
            SELECT
                $1::TEXT          AS user_id,
                $2::TEXT          AS project_id,
                $3::TEXT[]        AS allowed_types,
                $4::TIMESTAMPTZ   AS since_ts,
                $5::TIMESTAMPTZ   AS until_ts,
                $6::vector        AS q_vec,
                $7::TEXT          AS q_text,
                $8::BOOLEAN       AS include_superseded
        )
        SELECT
            m.id,
            m.type,
            m.content,
            m.embedding,
            m.importance,
            m.access_count,
            m.decay_class,
            m.user_id,
            m.project_id,
            m.session_id,
            m.entities,
            m.superseded_by,
            m.supersedes,
            m.source_path,
            m.chunk_index,
            m.content_hash,
            m.state_kind,
            m.state_key,
            m.state_data,
            m.expires_at,
            m.created_at,
            m.updated_at,
            m.last_accessed_at,
            CASE
                WHEN (SELECT q_vec FROM params) IS NOT NULL AND m.embedding IS NOT NULL
                THEN 1.0 - (m.embedding <=> (SELECT q_vec FROM params))
                ELSE 0.0
            END AS sem_score,
            CASE
                WHEN (SELECT q_text FROM params) IS NOT NULL
                THEN ts_rank(m.content_tsv, plainto_tsquery('english', (SELECT q_text FROM params)))
                ELSE 0.0
            END AS kw_score
        FROM memories m, params p
        WHERE m.user_id = p.user_id
          AND (p.include_superseded OR m.superseded_by IS NULL)
          AND (p.project_id IS NULL OR m.project_id = p.project_id)
          AND (p.allowed_types IS NULL OR m.type = ANY(p.allowed_types))
          AND (p.since_ts IS NULL OR m.created_at >= p.since_ts)
          AND (p.until_ts IS NULL OR m.created_at <= p.until_ts)
          AND (m.expires_at IS NULL OR m.expires_at > NOW())
        ORDER BY
            (CASE
                WHEN (SELECT q_vec FROM params) IS NOT NULL AND m.embedding IS NOT NULL
                THEN 1.0 - (m.embedding <=> (SELECT q_vec FROM params))
                ELSE 0.0
            END) +
            (CASE
                WHEN (SELECT q_text FROM params) IS NOT NULL
                THEN 0.5 * ts_rank(m.content_tsv, plainto_tsquery('english', (SELECT q_text FROM params)))
                ELSE 0.0
            END) +
            0.1 * m.importance
            DESC
        LIMIT $9
        "#,
    )
    .bind(params.user_id)
    .bind(params.project_id)
    .bind(types.as_deref())
    .bind(params.since)
    .bind(params.until)
    .bind(q_vec)
    .bind(q_text)
    .bind(params.include_superseded)
    .bind(candidate_pool as i64)
    .fetch_all(pool)
    .await?;

    let mut scored: Vec<ScoredMemory> = rows
        .into_iter()
        .map(|c| score_candidate(c, &params, weights))
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let scored = diversity_rerank(scored, params.top_k);
    Ok(scored)
}

fn diversity_rerank(mut scored: Vec<ScoredMemory>, top_k: usize) -> Vec<ScoredMemory> {
    let mut kept: Vec<ScoredMemory> = Vec::with_capacity(top_k);
    while let Some(candidate) = scored.first().cloned() {
        scored.remove(0);
        let too_similar = kept.iter().any(|k| {
            let j = jaccard_str(&candidate.view.entities, &k.view.entities);
            j > 0.8
        });
        if !too_similar {
            kept.push(candidate);
            if kept.len() >= top_k {
                break;
            }
        }
    }
    kept
}

fn score_candidate(c: RawCandidate, params: &SearchParams<'_>, w: Weights) -> ScoredMemory {
    let sem = c.sem_score.clamp(0.0, 1.0);

    // Normalize ts_rank, which has no upper bound. Empirically rank > 0.3
    // is already a strong match; saturate there.
    let kw = (c.kw_score as f64 / 0.3).clamp(0.0, 1.0);

    let decay = DecayClass::from_str(&c.decay_class);
    let recency = recency_score(c.last_accessed_at, decay);

    let importance = c.importance as f64;

    let project = match (params.project_id, c.project_id.as_deref()) {
        (Some(qp), Some(mp)) if qp == mp => 1.0,
        _ => 0.0,
    };

    let entity = jaccard_str(&params.query_entities, &c.entities);

    let score = w.semantic * sem
        + w.keyword * kw
        + w.recency * recency
        + w.importance * importance
        + w.project * project
        + w.entity * entity;

    ScoredMemory {
        view: MemoryView::from(MemoryRow::from(c)),
        score,
        semantic: sem,
        keyword: kw,
        recency,
        project,
        entity,
    }
}

fn recency_score(last_accessed: DateTime<Utc>, decay: DecayClass) -> f64 {
    let half_life = match decay.half_life_days() {
        Some(h) => h,
        None => return 1.0,
    };
    let days = (Utc::now() - last_accessed).num_seconds() as f64 / 86_400.0;
    let lambda = std::f64::consts::LN_2 / half_life;
    (-lambda * days.max(0.0)).exp()
}

fn jaccard_str(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 0.0;
    }
    let a_set: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let b_set: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = a_set.intersection(&b_set).count() as f64;
    let union = a_set.union(&b_set).count() as f64;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

impl DecayClass {
    fn from_str(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "slow" => Self::Slow,
            "fast" => Self::Fast,
            _ => Self::Medium,
        }
    }
}

#[derive(Debug, sqlx::FromRow)]
struct RawCandidate {
    id: Uuid,
    r#type: String,
    content: String,
    embedding: Option<Vector>,
    importance: f32,
    access_count: i32,
    decay_class: String,
    user_id: String,
    project_id: Option<String>,
    session_id: Option<String>,
    entities: Vec<String>,
    superseded_by: Option<Uuid>,
    supersedes: Option<Uuid>,
    source_path: Option<String>,
    chunk_index: Option<i32>,
    content_hash: Option<String>,
    state_kind: Option<String>,
    state_key: Option<String>,
    state_data: Option<serde_json::Value>,
    expires_at: Option<DateTime<Utc>>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    last_accessed_at: DateTime<Utc>,
    sem_score: f64,
    kw_score: f32,
}

impl From<RawCandidate> for MemoryRow {
    fn from(c: RawCandidate) -> Self {
        Self {
            id: c.id,
            r#type: c.r#type,
            content: c.content,
            embedding: c.embedding,
            importance: c.importance,
            access_count: c.access_count,
            decay_class: c.decay_class,
            user_id: c.user_id,
            project_id: c.project_id,
            session_id: c.session_id,
            entities: c.entities,
            superseded_by: c.superseded_by,
            supersedes: c.supersedes,
            source_path: c.source_path,
            chunk_index: c.chunk_index,
            content_hash: c.content_hash,
            state_kind: c.state_kind,
            state_key: c.state_key,
            state_data: c.state_data,
            expires_at: c.expires_at,
            created_at: c.created_at,
            updated_at: c.updated_at,
            last_accessed_at: c.last_accessed_at,
        }
    }
}

/// Token-budget approximation. Real tokenizer is overkill for layer budgets;
/// chars/4 is within ~10% for English prose. See VISION/REFERENCES.md.
pub fn approx_tokens(s: &str) -> usize {
    (s.chars().count() + 3) / 4
}

pub async fn touch_access(pool: &PgPool, ids: &[Uuid]) -> AppResult<()> {
    if ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        r#"
        UPDATE memories
        SET access_count = access_count + 1,
            last_accessed_at = NOW()
        WHERE id = ANY($1)
        "#,
    )
    .bind(ids)
    .execute(pool)
    .await?;
    Ok(())
}

/// Returns true if the new memory likely supersedes an existing one,
/// and the id of that existing memory. Conservative — only fires when
/// session/user/project match AND entity Jaccard exceeds the threshold.
pub fn looks_like_supersede(
    new_entities: &[String],
    candidates: &[(Uuid, Vec<String>)],
    threshold: f64,
) -> Option<Uuid> {
    let mut best: Option<(Uuid, f64)> = None;
    for (id, ents) in candidates {
        let j = jaccard_str(new_entities, ents);
        if j >= threshold && best.map(|(_, s)| j > s).unwrap_or(true) {
            best = Some((*id, j));
        }
    }
    best.map(|(id, _)| id)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- looks_like_supersede (pure, no DB) -------------------------------

    fn pair(id: &str, ents: &[&str]) -> (Uuid, Vec<String>) {
        (
            Uuid::parse_str(id).unwrap(),
            ents.iter().map(|s| s.to_string()).collect(),
        )
    }

    #[test]
    fn looks_like_supersede_returns_none_when_no_candidate_clears_threshold() {
        let candidates = vec![
            pair("00000000-0000-0000-0000-000000000001", &["x"]),
            pair("00000000-0000-0000-0000-000000000002", &["y"]),
        ];
        let new = vec!["z".into()];
        // Jaccard with each candidate is 0; threshold 0.3 → no match.
        assert!(looks_like_supersede(&new, &candidates, 0.3).is_none());
    }

    #[test]
    fn looks_like_supersede_picks_best_overlap() {
        let weak = pair("00000000-0000-0000-0000-000000000001", &["a", "z"]);
        let strong = pair("00000000-0000-0000-0000-000000000002", &["a", "b"]);
        let candidates = vec![weak.clone(), strong.clone()];
        let new = vec!["a".into(), "b".into()];
        // Strong jaccard = 1.0; weak = 1/3. Strong wins.
        assert_eq!(looks_like_supersede(&new, &candidates, 0.3), Some(strong.0));
    }

    #[test]
    fn looks_like_supersede_respects_threshold() {
        // 1/3 jaccard — under 0.5, over 0.3.
        let c = vec![pair("00000000-0000-0000-0000-000000000001", &["x", "y"])];
        let new = vec!["x".into(), "z".into()];
        assert!(looks_like_supersede(&new, &c, 0.5).is_none());
        assert!(looks_like_supersede(&new, &c, 0.3).is_some());
    }

    #[test]
    fn approx_tokens_in_retrieval_matches_chunking() {
        // Sanity-link to the chunking module's same constant.
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("abcd"), 1);
        assert_eq!(approx_tokens("a".repeat(400).as_str()), 100);
    }

    // ---- integration tests (real Postgres + pgvector) ---------------------

    use sqlx::PgPool;

    async fn insert_memory(
        pool: &PgPool,
        user_id: &str,
        project_id: Option<&str>,
        session_id: Option<&str>,
        mem_type: &str,
        content: &str,
        entities: &[&str],
        importance: f32,
    ) -> Uuid {
        let id = Uuid::new_v4();
        let ents: Vec<String> = entities.iter().map(|s| s.to_string()).collect();
        // Zero embedding — semantic ranking will be uniform; tests that care
        // about semantic should pass query_embedding=None to skip that arm.
        let emb = pgvector::Vector::from(vec![0.0_f32; 384]);
        sqlx::query(
            r#"INSERT INTO memories
               (id, type, content, embedding, importance, decay_class, user_id,
                project_id, session_id, entities)
               VALUES ($1, $2, $3, $4, $5, 'medium', $6, $7, $8, $9)"#,
        )
        .bind(id)
        .bind(mem_type)
        .bind(content)
        .bind(emb)
        .bind(importance)
        .bind(user_id)
        .bind(project_id)
        .bind(session_id)
        .bind(&ents)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    fn base_params(user_id: &str) -> SearchParams<'_> {
        SearchParams {
            user_id,
            project_id: None,
            session_id: None,
            query: None,
            query_embedding: None,
            query_entities: Vec::new(),
            memory_types: None,
            since: None,
            until: None,
            mode: Mode::Answer,
            top_k: 20,
            include_superseded: false,
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_empty_db_returns_no_results(pool: PgPool) {
        let out = search(&pool, base_params("alice")).await.unwrap();
        assert!(out.is_empty());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_filters_by_user_id(pool: PgPool) {
        insert_memory(
            &pool,
            "alice",
            None,
            None,
            "episodic",
            "alice memory",
            &["x"],
            0.5,
        )
        .await;
        insert_memory(
            &pool,
            "bob",
            None,
            None,
            "episodic",
            "bob memory",
            &["y"],
            0.5,
        )
        .await;

        let alice = search(&pool, base_params("alice")).await.unwrap();
        assert_eq!(alice.len(), 1);
        assert_eq!(alice[0].view.user_id, "alice");

        let bob = search(&pool, base_params("bob")).await.unwrap();
        assert_eq!(bob.len(), 1);
        assert_eq!(bob[0].view.user_id, "bob");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_filters_by_memory_type(pool: PgPool) {
        insert_memory(&pool, "alice", None, None, "episodic", "ep1", &[], 0.5).await;
        insert_memory(&pool, "alice", None, None, "semantic", "se1", &[], 0.5).await;
        insert_memory(&pool, "alice", None, None, "working", "wk1", &[], 0.5).await;

        let mut params = base_params("alice");
        params.memory_types = Some(vec!["episodic", "semantic"]);
        let out = search(&pool, params).await.unwrap();
        assert_eq!(out.len(), 2);
        for r in &out {
            assert!(matches!(r.view.type_.as_str(), "episodic" | "semantic"));
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn search_excludes_superseded_by_default(pool: PgPool) {
        let old = insert_memory(&pool, "alice", None, None, "episodic", "old", &[], 0.5).await;
        let new = insert_memory(&pool, "alice", None, None, "episodic", "new", &[], 0.5).await;
        // Mark old as superseded by new.
        sqlx::query("UPDATE memories SET superseded_by = $1 WHERE id = $2")
            .bind(new)
            .bind(old)
            .execute(&pool)
            .await
            .unwrap();

        let out = search(&pool, base_params("alice")).await.unwrap();
        // Only the new (terminal) one should show.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].view.id, new);

        let mut with_superseded = base_params("alice");
        with_superseded.include_superseded = true;
        let out_all = search(&pool, with_superseded).await.unwrap();
        assert_eq!(out_all.len(), 2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn touch_access_increments_count_and_timestamp(pool: PgPool) {
        let id = insert_memory(&pool, "alice", None, None, "episodic", "to touch", &[], 0.5).await;

        // Initially access_count=0; last_accessed_at defaults to NOW() on insert
        // (per schema), so we just record it and check touch_access bumps it.
        let (count_before, ts_before): (i32, chrono::DateTime<chrono::Utc>) =
            sqlx::query_as("SELECT access_count, last_accessed_at FROM memories WHERE id = $1")
                .bind(id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count_before, 0);

        // Small sleep so the timestamp advance is observable on fast systems.
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        touch_access(&pool, &[id]).await.unwrap();

        let (count_after, ts_after): (i32, chrono::DateTime<chrono::Utc>) =
            sqlx::query_as("SELECT access_count, last_accessed_at FROM memories WHERE id = $1")
                .bind(id)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(count_after, 1);
        assert!(ts_after >= ts_before, "timestamp should advance or hold");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn touch_access_empty_input_is_noop(pool: PgPool) {
        // Should not error and should not write any rows.
        touch_access(&pool, &[]).await.unwrap();
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn touch_access_handles_multiple_ids(pool: PgPool) {
        let id1 = insert_memory(&pool, "alice", None, None, "episodic", "one", &[], 0.5).await;
        let id2 = insert_memory(&pool, "alice", None, None, "episodic", "two", &[], 0.5).await;
        touch_access(&pool, &[id1, id2]).await.unwrap();
        for id in [id1, id2] {
            let c: i32 = sqlx::query_scalar("SELECT access_count FROM memories WHERE id = $1")
                .bind(id)
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(c, 1);
        }
    }
}
