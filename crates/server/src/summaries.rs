//! RAPTOR-style hierarchical summaries over the curated layer.
//!
//! The curation pipeline builds the level-0 curated nodes (episodic +
//! semantic). This module builds the tree *above* them: cluster the nodes at a
//! level, summarize each
//! cluster into one node at `level + 1` (`kind='summary'`), link it to every
//! child with a `summarizes` edge, embed it, and recurse — until a level has
//! few enough nodes (`SUMMARY_FANOUT_STOP`) or the depth cap (`MAX_LEVELS`) is
//! hit. This gives a "summarize first, drill to raw on demand" feed: retrieval
//! reads the top summaries first and only descends when the caller wants the
//! underlying detail.
//!
//! Everything is scope-bounded (a summary never mixes user/project/mode) and
//! rebuildable — `curation::rebuild` re-derives the whole tree after re-deriving
//! level 0, and produces the same shape.
//!
//! Clustering reuses the entity-Jaccard signal the level-0 distillation pass
//! already uses (`curation::cluster_indices_by_jaccard`). Rationale:
//! it needs no extra embedding round-trips, behaves identically under the
//! heuristic (no-LLM) provider, and keeps one clustering story across the whole
//! curated layer. Entities for a summary node are the union of its children's
//! entities, so overlap propagates up the tree.
//!
//! Summarization: a cluster's children are handed to `AiProvider::distill_facts`
//! (via `NlpService`) and the returned fact becomes the summary content. When
//! the provider can't distill (heuristic-only deployment) we fall back to an
//! EXTRACTIVE summary — concatenate the highest-importance children, truncated
//! to a budget — never fabricating text.

use sqlx::PgPool;
use uuid::Uuid;

use flashback_nlp::EpisodeRef;

use crate::curation::{cluster_indices_by_jaccard, edges, Scope};
use crate::error::AppResult;
use crate::nlp::NlpService;

/// A level stops recursing once it has this few nodes — that becomes the root
/// tier of the tree (the "top summaries" the feed reads first).
const SUMMARY_FANOUT_STOP: usize = 5;
/// Hard depth cap so a pathological corpus can't build an unbounded tower.
const MAX_LEVELS: i32 = 5;
/// Jaccard threshold for merging two nodes at a level into one summary cluster.
/// Matches the level-0 distillation threshold for one clustering story.
const SUMMARY_CLUSTER_JACCARD: f32 = 0.4;
/// Character budget for an extractive (heuristic) summary before truncation.
const EXTRACTIVE_BUDGET: usize = 600;

/// Tunables for the RAPTOR build. `Default` is the production behavior; tests
/// drive a smaller `fanout_stop` to build a deterministic multi-level tree from
/// a compact corpus.
#[derive(Debug, Clone, Copy)]
pub struct SummaryConfig {
    /// Stop recursing once a level has at most this many nodes.
    pub fanout_stop: usize,
    /// Hard depth cap.
    pub max_levels: i32,
    /// Entity-Jaccard merge threshold.
    pub cluster_jaccard: f32,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            fanout_stop: SUMMARY_FANOUT_STOP,
            max_levels: MAX_LEVELS,
            cluster_jaccard: SUMMARY_CLUSTER_JACCARD,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SummaryStats {
    /// Summary nodes written across all levels.
    pub summaries: i64,
    /// Highest level reached (0 = no summaries built).
    pub max_level: i32,
}

/// A node participating in a summary level, with entities used for clustering.
struct LevelNode {
    id: Uuid,
    content: String,
    importance: Option<f32>,
    entities: Vec<String>,
}

/// Build the RAPTOR summary tree for a scope on top of its existing level-0
/// curated nodes. Idempotent within a rebuild: callers wipe curated nodes
/// first, so this always builds from a clean level 0. Scope-bounded — never
/// crosses user/project/mode.
pub async fn build_summaries(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
) -> AppResult<SummaryStats> {
    build_summaries_with(pool, nlp, scope, SummaryConfig::default()).await
}

/// `build_summaries` with an explicit config (tests use a small `fanout_stop`).
pub async fn build_summaries_with(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
    cfg: SummaryConfig,
) -> AppResult<SummaryStats> {
    let mut stats = SummaryStats::default();
    let mut level = 0_i32;

    loop {
        if level >= cfg.max_levels {
            break;
        }
        let nodes = load_level(pool, nlp, scope, level).await?;
        // Need at least two nodes to summarize, and stop once a level is already
        // small enough to serve as the tree's root tier.
        if nodes.len() <= cfg.fanout_stop {
            break;
        }

        let entity_sets: Vec<&[String]> = nodes.iter().map(|n| n.entities.as_slice()).collect();
        let clusters = cluster_indices_by_jaccard(&entity_sets, cfg.cluster_jaccard);

        // If clustering can't group anything (every node isolated), summarizing
        // one-node "clusters" would just copy content up forever. Stop.
        let any_multi = clusters.iter().any(|c| c.len() >= 2);
        if !any_multi {
            break;
        }

        let parent_level = level + 1;
        let mut built_this_level = 0_i64;
        for cluster in &clusters {
            // A singleton cluster carries straight up (its own node still needs a
            // parent so the tree stays connected), but a summary only earns a new
            // node when it actually merges ≥2 children — otherwise we'd inflate
            // the tree with copies. Singletons are promoted by reference: they
            // simply reappear at the next level's load via their own summary edge
            // from a multi-node parent, or terminate here.
            if cluster.len() < 2 {
                continue;
            }
            let children: Vec<&LevelNode> = cluster.iter().map(|&i| &nodes[i]).collect();
            let node_id = write_summary(pool, nlp, scope, parent_level, &children).await?;
            let _ = node_id;
            built_this_level += 1;
        }

        if built_this_level == 0 {
            break;
        }
        stats.summaries += built_this_level;
        stats.max_level = parent_level;
        level = parent_level;
    }

    Ok(stats)
}

/// Load the summarizable nodes at `level` for a scope, with entities. At level 0
/// these are the active episodic/semantic nodes; above 0 they are the summary
/// nodes just written. Entities are extracted from content on the fly (the same
/// contract the curation pass uses — the curated layer has no entities column).
async fn load_level(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
    level: i32,
) -> AppResult<Vec<LevelNode>> {
    // Level-0 nodes that a semantic node already supersedes are dropped from the
    // active set (mirrors distill's filter); summary levels have no supersedes.
    let rows: Vec<(Uuid, String, Option<f32>)> = sqlx::query_as(
        r#"
        SELECT n.id, n.content, n.importance
        FROM curated_nodes n
        WHERE n.level = $1
          AND ($2 = '*' OR n.user_id = $2)
          AND n.project_id IS NOT DISTINCT FROM $3
          AND n.mode IS NOT DISTINCT FROM $4
          AND NOT EXISTS (
              SELECT 1 FROM curated_edges s
              WHERE s.to_id = n.id AND s.kind = 'supersedes'
          )
        ORDER BY n.created_at ASC
        LIMIT $5
        "#,
    )
    .bind(level)
    .bind(&scope.user_id)
    .bind(&scope.project_id)
    .bind(&scope.mode)
    .bind(crate::curation::curation_batch_cap())
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(id, content, importance)| {
            let entities = nlp.extract_entities(&content);
            LevelNode {
                id,
                content,
                importance,
                entities,
            }
        })
        .collect())
}

/// Write one summary node over `children`: distill (or extractively summarize)
/// their content, insert at `level`, add a `summarizes` edge to every child,
/// carry the max child importance, and embed it.
async fn write_summary(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
    level: i32,
    children: &[&LevelNode],
) -> AppResult<Uuid> {
    let content = summarize_cluster(nlp, children).await;
    let importance = children
        .iter()
        .filter_map(|c| c.importance)
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |a| a.max(v)))
        });

    let node_id = Uuid::new_v4();
    edges::insert_node_at_level(
        pool, node_id, "summary", &content, level, scope, importance, None,
    )
    .await?;
    for child in children {
        edges::add_edge(pool, node_id, child.id, "summarizes").await?;
    }
    edges::embed_node(pool, nlp, scope, node_id, &content).await;
    Ok(node_id)
}

/// Summarize a cluster of children into one string. Prefers the provider's
/// distillation; falls back to an extractive summary (highest-importance
/// children concatenated + truncated) when the provider can't distill. Never
/// fabricates — the extractive path only reuses the children's own text.
async fn summarize_cluster(nlp: &dyn NlpService, children: &[&LevelNode]) -> String {
    if nlp.provider_can_distill() {
        let refs: Vec<EpisodeRef> = children
            .iter()
            .map(|c| EpisodeRef {
                id: c.id,
                content: c.content.clone(),
                topic: None,
                entities: c.entities.clone(),
                // Summaries compress content, they don't adjudicate currency —
                // the recency rule is distillation's job, one level down.
                when: None,
            })
            .collect();
        if let Ok(facts) = nlp.distill_facts(&refs).await {
            if let Some(first) = facts.into_iter().find(|f| !f.content.trim().is_empty()) {
                return first.content;
            }
        }
        // Distillation configured but empty/failed — fall through to extractive
        // rather than emit nothing.
    }
    extractive_summary(children)
}

/// Extractive fallback: order children by importance (desc), concatenate their
/// content, truncate to `EXTRACTIVE_BUDGET` on a char boundary. Reuses only the
/// children's own words — no fabrication.
fn extractive_summary(children: &[&LevelNode]) -> String {
    let mut ordered: Vec<&&LevelNode> = children.iter().collect();
    ordered.sort_by(|a, b| {
        b.importance
            .unwrap_or(0.0)
            .partial_cmp(&a.importance.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let joined = ordered
        .iter()
        .map(|n| n.content.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");

    truncate_on_boundary(&joined, EXTRACTIVE_BUDGET)
}

/// Truncate to at most `max` bytes without splitting a UTF-8 char.
fn truncate_on_boundary(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let mut out = s[..end].trim_end().to_string();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AppError;
    use async_trait::async_trait;
    use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};
    use sqlx::PgPool;

    // -- pure helpers ------------------------------------------------------

    fn node(content: &str, importance: Option<f32>) -> LevelNode {
        LevelNode {
            id: Uuid::new_v4(),
            content: content.to_string(),
            importance,
            entities: content.split_whitespace().map(str::to_string).collect(),
        }
    }

    #[test]
    fn extractive_orders_by_importance_and_joins() {
        let a = node("low priority detail", Some(0.1));
        let b = node("high priority headline", Some(0.9));
        let refs = [&a, &b];
        let out = extractive_summary(&refs);
        // Highest-importance child leads.
        assert!(out.starts_with("high priority headline"));
        assert!(out.contains("low priority detail"));
    }

    #[test]
    fn extractive_truncates_on_char_boundary() {
        let long = "x".repeat(5000);
        let a = node(&long, Some(0.5));
        let refs = [&a];
        let out = extractive_summary(&refs);
        assert!(out.len() <= EXTRACTIVE_BUDGET + 4); // budget + ellipsis bytes
        assert!(out.ends_with('…'));
    }

    #[test]
    fn truncate_respects_multibyte() {
        // A string of multibyte chars; truncation must not panic or split.
        let s = "日本語".repeat(500);
        let out = truncate_on_boundary(&s, 100);
        assert!(out.len() <= 100 + 4);
        // Round-trips as valid UTF-8 (String guarantees it; assert non-empty).
        assert!(!out.is_empty());
    }

    // -- test doubles ------------------------------------------------------

    /// Provider that CAN distill (models an LLM deployment). Real heuristic
    /// entity extraction so clustering is exercised for real; distill returns a
    /// single synthetic fact so the summary path runs without an LLM.
    #[derive(Clone)]
    struct DistillingNlp;

    #[async_trait]
    impl NlpService for DistillingNlp {
        fn provider_name(&self) -> &'static str {
            "test-distill"
        }
        fn provider_can_distill(&self) -> bool {
            true
        }
        fn embedder_model_name(&self) -> &str {
            "test-embedder"
        }
        fn embedder_dimension(&self) -> usize {
            384
        }
        async fn embed_one(&self, _t: &str) -> Result<Vec<f32>, AppError> {
            Ok(vec![0.1_f32; 384])
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| vec![0.1_f32; 384]).collect())
        }
        fn extract_entities(&self, text: &str) -> Vec<String> {
            flashback_nlp::extract_entities(text)
        }
        async fn extract_full(&self, _t: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            episodes: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            // Distinctive multi-word phrase so the summary node's own extracted
            // entities are non-empty and identical across summaries — letting a
            // higher level cluster them (drives the >=2 level tree in tests).
            Ok(vec![DistilledFact {
                content: "cluster rollup summary node".to_string(),
                topic: None,
                source_episode_ids: episodes.iter().map(|e| e.id).collect(),
                confidence: 0.9,
            }])
        }
    }

    /// Heuristic-style provider that CANNOT distill (no-LLM deployment). Forces
    /// the extractive summary fallback.
    #[derive(Clone)]
    struct HeuristicNlp;

    #[async_trait]
    impl NlpService for HeuristicNlp {
        fn provider_name(&self) -> &'static str {
            "heuristic"
        }
        fn provider_can_distill(&self) -> bool {
            false
        }
        fn embedder_model_name(&self) -> &str {
            "test-embedder"
        }
        fn embedder_dimension(&self) -> usize {
            384
        }
        async fn embed_one(&self, _t: &str) -> Result<Vec<f32>, AppError> {
            Ok(vec![0.1_f32; 384])
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok((0..texts.len()).map(|_| vec![0.1_f32; 384]).collect())
        }
        fn extract_entities(&self, text: &str) -> Vec<String> {
            flashback_nlp::extract_entities(text)
        }
        async fn extract_full(&self, _t: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            _e: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("heuristic".into()))
        }
    }

    // -- DB helpers --------------------------------------------------------

    /// Insert a level-0 curated node directly with a known content (so its
    /// extracted entities are controllable) plus its embedding. Bypasses the
    /// promote/distill pass — this module builds the tree *over* level-0 nodes,
    /// and we want deterministic clusters.
    async fn insert_level0(
        pool: &PgPool,
        nlp: &dyn NlpService,
        scope: &Scope,
        kind: &str,
        content: &str,
        importance: Option<f32>,
    ) -> Uuid {
        let id = Uuid::new_v4();
        edges::insert_node_at_level(pool, id, kind, content, 0, scope, importance, None)
            .await
            .unwrap();
        edges::embed_node(pool, nlp, scope, id, content).await;
        id
    }

    async fn count_by_kind(pool: &PgPool, user_id: &str, kind: &str) -> i64 {
        sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes WHERE ($1 = '*' OR user_id = $1) AND kind = $2",
        )
        .bind(user_id)
        .bind(kind)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    async fn max_level(pool: &PgPool, user_id: &str) -> i32 {
        sqlx::query_scalar(
            "SELECT COALESCE(MAX(level), 0) FROM curated_nodes WHERE ($1 = '*' OR user_id = $1)",
        )
        .bind(user_id)
        .fetch_one(pool)
        .await
        .unwrap()
    }

    async fn count_summarizes_edges(pool: &PgPool) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM curated_edges WHERE kind = 'summarizes'")
            .fetch_one(pool)
            .await
            .unwrap()
    }

    /// Seed six level-0 nodes across three tight entity clusters. Each pair
    /// shares a distinctive multi-word phrase so entity Jaccard groups them.
    async fn seed_three_clusters(pool: &PgPool, nlp: &dyn NlpService, scope: &Scope) {
        // Three tight pairs: each pair extracts an identical entity phrase
        // (jaccard 1.0 -> one cluster), the pairs are disjoint (jaccard 0).
        for content in [
            "the deploy target moved to staging",
            "the deploy target moved to production",
            "the auth layer moved to staging",
            "the auth layer moved to production",
            "the price plan moved to staging",
            "the price plan moved to production",
        ] {
            insert_level0(pool, nlp, scope, "episodic", content, Some(0.5)).await;
        }
    }

    // -- multi-level tree --------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn builds_multi_level_tree_with_summarizes_edges(pool: PgPool) {
        let scope = Scope::new("alice");
        seed_three_clusters(&pool, &DistillingNlp, &scope).await;

        // fanout_stop = 1 forces recursion until a level has a single node.
        let cfg = SummaryConfig {
            fanout_stop: 1,
            ..SummaryConfig::default()
        };
        let stats = build_summaries_with(&pool, &DistillingNlp, &scope, cfg)
            .await
            .unwrap();

        // Level 1: three cluster summaries. Level 2: those three (identical
        // synthetic content -> identical entities) collapse into one. So the
        // tree reaches at least level 2 with summaries at 1 and 2.
        assert!(
            stats.max_level >= 2,
            "expected >=2 levels, got {}",
            stats.max_level
        );
        assert_eq!(max_level(&pool, "alice").await, stats.max_level);

        let l1 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM curated_nodes WHERE user_id='alice' AND kind='summary' AND level=1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(l1, 3, "three level-1 cluster summaries");

        let l2 = sqlx::query_scalar::<_, i64>(
            "SELECT COUNT(*) FROM curated_nodes WHERE user_id='alice' AND kind='summary' AND level=2",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(l2 >= 1, "at least one level-2 summary");

        // Every summary points at its children via `summarizes` edges, and each
        // level-1 summary was embedded.
        assert!(count_summarizes_edges(&pool).await >= 6); // 3 clusters x 2 children
        let embedded: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_embeddings e JOIN curated_nodes n ON n.id=e.node_id \
             WHERE n.kind='summary'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(embedded, l1 + l2);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn rebuild_reproduces_the_tree(pool: PgPool) {
        let scope = Scope::new("alice");
        seed_three_clusters(&pool, &DistillingNlp, &scope).await;
        let cfg = SummaryConfig {
            fanout_stop: 1,
            ..SummaryConfig::default()
        };

        let first = build_summaries_with(&pool, &DistillingNlp, &scope, cfg)
            .await
            .unwrap();
        let sum1 = count_by_kind(&pool, "alice", "summary").await;

        // Wipe just the summaries (levels > 0) + their edges, then rebuild.
        sqlx::query(
            "DELETE FROM curated_edges WHERE from_id IN \
             (SELECT id FROM curated_nodes WHERE user_id='alice' AND level > 0)",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query("DELETE FROM curated_nodes WHERE user_id='alice' AND level > 0")
            .execute(&pool)
            .await
            .unwrap();
        assert_eq!(count_by_kind(&pool, "alice", "summary").await, 0);

        let second = build_summaries_with(&pool, &DistillingNlp, &scope, cfg)
            .await
            .unwrap();
        assert_eq!(second.max_level, first.max_level);
        assert_eq!(count_by_kind(&pool, "alice", "summary").await, sum1);
    }

    // -- extractive fallback (no-LLM) --------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn heuristic_provider_builds_extractive_summaries(pool: PgPool) {
        let scope = Scope::new("alice");
        // Two nodes with a shared phrase so they cluster; heuristic can't distill
        // -> extractive summary reuses the children's own words.
        insert_level0(
            &pool,
            &HeuristicNlp,
            &scope,
            "episodic",
            "the deploy target moved to staging",
            Some(0.9),
        )
        .await;
        insert_level0(
            &pool,
            &HeuristicNlp,
            &scope,
            "episodic",
            "the deploy target moved to production",
            Some(0.2),
        )
        .await;

        let cfg = SummaryConfig {
            fanout_stop: 1,
            ..SummaryConfig::default()
        };
        let stats = build_summaries_with(&pool, &HeuristicNlp, &scope, cfg)
            .await
            .unwrap();
        assert!(stats.summaries >= 1);

        // The extractive summary must be built ONLY from the children's own text
        // (highest-importance child leads) — never fabricated.
        let content: String = sqlx::query_scalar(
            "SELECT content FROM curated_nodes WHERE user_id='alice' AND kind='summary' AND level=1 LIMIT 1",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(content.contains("staging")); // higher-importance child first
        assert!(content.contains("production"));
    }

    // -- scope isolation ---------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn summaries_never_cross_scope(pool: PgPool) {
        let alice = Scope::new("alice");
        let bob = Scope::new("bob");
        seed_three_clusters(&pool, &DistillingNlp, &alice).await;
        seed_three_clusters(&pool, &DistillingNlp, &bob).await;

        let cfg = SummaryConfig {
            fanout_stop: 1,
            ..SummaryConfig::default()
        };
        // Build only alice's tree.
        build_summaries_with(&pool, &DistillingNlp, &alice, cfg)
            .await
            .unwrap();

        assert!(count_by_kind(&pool, "alice", "summary").await > 0);
        assert_eq!(count_by_kind(&pool, "bob", "summary").await, 0);

        // Every `summarizes` edge stays within alice's node set.
        let leaked: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_edges e \
             JOIN curated_nodes f ON f.id = e.from_id \
             JOIN curated_nodes t ON t.id = e.to_id \
             WHERE e.kind='summarizes' AND (f.user_id <> 'alice' OR t.user_id <> 'alice')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(leaked, 0);
    }
}
