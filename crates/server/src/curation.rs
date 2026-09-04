//! Curation pipeline over `raw_records` — the derived layer.
//!
//! Built on the immutable raw layer. The contract: raw is read-only and
//! immutable; the whole curated set is rebuildable from raw. `curate` is the
//! hot path — an incremental pass that only touches what changed since the
//! last one — and `rebuild` (wipe + re-derive) is the recovery tool that
//! proves the rebuildability contract, not the schedule.
//!
//! Raw `type` names a kind of evidence, never a tier — a writer says what it
//! arrived, this module says what that arrival became. Tier vocabulary
//! (`episodic`, `semantic`, `summary`) exists only in `curated_nodes.kind`.
//!
//! Two derivations, both scoped to a single (user, mode) tuple and
//! never crossing it:
//!
//!   - `promote_raw_to_episodic`: turns the promotable raw types into
//!     `curated_nodes` rows `kind='episodic'`, each with
//!     `curated_edges('derived_from')` back to every raw id it covers. A
//!     `conversation` turn is not an episode on its own — the episode is the
//!     whole conversation — so those group by `thread_id` into one node per
//!     session, carrying a transcript and a derived title in `meta`.
//!     `document` rows stand alone, one node each.
//!     Idempotent: a session that already has an episode is skipped, and a
//!     standalone row is skipped once a `derived_from` edge points at it.
//!
//!   - `distill_semantic`: clusters active episodic curated nodes by entity
//!     overlap — each episode carrying its whole transcript, the union of its
//!     raw rows' entities, and the time span it covers — then asks the
//!     configured `AiProvider` to distill each cluster into semantic facts.
//!     Entities come from `entity_index` (the HippoRAG pointer table,
//!     populated during promotion); on-the-fly `extract_entities` over the
//!     node content is the fallback when the index has no rows yet. Each fact
//!     becomes a `kind='semantic'` node dated by its newest evidence, with
//!     `derived_from` edges to the raw rows of the episodes the distiller
//!     cited (whole-cluster fallback when it cites nothing valid). Requires a
//!     provider that can distill; the heuristic provider degrades gracefully
//!     (logs + no-op).
//!
//! Every new curated node is embedded into `curated_embeddings`. Promotion also
//! populates `entity_index` and emits `curated_edges(kind='entity')` tying the
//! episodic node to its source raw id, labelled with the entity (the glass-box
//! view of what a node clustered on; the index is the scan-friendly hop).

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use pgvector::Vector;
use sqlx::PgPool;
use uuid::Uuid;

use flashback_nlp::EpisodeRef;

use crate::error::AppResult;
use crate::nlp::NlpService;

/// Jaccard threshold for merging two episodic nodes into one semantic cluster.
/// Only the FALLBACK signal now — used when a node has no stored embedding.
const CLUSTER_JACCARD: f32 = 0.4;

/// Cosine-to-centroid threshold for embedding clustering. The primary signal:
/// meaning, not shared entity strings, is what makes two episodes one topic.
pub(crate) const CLUSTER_COSINE: f32 = 0.60;

/// Ceiling on episodes per cluster. Bounds the distill prompt (episode content
/// is transcript-budgeted, so prompt size is at most cap x budget) and stops a
/// hub topic from dragging half the corpus into one call.
pub(crate) const CLUSTER_MAX: usize = 12;

/// Already-distilled episodes carried into clustering as CONTEXT, newest
/// first. A new episode on an old topic clusters with the evidence that
/// produced the existing fact, so the refreshed fact is distilled with full
/// context instead of in a vacuum — and can then supersede its predecessor.
const DISTILL_CONTEXT: i64 = 200;

/// Cosine floor for one semantic fact superseding another. Deliberately
/// conservative: only a near-same-topic newer fact closes an older one's
/// validity window (the filters everywhere already exclude superseded nodes).
/// Too low retires unrelated facts; too high never fires.
const SUPERSEDE_COSINE: f64 = 0.85;

/// Upper bound on how many raw/episodic rows a single curation pass pulls into
/// memory (and, for distillation, feeds the O(n²) entity clusterer). Promotion
/// is idempotent and the clusterer is quadratic, so a large backlog is drained
/// over successive scheduled passes instead of loading a whole corpus at once —
/// this keeps one rebuild bounded in both memory and CPU. Tunable per deployment
/// via `FLASHBACK_CURATION_BATCH`.
pub(crate) fn curation_batch_cap() -> i64 {
    std::env::var("FLASHBACK_CURATION_BATCH")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .filter(|n| *n > 0)
        .unwrap_or(5_000)
}

#[derive(Debug, Default, Clone)]
pub struct CurationStats {
    pub promoted: i64,
    /// Existing episodes refreshed in place because their session grew.
    pub refreshed: i64,
    pub distilled: i64,
    pub clusters_seen: i64,
    /// Older facts whose validity window a newer fact closed this pass.
    pub superseded: i64,
    /// Clusters whose distill call failed twice this pass — retried next pass.
    pub clusters_failed: i64,
    pub skipped_distill: bool,
    /// Summary nodes built across all RAPTOR levels.
    pub summarized: i64,
    /// Deepest summary level reached (0 = flat, no summaries).
    pub max_level: i32,
    /// Promotable work still waiting after the pass (batch cap left it for the
    /// next tick). Non-zero is normal mid-drain; non-zero forever is the
    /// starvation this field exists to make visible.
    pub pending_sessions: i64,
    pub pending_solos: i64,
    /// Episodes still carrying no `distilled_at` marker after the pass —
    /// the distill-stage backlog, counted so IT can't starve silently either.
    pub pending_undistilled: i64,
    /// Another pass held this user's curation lock; nothing was done.
    pub locked_out: bool,
}

// ---------------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------------

/// The (user, mode) tuple every curation pass is scoped to. Topics do not
/// partition — health can feed taxes. Mode does: vector dimensions differ.
#[derive(Debug, Clone)]
pub struct Scope {
    pub user_id: String,
    pub mode: Option<String>,
}

impl Scope {
    pub fn new(user_id: impl Into<String>) -> Self {
        Self {
            user_id: user_id.into(),
            mode: None,
        }
    }

    pub fn with_mode(mut self, mode: Option<String>) -> Self {
        self.mode = mode;
        self
    }

    /// The embedder key this scope's mode pins, for embedding curated nodes in
    /// the same geometry as the raw records they derive from. A mode-less scope
    /// (mode = NULL) or an unresolvable mode falls back to the default embedder
    /// key (all-MiniLM-L6-v2, 384-dim).
    pub(crate) async fn embedder_key(&self, pool: &PgPool) -> String {
        const DEFAULT: &str = "sentence-transformers/all-MiniLM-L6-v2";
        let Some(mode) = self.mode.as_deref() else {
            return DEFAULT.to_string();
        };
        match crate::modes::get_mode(pool, &self.user_id, mode).await {
            Ok(Some(m)) => m.embedder,
            _ => DEFAULT.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// promote_raw_to_episodic
// ---------------------------------------------------------------------------

#[derive(Debug, sqlx::FromRow)]
struct PromotableRaw {
    id: Uuid,
    content: String,
    event_time: DateTime<Utc>,
    source: String,
    payload: Option<serde_json::Value>,
}

/// Raw types curation derives episodes from. `state_object` is deliberately
/// absent — the reference/catalog layer owns those rows and derives its own
/// view of them.
const PROMOTABLE_TYPES: [&str; 2] = ["conversation", "document"];

/// Characters of transcript an episode node carries. The turns stay whole and
/// separately searchable in raw; this is the representative text that gets
/// embedded and handed to distillation, so it is bounded rather than complete.
const EPISODE_CONTENT_BUDGET: usize = 4_000;

/// Characters of a derived conversation title.
const TITLE_BUDGET: usize = 80;

/// Truncate on a character boundary, appending an ellipsis when it bit.
fn truncate_chars(s: &str, max: usize) -> String {
    let mut out: String = s.chars().take(max).collect();
    if s.chars().count() > max {
        out.push('…');
    }
    out
}

/// The speaker label for a turn. Writers name a source as `host:agent:role`
/// (ritsu) or a bare origin (`chatgpt`); the trailing segment is the useful
/// label in both shapes.
fn speaker_of(source: &str) -> &str {
    source.rsplit(':').next().unwrap_or(source)
}

/// Render a conversation's turns as a bounded, ordered transcript. Only ever
/// reuses the turns' own text — nothing here fabricates.
fn render_transcript(turns: &[PromotableRaw]) -> String {
    let mut out = String::new();
    for t in turns {
        let line = format!("{}: {}\n", speaker_of(&t.source), t.content.trim());
        if out.chars().count() + line.chars().count() > EPISODE_CONTENT_BUDGET {
            out.push('…');
            break;
        }
        out.push_str(&line);
    }
    out
}

/// A conversation's title: the source's own title when the import carried one,
/// otherwise the opening turn, truncated. Extractive by design — a title is a
/// label for something the user said, not a claim about it.
fn derive_title(turns: &[PromotableRaw]) -> String {
    let imported = turns.iter().find_map(|t| {
        t.payload
            .as_ref()
            .and_then(|p| p.get("conversation_title"))
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    });
    if let Some(title) = imported {
        return truncate_chars(title, TITLE_BUDGET);
    }
    let opening = turns
        .iter()
        .find(|t| !t.content.trim().is_empty())
        .map(|t| t.content.trim())
        .unwrap_or_default();
    let first_line = opening.lines().next().unwrap_or(opening).trim();
    truncate_chars(first_line, TITLE_BUDGET)
}

/// The active turns of one session, oldest first.
async fn load_session_turns(
    pool: &PgPool,
    scope: &Scope,
    thread_id: &str,
) -> AppResult<Vec<PromotableRaw>> {
    let turns = sqlx::query_as::<_, PromotableRaw>(
        r#"
        SELECT r.id, r.content, r.event_time, r.source, r.payload
        FROM raw_records r
        WHERE r.type = 'conversation'
          AND r.user_id = $1
          AND (SELECT dm.mode FROM derived_record_mode dm WHERE dm.record_id = r.id) IS NOT DISTINCT FROM $2
          AND r.thread_id = $3
          AND NOT EXISTS (SELECT 1 FROM derived_superseded d WHERE d.record_id = r.id)
        ORDER BY r.event_time ASC, r.ingest_time ASC
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(thread_id)
    .fetch_all(pool)
    .await?;
    Ok(turns)
}

/// Derive and write one session's episode under `node_id` — the transcript,
/// title, span meta, per-turn lineage and entity edges, and the embedding.
/// `refresh` re-derives an EXISTING node in place: same id (retrieval
/// references and decay state survive), fresh content/meta/edges/embedding.
/// Rewriting `meta` whole drops any `distilled_at` marker, which is the point —
/// a grown conversation must be distillable again.
async fn write_session_episode(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
    node_id: Uuid,
    thread_id: &str,
    turns: &[PromotableRaw],
    refresh: bool,
) -> AppResult<()> {
    let content = render_transcript(turns);
    let title = derive_title(turns);
    let started = turns[0].event_time;
    let ended = turns[turns.len() - 1].event_time;
    // No importance is inherited any more: it used to come from a number the
    // writer put on the raw row, which meant a rebuild could never disagree with
    // it. Curation should decide this itself; until it does, it stays unset.
    let importance: Option<f32> = None;

    if refresh {
        sqlx::query(
            "UPDATE curated_nodes SET content = $2, importance = $3, event_time = $4 WHERE id = $1",
        )
        .bind(node_id)
        .bind(&content)
        .bind(importance)
        .bind(started)
        .execute(pool)
        .await?;
        // The old edges and embedding describe the old transcript.
        sqlx::query(
            "DELETE FROM curated_edges WHERE from_id = $1 AND kind IN ('derived_from', 'entity')",
        )
        .bind(node_id)
        .execute(pool)
        .await?;
        sqlx::query("DELETE FROM curated_embeddings WHERE node_id = $1")
            .bind(node_id)
            .execute(pool)
            .await?;
    } else {
        insert_node(
            pool,
            node_id,
            "episodic",
            &content,
            scope,
            importance,
            Some(started),
        )
        .await?;
    }
    set_node_meta(
        pool,
        node_id,
        serde_json::json!({
            "thread_id": thread_id,
            "title": title,
            "turns": turns.len(),
            "started_at": started,
            "ended_at": ended,
        }),
    )
    .await?;
    embed_node(pool, nlp, scope, node_id, &content).await;

    // Lineage and entity index run per turn, so the glass-box view still
    // resolves to the individual raw rows an episode was built from.
    for t in turns {
        add_edge(pool, node_id, t.id, "derived_from").await?;
        let entities = nlp.extract_entities(&t.content);
        index_entities(pool, &scope.user_id, t.id, &entities).await?;
        for entity in &entities {
            add_entity_edge(pool, node_id, t.id, entity).await?;
        }
    }
    Ok(())
}

/// Derive episodic nodes from the promotable raw records in scope.
///
/// A `conversation` turn is not an episode on its own — an episode is the whole
/// conversation, the block of time the turns share. So conversation rows group
/// by `thread_id` and yield **one** node per session, carrying the transcript,
/// a derived title, and `derived_from` edges to every turn it covers. Rows with
/// no container (and `document` rows) stand alone, one node each.
///
/// Idempotent at the session level: a session that already has an episode is
/// skipped rather than duplicated. A session that has since GROWN is refreshed
/// in place — same node id, fresh transcript/meta/edges/embedding — so its new
/// turns are picked up without waiting for a destructive rebuild.
pub async fn promote_raw_to_episodic(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
) -> AppResult<CurationStats> {
    let mut stats = CurationStats::default();

    // --- conversations: one episode per not-yet-curated session -------------
    // Active = not superseded by a newer row, and not expired.
    let sessions: Vec<String> = sqlx::query_scalar(
        r#"
        SELECT DISTINCT r.thread_id
        FROM raw_records r
        WHERE r.type = 'conversation'
          AND r.thread_id IS NOT NULL
          AND r.user_id = $1
          AND (SELECT dm.mode FROM derived_record_mode dm WHERE dm.record_id = r.id) IS NOT DISTINCT FROM $2
          AND NOT EXISTS (SELECT 1 FROM derived_superseded d WHERE d.record_id = r.id)
          AND NOT EXISTS (
              SELECT 1 FROM curated_nodes n
              WHERE n.kind = 'episodic'
                AND n.user_id = $1
                AND n.mode IS NOT DISTINCT FROM $2
                AND n.meta->>'thread_id' = r.thread_id
          )
        ORDER BY r.thread_id
        LIMIT $3
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(curation_batch_cap())
    .fetch_all(pool)
    .await?;

    for thread_id in sessions {
        let turns = load_session_turns(pool, scope, &thread_id).await?;
        if turns.is_empty() {
            continue;
        }
        let node_id = Uuid::new_v4();
        write_session_episode(pool, nlp, scope, node_id, &thread_id, &turns, false).await?;
        stats.promoted += 1;
    }

    // --- grown sessions: refresh the episode in place ------------------------
    // A session that gained turns since its episode was derived is stale: the
    // transcript misses the new turns and any distillation marker no longer
    // covers the conversation. Refresh reuses the SAME node id, so retrieval
    // references and decay state survive; rewriting `meta` clears
    // `distilled_at`, making the episode eligible for distillation again.
    let grown: Vec<(Uuid, String)> = sqlx::query_as(
        r#"
        SELECT n.id, n.meta->>'thread_id' AS thread_id
        FROM curated_nodes n
        WHERE n.kind = 'episodic'
          AND n.user_id = $1
          AND n.mode IS NOT DISTINCT FROM $2
          AND n.meta->>'thread_id' IS NOT NULL
          AND (n.meta->>'turns')::bigint <> (
              SELECT COUNT(*)
              FROM raw_records r
              WHERE r.type = 'conversation'
                AND r.user_id = $1
                AND (SELECT dm.mode FROM derived_record_mode dm WHERE dm.record_id = r.id) IS NOT DISTINCT FROM $2
                AND r.thread_id = n.meta->>'thread_id'
                AND NOT EXISTS (SELECT 1 FROM derived_superseded d WHERE d.record_id = r.id)
          )
        ORDER BY n.id
        LIMIT $3
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(curation_batch_cap())
    .fetch_all(pool)
    .await?;

    for (node_id, thread_id) in grown {
        let turns = load_session_turns(pool, scope, &thread_id).await?;
        if turns.is_empty() {
            continue;
        }
        write_session_episode(pool, nlp, scope, node_id, &thread_id, &turns, true).await?;
        stats.refreshed += 1;
    }

    // --- standalone rows: documents, container-less turns -------------------
    let solo = sqlx::query_as::<_, PromotableRaw>(
        r#"
        SELECT r.id, r.content, r.event_time, r.source, r.payload
        FROM raw_records r
        WHERE r.type = ANY($4)
          AND (r.type <> 'conversation' OR r.thread_id IS NULL)
          AND r.user_id = $1
          AND (SELECT dm.mode FROM derived_record_mode dm WHERE dm.record_id = r.id) IS NOT DISTINCT FROM $2
          AND NOT EXISTS (SELECT 1 FROM derived_superseded d WHERE d.record_id = r.id)
          AND NOT EXISTS (
              SELECT 1 FROM curated_edges e
              JOIN curated_nodes n ON n.id = e.from_id
              WHERE e.to_id = r.id AND e.kind = 'derived_from' AND n.kind = 'episodic'
          )
        ORDER BY r.event_time ASC
        LIMIT $3
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(curation_batch_cap())
    .bind(PROMOTABLE_TYPES.as_slice())
    .fetch_all(pool)
    .await?;

    for r in solo {
        let node_id = Uuid::new_v4();
        insert_node(
            pool,
            node_id,
            "episodic",
            &r.content,
            scope,
            None,
            Some(r.event_time),
        )
        .await?;
        add_edge(pool, node_id, r.id, "derived_from").await?;
        embed_node(pool, nlp, scope, node_id, &r.content).await;

        // Populate the entity index for this raw record and emit one labelled
        // `entity` edge (node -> raw id) per entity. The index is what the
        // distill pass clusters on; the edge is the glass-box lineage.
        let entities = nlp.extract_entities(&r.content);
        index_entities(pool, &scope.user_id, r.id, &entities).await?;
        for entity in &entities {
            add_entity_edge(pool, node_id, r.id, entity).await?;
        }
        stats.promoted += 1;
    }

    Ok(stats)
}

/// Attach derived identity (session, title, span) to a curated node. Curated
/// rows are the pipeline's own, so a plain UPDATE is fine here — the append-only
/// contract binds `raw_records`, not this table.
async fn set_node_meta(pool: &PgPool, node_id: Uuid, meta: serde_json::Value) -> AppResult<()> {
    sqlx::query("UPDATE curated_nodes SET meta = $2 WHERE id = $1")
        .bind(node_id)
        .bind(meta)
        .execute(pool)
        .await?;
    Ok(())
}

/// Insert (user_id, entity, record_id) rows into the HippoRAG pointer table.
/// Idempotent on the composite PK. Populated during promotion so the distill
/// pass and future graph hops read the index instead of re-extracting.
pub(crate) async fn index_entities(
    pool: &PgPool,
    user_id: &str,
    record_id: Uuid,
    entities: &[String],
) -> AppResult<()> {
    for entity in entities {
        sqlx::query(
            "INSERT INTO entity_index (user_id, entity, record_id) VALUES ($1, $2, $3) \
             ON CONFLICT (user_id, entity, record_id) DO NOTHING",
        )
        .bind(user_id)
        .bind(entity)
        .bind(record_id)
        .execute(pool)
        .await?;
    }
    Ok(())
}

/// The indexed entities per raw id for a scope, read from `entity_index`. Only
/// the active `working` raw records in scope are considered (the same set
/// promotion feeds), so a stale index row never leaks entities for a
/// superseded record into the cluster.
async fn entities_for_scope(pool: &PgPool, scope: &Scope) -> AppResult<HashMap<Uuid, Vec<String>>> {
    // Ceiling on index rows pulled into the map. Entities-per-record is small,
    // so a generous multiple of the per-pass node cap covers the clustered set
    // while still bounding memory on a pathologically large index.
    let cap = curation_batch_cap().saturating_mul(32);
    let rows = sqlx::query_as::<_, (Uuid, String)>(
        r#"
        SELECT ei.record_id, ei.entity
        FROM entity_index ei
        JOIN raw_records r ON r.id = ei.record_id
        WHERE ei.user_id = $1
          AND (SELECT dm.mode FROM derived_record_mode dm WHERE dm.record_id = r.id) IS NOT DISTINCT FROM $2
        LIMIT $3
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(cap)
    .fetch_all(pool)
    .await?;

    let mut by_raw: HashMap<Uuid, Vec<String>> = HashMap::new();
    for (raw_id, entity) in rows {
        by_raw.entry(raw_id).or_default().push(entity);
    }
    Ok(by_raw)
}

// ---------------------------------------------------------------------------
// distill_semantic
// ---------------------------------------------------------------------------

/// An active episodic curated node, carried whole into distillation: its own
/// content (the bounded transcript promotion rendered — never a single raw
/// turn), the union of entities across every raw row it derives from, and the
/// time span it covers.
struct EpisodeNode {
    node_id: Uuid,
    /// Every raw id this episodic node was derived from.
    source_raw_ids: Vec<Uuid>,
    content: String,
    entities: Vec<String>,
    /// The node's stored embedding in the scope's geometry, L2-normalized.
    /// Missing (embed failed at write time) falls back to entity overlap.
    embedding: Option<Vec<f32>>,
    /// True when this episode is context (already distilled): it can anchor a
    /// cluster but never triggers one, is never re-marked, never bumped.
    already_distilled: bool,
    /// When the episode started / ended. Equal for a single-record episode.
    started: Option<DateTime<Utc>>,
    ended: Option<DateTime<Utc>>,
}

impl EpisodeNode {
    /// The `when` the distill prompt sees: an instant, or a `start..end` span.
    fn when(&self) -> Option<String> {
        match (self.started, self.ended) {
            (Some(s), Some(e)) if s != e => Some(format!(
                "{}..{}",
                s.to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
                e.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
            )),
            (Some(s), _) => Some(s.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
            _ => None,
        }
    }
}

/// Cluster active episodic curated nodes by entity overlap and distill each
/// cluster into semantic facts. Episodes go to the distiller whole — the
/// transcript promotion rendered, the union of entities across their raw rows,
/// and when they happened — so a fact is derived from conversations, not from
/// one arbitrary turn per conversation. No-op (logged) when the provider can't
/// distill.
pub async fn distill_semantic(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
) -> AppResult<CurationStats> {
    let mut stats = CurationStats::default();

    if !nlp.provider_can_distill() {
        tracing::warn!(
            provider = nlp.provider_name(),
            "curation: distill_semantic skipped — provider does not support fact distillation"
        );
        stats.skipped_distill = true;
        return Ok(stats);
    }

    // Active episodic nodes, read WHOLE: the node's own content is the bounded
    // transcript promotion rendered, and `meta` carries the session span. The
    // deterministic ordering (time, then id) makes a pass reproducible — the
    // previous shape joined through raw rows and inherited whatever content the
    // join happened to surface first.
    let node_rows = sqlx::query_as::<
        _,
        (
            Uuid,
            String,
            Option<DateTime<Utc>>,
            Option<serde_json::Value>,
        ),
    >(
        r#"
        SELECT n.id, n.content, n.event_time, n.meta
        FROM curated_nodes n
        WHERE n.kind = 'episodic'
          AND n.user_id = $1
          AND n.mode IS NOT DISTINCT FROM $2
          AND (n.meta->>'distilled_at') IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM curated_edges s
              WHERE s.to_id = n.id AND s.kind = 'supersedes'
          )
        ORDER BY COALESCE((n.meta->>'distill_attempts')::int, 0) ASC,
                 n.event_time ASC NULLS LAST, n.id ASC
        LIMIT $3
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(curation_batch_cap())
    .fetch_all(pool)
    .await?;
    if node_rows.is_empty() {
        return Ok(stats);
    }

    // Already-distilled episodes ride along as CONTEXT (newest first, capped):
    // a new episode on an old topic clusters with the evidence behind the
    // existing fact, so its fact is distilled with history instead of in a
    // vacuum — and the supersede pass below can then retire the predecessor.
    let context_rows = sqlx::query_as::<
        _,
        (
            Uuid,
            String,
            Option<DateTime<Utc>>,
            Option<serde_json::Value>,
        ),
    >(
        r#"
        SELECT n.id, n.content, n.event_time, n.meta
        FROM curated_nodes n
        WHERE n.kind = 'episodic'
          AND n.user_id = $1
          AND n.mode IS NOT DISTINCT FROM $2
          AND (n.meta->>'distilled_at') IS NOT NULL
          AND NOT EXISTS (
              SELECT 1 FROM curated_edges s
              WHERE s.to_id = n.id AND s.kind = 'supersedes'
          )
        ORDER BY n.event_time DESC NULLS LAST, n.id ASC
        LIMIT $3
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(DISTILL_CONTEXT)
    .fetch_all(pool)
    .await?;

    let node_rows: Vec<_> = node_rows
        .into_iter()
        .map(|r| (r, false))
        .chain(context_rows.into_iter().map(|r| (r, true)))
        .collect();
    if node_rows.len() < 2 {
        return Ok(stats);
    }

    // Lineage for the batch in one query: every raw id each node derives from.
    let node_ids: Vec<Uuid> = node_rows.iter().map(|((id, ..), _)| *id).collect();
    let edge_rows = sqlx::query_as::<_, (Uuid, Uuid)>(
        "SELECT from_id, to_id FROM curated_edges \
         WHERE kind = 'derived_from' AND from_id = ANY($1) \
         ORDER BY from_id, to_id",
    )
    .bind(&node_ids)
    .fetch_all(pool)
    .await?;
    let mut raws_by_node: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for (node_id, raw_id) in edge_rows {
        raws_by_node.entry(node_id).or_default().push(raw_id);
    }

    // The entities each raw id was indexed under, read from `entity_index`
    // (the HippoRAG pointer table). An episode clusters on the UNION of its
    // raw rows' entities — the whole conversation's signal, not one turn's.
    // On-the-fly extraction over the node content is the fallback for records
    // promoted before the index existed.
    let entities_by_raw = entities_for_scope(pool, scope).await?;

    // The stored embeddings in this scope's geometry — the primary clustering
    // signal. A node whose embed failed at write time simply has no row and
    // clusters on the entity fallback.
    let embedder_key = scope.embedder_key(pool).await;
    let dim = flashback_nlp::model_for_key(&embedder_key)
        .map(|(_, d)| d)
        .unwrap_or(384);
    let col = crate::routes::records::embedding_col_for_dim(dim);
    let emb_sql = format!(
        "SELECT node_id, {col} FROM curated_embeddings \
         WHERE node_id = ANY($1) AND model = $2 AND {col} IS NOT NULL"
    );
    let emb_rows: Vec<(Uuid, pgvector::Vector)> = sqlx::query_as(sqlx::AssertSqlSafe(emb_sql))
        .bind(&node_ids)
        .bind(&embedder_key)
        .fetch_all(pool)
        .await
        .unwrap_or_else(|e| {
            tracing::warn!(
                "curation: embedding fetch failed ({e}) — this pass clusters \
                 on the entity fallback only"
            );
            Vec::new()
        });
    let mut emb_by_node: HashMap<Uuid, Vec<f32>> = emb_rows
        .into_iter()
        .map(|(id, v)| (id, v.to_vec()))
        .collect();

    let mut episodes: Vec<EpisodeNode> = Vec::with_capacity(node_rows.len());
    for ((node_id, content, event_time, meta), already_distilled) in node_rows {
        let source_raw_ids = raws_by_node.remove(&node_id).unwrap_or_default();
        if source_raw_ids.is_empty() {
            // An episode nothing points back at can't be cited; skip it.
            continue;
        }
        let mut entities: Vec<String> = source_raw_ids
            .iter()
            .filter_map(|raw_id| entities_by_raw.get(raw_id))
            .flatten()
            .cloned()
            .collect();
        entities.sort();
        entities.dedup();
        if entities.is_empty() {
            entities = nlp.extract_entities(&content);
        }
        let span = |key: &str| {
            meta.as_ref()
                .and_then(|m| m.get(key))
                .and_then(|v| v.as_str())
                .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
                .map(|t| t.with_timezone(&Utc))
        };
        let started = span("started_at").or(event_time);
        let ended = span("ended_at").or(started);
        episodes.push(EpisodeNode {
            embedding: emb_by_node.remove(&node_id),
            already_distilled,
            node_id,
            source_raw_ids,
            content,
            entities,
            started,
            ended,
        });
    }
    if episodes.len() < 2 {
        return Ok(stats);
    }

    let clusters = cluster_by_entities(&episodes);
    // Only clusters that can actually distill count — context-only groups
    // are not work.
    stats.clusters_seen = clusters
        .iter()
        .filter(|c| c.iter().any(|&i| !episodes[i].already_distilled))
        .count() as i64;

    // Singletons stay eligible (a future related episode can still pull them
    // into a cluster), but their attempt count is bumped so the batch ordering
    // rotates PAST them next pass. Without this, one-off topics accumulate at
    // the head of the oldest-first batch and — past the cap — newer episodes
    // never enter distillation at all: the promotion-stage starvation bug,
    // reborn one stage down.
    let singleton_ids: Vec<Uuid> = clusters
        .iter()
        .filter(|c| c.len() < 2)
        .flat_map(|c| c.iter().map(|&i| &episodes[i]))
        .filter(|e| !e.already_distilled)
        .map(|e| e.node_id)
        .collect();
    if !singleton_ids.is_empty() {
        sqlx::query(
            "UPDATE curated_nodes \
             SET meta = COALESCE(meta, '{}'::jsonb) || jsonb_build_object( \
                 'distill_attempts', COALESCE((meta->>'distill_attempts')::int, 0) + 1) \
             WHERE id = ANY($1)",
        )
        .bind(&singleton_ids)
        .execute(pool)
        .await?;
    }

    for cluster in clusters
        .iter()
        .filter(|c| c.len() >= 2 && c.iter().any(|&i| !episodes[i].already_distilled))
    {
        distill_one_cluster(
            pool,
            nlp,
            scope,
            cluster,
            &episodes,
            &embedder_key,
            &mut stats,
        )
        .await?;
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// The per-user curation lock
// ---------------------------------------------------------------------------

/// Advisory-lock namespace for curation. The scheduler tick, the admin curate
/// button and the rebuild endpoint can otherwise overlap — and under the
/// incremental path a race no longer self-heals: two passes that both see a
/// session as unpromoted insert duplicate episodes that then cluster with each
/// other and distill duplicate facts, permanently. One pass per user at a
/// time, everywhere, is the invariant.
const CURATION_LOCK_NS: i32 = 421_337;

/// Try to take the per-user curation lock. `Some(conn)` holds the lock for as
/// long as the connection is kept AND must be released via
/// `release_curation_lock` on the same connection — a pooled session outlives
/// this call, so dropping without unlocking would wedge the user until that
/// session closes. Postgres frees the lock if the process dies.
async fn try_curation_lock(
    pool: &PgPool,
    user_id: &str,
) -> AppResult<Option<sqlx::pool::PoolConnection<sqlx::Postgres>>> {
    let mut conn = pool.acquire().await?;
    let got: bool = sqlx::query_scalar("SELECT pg_try_advisory_lock($1, hashtext($2))")
        .bind(CURATION_LOCK_NS)
        .bind(user_id)
        .fetch_one(&mut *conn)
        .await?;
    Ok(got.then_some(conn))
}

async fn release_curation_lock(
    conn: &mut sqlx::pool::PoolConnection<sqlx::Postgres>,
    user_id: &str,
) {
    let _ = sqlx::query("SELECT pg_advisory_unlock($1, hashtext($2))")
        .bind(CURATION_LOCK_NS)
        .bind(user_id)
        .execute(&mut **conn)
        .await;
}

// ---------------------------------------------------------------------------
// curate — the incremental hot path
// ---------------------------------------------------------------------------

/// One incremental curation pass for a user: promote new sessions and solo
/// records, refresh grown sessions in place, distill what has not been
/// distilled, and rebuild the summary tree only when level 0 actually changed.
/// Never wipes episodic or semantic state — `rebuild` is the recovery tool,
/// this is the scheduled path. Repeated calls with no new raw records do no
/// LLM work at all. Single-flight per user: a pass that finds another running
/// (scheduler vs button vs rebuild) skips with `locked_out` set instead of
/// racing it into duplicate state.
pub async fn curate(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
) -> AppResult<CurationStats> {
    let Some(mut lock) = try_curation_lock(pool, user_id).await? else {
        tracing::warn!(user = %user_id, "curation already running for this user; pass skipped");
        return Ok(CurationStats {
            locked_out: true,
            ..Default::default()
        });
    };
    let result = curate_inner(pool, nlp, user_id).await;
    release_curation_lock(&mut lock, user_id).await;
    result
}

async fn curate_inner(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
) -> AppResult<CurationStats> {
    let buckets: Vec<(Option<String>,)> = sqlx::query_as(
        r#"
        SELECT DISTINCT dm.mode FROM raw_records r
        LEFT JOIN derived_record_mode dm ON dm.record_id = r.id
        WHERE r.user_id = $1 ORDER BY 1
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    let mut total = CurationStats::default();
    for (mode,) in buckets {
        let scope = Scope::new(user_id).with_mode(mode);
        let p = promote_raw_to_episodic(pool, nlp, &scope).await?;
        let d = distill_semantic(pool, nlp, &scope).await?;
        // The summary tree has no incremental maintenance yet: it is derived
        // whole from level 0, so it is re-derived only when level 0 changed —
        // an unchanged scope costs zero summary work. The wipe is scoped to
        // `kind='summary'` alone; episodes, facts and their markers survive.
        let s = if p.promoted + p.refreshed + d.distilled > 0 {
            wipe_summaries(pool, &scope).await?;
            crate::summaries::build_summaries(pool, nlp, &scope).await?
        } else {
            crate::summaries::SummaryStats::default()
        };
        total.promoted += p.promoted;
        total.refreshed += p.refreshed;
        total.distilled += d.distilled;
        total.clusters_seen += d.clusters_seen;
        total.clusters_failed += d.clusters_failed;
        total.superseded += d.superseded;
        total.skipped_distill = total.skipped_distill || d.skipped_distill;
        total.summarized += s.summaries;
        total.max_level = total.max_level.max(s.max_level);
        let (ps, po) = pending_promotable(pool, &scope).await?;
        total.pending_sessions += ps;
        total.pending_solos += po;
        total.pending_undistilled += pending_undistilled(pool, &scope).await?;
    }
    if total.pending_sessions + total.pending_solos > 0 {
        tracing::warn!(
            pending_sessions = total.pending_sessions,
            pending_solos = total.pending_solos,
            "curation: promotable backlog remains after this pass (batch cap); \
             it drains on subsequent passes — rising numbers mean starvation"
        );
    }
    if total.pending_undistilled > 0 {
        tracing::info!(
            pending_undistilled = total.pending_undistilled,
            "curation: episodes not yet distilled — singletons waiting for \
             related evidence are normal; a count that only ever rises while \
             new records arrive means the distill stage is starving"
        );
    }
    Ok(total)
}

/// Episodes in scope still without a `distilled_at` marker — the distill-stage
/// backlog. Includes lone singletons waiting for related evidence, which is
/// why a non-zero value is normal; the trend is the signal.
async fn pending_undistilled(pool: &PgPool, scope: &Scope) -> AppResult<i64> {
    let n: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM curated_nodes n
        WHERE n.kind = 'episodic'
          AND n.user_id = $1
          AND n.mode IS NOT DISTINCT FROM $2
          AND (n.meta->>'distilled_at') IS NULL
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .fetch_one(pool)
    .await?;
    Ok(n)
}

/// Incremental pass for every user with raw records — the scheduler's call.
pub async fn curate_all_users(pool: &PgPool, nlp: &dyn NlpService) -> Vec<(String, CurationStats)> {
    let users: Vec<String> =
        sqlx::query_scalar("SELECT DISTINCT user_id FROM raw_records ORDER BY user_id")
            .fetch_all(pool)
            .await
            .unwrap_or_default();
    let mut out = Vec::new();
    for u in users {
        match curate(pool, nlp, &u).await {
            Ok(stats) => out.push((u, stats)),
            Err(e) => tracing::warn!(user = %u, "curation pass failed: {e}"),
        }
    }
    out
}

/// Drop a scope's summary nodes (and their edges; embeddings cascade) so the
/// tree can be re-derived from the current level 0. Never touches episodic or
/// semantic nodes.
async fn wipe_summaries(pool: &PgPool, scope: &Scope) -> AppResult<()> {
    sqlx::query(
        r#"
        DELETE FROM curated_edges
        WHERE from_id IN (
            SELECT id FROM curated_nodes
            WHERE kind = 'summary' AND user_id = $1
               AND mode IS NOT DISTINCT FROM $2
        )
        OR to_id IN (
            SELECT id FROM curated_nodes
            WHERE kind = 'summary' AND user_id = $1
               AND mode IS NOT DISTINCT FROM $2
        )
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .execute(pool)
    .await?;
    sqlx::query(
        "DELETE FROM curated_nodes WHERE kind = 'summary' AND user_id = $1 \
          AND mode IS NOT DISTINCT FROM $2",
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .execute(pool)
    .await?;
    Ok(())
}

/// How much promotable work is still waiting in a scope: sessions without an
/// episode, and solo rows without a `derived_from` edge. What the batch cap
/// left behind — the number that makes starvation visible instead of silent.
async fn pending_promotable(pool: &PgPool, scope: &Scope) -> AppResult<(i64, i64)> {
    let sessions: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(DISTINCT r.thread_id)
        FROM raw_records r
        WHERE r.type = 'conversation'
          AND r.thread_id IS NOT NULL
          AND r.user_id = $1
          AND (SELECT dm.mode FROM derived_record_mode dm WHERE dm.record_id = r.id) IS NOT DISTINCT FROM $2
          AND NOT EXISTS (SELECT 1 FROM derived_superseded d WHERE d.record_id = r.id)
          AND NOT EXISTS (
              SELECT 1 FROM curated_nodes n
              WHERE n.kind = 'episodic'
                AND n.user_id = $1
                AND n.mode IS NOT DISTINCT FROM $2
                AND n.meta->>'thread_id' = r.thread_id
          )
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .fetch_one(pool)
    .await?;
    let solos: i64 = sqlx::query_scalar(
        r#"
        SELECT COUNT(*)
        FROM raw_records r
        WHERE r.type = ANY($3)
          AND (r.type <> 'conversation' OR r.thread_id IS NULL)
          AND r.user_id = $1
          AND (SELECT dm.mode FROM derived_record_mode dm WHERE dm.record_id = r.id) IS NOT DISTINCT FROM $2
          AND NOT EXISTS (SELECT 1 FROM derived_superseded d WHERE d.record_id = r.id)
          AND NOT EXISTS (
              SELECT 1 FROM curated_edges e
              JOIN curated_nodes n ON n.id = e.from_id
              WHERE e.to_id = r.id AND e.kind = 'derived_from' AND n.kind = 'episodic'
          )
        "#,
    )
    .bind(&scope.user_id)
    .bind(&scope.mode)
    .bind(PROMOTABLE_TYPES.as_slice())
    .fetch_one(pool)
    .await?;
    Ok((sessions, solos))
}

/// Write `supersedes` edges from a newly distilled fact to every ACTIVE
/// semantic fact in scope whose embedding sits within `SUPERSEDE_COSINE` and
/// whose evidence is STRICTLY older — same-evidence siblings from one pass
/// never retire each other. An UNDATED old fact counts as older (retirable):
/// a fact that cannot prove its evidence is newer yields to one that can.
/// Deliberately forward-only: closure happens when the newer fact distills.
/// Backfilled history distilled after a newer fact does NOT retroactively
/// retire it, and same-time facts coexist — the reconciliation pass owns the
/// bidirectional case; this hook handles the live-forward one. Returns how
/// many windows closed.
async fn supersede_older_facts(
    pool: &PgPool,
    scope: &Scope,
    embedder_key: &str,
    new_id: Uuid,
    new_time: DateTime<Utc>,
    dim: usize,
    emb: &[f32],
) -> AppResult<i64> {
    let col = crate::routes::records::embedding_col_for_dim(dim);
    let sql = format!(
        "INSERT INTO curated_edges (from_id, to_id, kind) \
         SELECT $1, n.id, 'supersedes' \
         FROM curated_nodes n \
         JOIN curated_embeddings ce \
           ON ce.node_id = n.id AND ce.model = $7 AND ce.{col} IS NOT NULL \
         WHERE n.kind = 'semantic' \
           AND n.user_id = $2 \
           AND n.mode IS NOT DISTINCT FROM $3 \
           AND n.id <> $1 \
           AND (n.event_time IS NULL OR n.event_time < $4) \
           AND NOT EXISTS ( \
               SELECT 1 FROM curated_edges s \
               WHERE s.to_id = n.id AND s.kind = 'supersedes' \
           ) \
           AND 1 - (ce.{col} <=> $5) >= $6 \
         ON CONFLICT (from_id, to_id, kind) DO NOTHING"
    );
    let res = sqlx::query(sqlx::AssertSqlSafe(sql))
        .bind(new_id)
        .bind(&scope.user_id)
        .bind(&scope.mode)
        .bind(new_time)
        .bind(Vector::from(emb.to_vec()))
        .bind(SUPERSEDE_COSINE)
        .bind(embedder_key)
        .execute(pool)
        .await?;
    Ok(res.rows_affected() as i64)
}

// ---------------------------------------------------------------------------
// rebuild
// ---------------------------------------------------------------------------

/// Wipe the curated layer for a user and re-derive it from raw. Proves the
/// "curation is rebuildable from raw" contract. Deletes scoped by user; raw is
/// never touched. Edges + embeddings cascade from the node delete (embeddings
/// via FK; edges are wiped explicitly since they carry no FK to nodes).
pub async fn rebuild(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
) -> AppResult<CurationStats> {
    // Same single-flight lock as `curate`: a scheduler tick landing mid-wipe
    // would re-promote sessions the wipe removed but the rebuild has not yet
    // re-derived, and the duplicates would then be permanent.
    let Some(mut lock) = try_curation_lock(pool, user_id).await? else {
        tracing::warn!(user = %user_id, "curation already running for this user; rebuild skipped");
        return Ok(CurationStats {
            locked_out: true,
            ..Default::default()
        });
    };
    let result = rebuild_inner(pool, nlp, user_id).await;
    release_curation_lock(&mut lock, user_id).await;
    result
}

/// Re-derive the edges above raw. Called by every rebuild.
///
/// Two of these are pure functions of what writers sent, so they are wiped and
/// rebuilt from scratch — which is what heals a parent that arrived after its
/// child, and what gives a bulk import its links at all.
///
/// The register is not. Which embedding register a record belongs to was an
/// answer some model gave; nothing in the row can reproduce it, so existing rows
/// are left alone and only records with no register at all get the default. A
/// wipe here would destroy classifications permanently.
pub async fn rederive_edges(pool: &PgPool, user_id: &str) -> AppResult<()> {
    let mut tx = pool.begin().await?;

    sqlx::query("DELETE FROM derived_link WHERE user_id = $1")
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    // Only an unambiguous claim links. Two records sharing a source_ref under
    // different sources is a legal corpus, and guessing between them would
    // invent an order the writer never stated.
    sqlx::query(
        r#"
        INSERT INTO derived_link (record_id, user_id, prev_id)
        SELECT r.id, r.user_id, m.prev_id
        FROM raw_records r
        JOIN LATERAL (
            SELECT (array_agg(q.id))[1] AS prev_id, count(*) AS n
            FROM raw_records q
            WHERE q.user_id = r.user_id
              AND q.source_ref = r.prev_source_ref
              AND q.id <> r.id
        ) m ON m.n = 1
        WHERE r.user_id = $1 AND r.prev_source_ref IS NOT NULL
        "#,
    )
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    sqlx::query("DELETE FROM derived_superseded WHERE user_id = $1")
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    sqlx::query(
        r#"
        INSERT INTO derived_superseded (record_id, user_id, superseded_by, origin)
        SELECT r.supersedes, r.user_id, r.id, 'writer'
        FROM raw_records r
        WHERE r.user_id = $1 AND r.supersedes IS NOT NULL
        ON CONFLICT (record_id) DO NOTHING
        "#,
    )
    .bind(user_id)
    .execute(&mut *tx)
    .await?;
    // A state identity's terminal is whichever row arrived last. Deriving it
    // here rather than at write time is also what stops two concurrent writers
    // both believing they are current.
    sqlx::query(
        r#"
        INSERT INTO derived_superseded (record_id, user_id, superseded_by, origin)
        SELECT prev_id, user_id, next_id, 'inferred'
        FROM (
            SELECT r.id AS prev_id, r.user_id,
                   lead(r.id) OVER (
                       PARTITION BY r.user_id, r.payload->>'kind', r.payload->>'key'
                       ORDER BY r.event_time, r.ingest_time, r.seq
                   ) AS next_id
            FROM raw_records r
            WHERE r.user_id = $1
              AND r.type = 'state_object'
              AND r.payload->>'kind' IS NOT NULL
              AND r.payload->>'key' IS NOT NULL
        ) s
        WHERE next_id IS NOT NULL
        ON CONFLICT (record_id) DO NOTHING
        "#,
    )
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    sqlx::query(
        r#"
        INSERT INTO derived_record_mode
            (record_id, user_id, mode, embedder, event_time, origin, decided_by)
        SELECT r.id, r.user_id, m.name, m.embedder, r.event_time, 'default', 'rederive-v1'
        FROM raw_records r
        JOIN modes m ON m.user_id = r.user_id AND m.is_default
        WHERE r.user_id = $1
          AND NOT EXISTS (
              SELECT 1 FROM derived_record_mode dm WHERE dm.record_id = r.id
          )
        ON CONFLICT (record_id) DO NOTHING
        "#,
    )
    .bind(user_id)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(())
}

async fn rebuild_inner(
    pool: &PgPool,
    nlp: &dyn NlpService,
    user_id: &str,
) -> AppResult<CurationStats> {
    let mut tx = pool.begin().await?;

    // Wipe this user's curated nodes (embeddings cascade) and any edges that
    // reference them. Edges are (from_id,to_id,kind) with no FK, so match by the
    // node ids we're about to drop.
    sqlx::query(
        r#"
        DELETE FROM curated_edges
        WHERE from_id IN (SELECT id FROM curated_nodes WHERE user_id = $1)
           OR to_id   IN (SELECT id FROM curated_nodes WHERE user_id = $1)
        "#,
    )
    .bind(user_id)
    .execute(&mut *tx)
    .await?;
    sqlx::query("DELETE FROM curated_nodes WHERE user_id = $1")
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    // Wipe this user's entity index too — promotion repopulates it
    // deterministically from raw, so a rebuild reproduces the same pointer set.
    sqlx::query("DELETE FROM entity_index WHERE user_id = $1")
        .bind(user_id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;

    // Heal the edges above raw before re-deriving the curated layer, so a
    // rebuild sees links and terminals that reflect everything that has arrived.
    rederive_edges(pool, user_id).await?;

    // Re-derive across every mode bucket so a rebuild reproduces the full set.
    let buckets: Vec<(Option<String>,)> = sqlx::query_as(
        r#"
        SELECT DISTINCT dm.mode FROM raw_records r
        LEFT JOIN derived_record_mode dm ON dm.record_id = r.id
        WHERE r.user_id = $1 ORDER BY 1
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;

    let mut total = CurationStats::default();
    for (mode,) in buckets {
        let scope = Scope::new(user_id).with_mode(mode);
        let p = promote_raw_to_episodic(pool, nlp, &scope).await?;
        let d = distill_semantic(pool, nlp, &scope).await?;
        // Build the RAPTOR summary tree over the level-0 nodes just derived for
        // this scope. Scope-bounded and rebuilt from scratch each pass, so the
        // tree is reproducible.
        let s = crate::summaries::build_summaries(pool, nlp, &scope).await?;
        total.promoted += p.promoted;
        total.refreshed += p.refreshed;
        total.distilled += d.distilled;
        total.clusters_seen += d.clusters_seen;
        total.clusters_failed += d.clusters_failed;
        total.superseded += d.superseded;
        total.skipped_distill = total.skipped_distill || d.skipped_distill;
        total.summarized += s.summaries;
        total.max_level = total.max_level.max(s.max_level);
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Derivations (glass-box read): curated nodes derived from a raw id.
// ---------------------------------------------------------------------------

#[derive(Debug, serde::Serialize, sqlx::FromRow)]
pub struct DerivationRow {
    pub id: Uuid,
    pub kind: String,
    pub content: String,
    pub level: i32,
    pub user_id: String,
    pub mode: Option<String>,
    pub importance: Option<f32>,
    pub event_time: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

/// The curated nodes with a `derived_from` edge to this raw id, scoped to the
/// caller (a node's user_id must match — the raw record itself is authorized by
/// the caller before this runs).
pub async fn derivations_of(
    pool: &PgPool,
    user_id: &str,
    raw_id: Uuid,
) -> AppResult<Vec<DerivationRow>> {
    let rows = sqlx::query_as::<_, DerivationRow>(
        r#"
        SELECT n.id, n.kind, n.content, n.level, n.user_id, n.mode,
               n.importance, n.event_time, n.created_at
        FROM curated_nodes n
        JOIN curated_edges e ON e.from_id = n.id AND e.kind = 'derived_from'
        WHERE e.to_id = $1 AND n.user_id = $2
        ORDER BY n.created_at ASC
        "#,
    )
    .bind(raw_id)
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

// ---------------------------------------------------------------------------
// Shared inserts
// ---------------------------------------------------------------------------

async fn insert_node(
    pool: &PgPool,
    id: Uuid,
    kind: &str,
    content: &str,
    scope: &Scope,
    importance: Option<f32>,
    event_time: Option<DateTime<Utc>>,
) -> AppResult<()> {
    edges::insert_node_at_level(pool, id, kind, content, 0, scope, importance, event_time).await
}

use edges::add_edge;
use edges::add_entity_edge;
use edges::embed_node;

/// Shared curated-layer writers, `pub(crate)` so the summaries module can build
/// higher levels of the same node/edge/embedding tables without duplicating the
/// insert idioms (trusted `kind` constant, `AssertSqlSafe`-free bound params,
/// best-effort embed).
pub(crate) mod edges {
    use super::{AppResult, NlpService, PgPool, Scope};
    use chrono::{DateTime, Utc};
    use pgvector::Vector;
    use uuid::Uuid;

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn insert_node_at_level(
        pool: &PgPool,
        id: Uuid,
        kind: &str,
        content: &str,
        level: i32,
        scope: &Scope,
        importance: Option<f32>,
        event_time: Option<DateTime<Utc>>,
    ) -> AppResult<()> {
        // `kind` is a trusted constant from this crate's call sites, never user
        // input; content/level/scope are bound params.
        sqlx::query(
            r#"
            INSERT INTO curated_nodes
                (id, kind, content, level, user_id, mode, importance, event_time)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
        )
        .bind(id)
        .bind(kind)
        .bind(content)
        .bind(level)
        .bind(&scope.user_id)
        .bind(&scope.mode)
        .bind(importance)
        .bind(event_time)
        .execute(pool)
        .await?;
        Ok(())
    }

    pub(crate) async fn add_edge(
        pool: &PgPool,
        from_id: Uuid,
        to_id: Uuid,
        kind: &str,
    ) -> AppResult<()> {
        sqlx::query(
            "INSERT INTO curated_edges (from_id, to_id, kind) VALUES ($1, $2, $3) \
             ON CONFLICT (from_id, to_id, kind) DO NOTHING",
        )
        .bind(from_id)
        .bind(to_id)
        .bind(kind)
        .execute(pool)
        .await?;
        Ok(())
    }

    /// An `entity` edge (curated node -> raw id) carrying the entity string in
    /// `label`. The edge PK is (from_id, to_id, kind) so one node/raw pair holds
    /// a single 'entity' edge; the label records the last entity that tied them
    /// (the scan-friendly per-entity fanout lives in `entity_index`). Idempotent.
    pub(crate) async fn add_entity_edge(
        pool: &PgPool,
        from_id: Uuid,
        to_id: Uuid,
        entity: &str,
    ) -> AppResult<()> {
        sqlx::query(
            "INSERT INTO curated_edges (from_id, to_id, kind, label) VALUES ($1, $2, 'entity', $3) \
             ON CONFLICT (from_id, to_id, kind) DO UPDATE SET label = EXCLUDED.label",
        )
        .bind(from_id)
        .bind(to_id)
        .bind(entity)
        .execute(pool)
        .await?;
        Ok(())
    }

    /// Best-effort embed — mirrors the raw ingest contract. A failed embed never
    /// fails curation; a backfill can fill the gap. The node still exists. The
    /// vector is written in the scope's mode geometry (its embedder + dimension),
    /// into the matching `embedding_<dim>` column, so a curated node never mixes
    /// registers with a raw record in a different mode.
    pub(crate) async fn embed_node(
        pool: &PgPool,
        nlp: &dyn NlpService,
        scope: &Scope,
        node_id: Uuid,
        content: &str,
    ) -> Option<(usize, Vec<f32>)> {
        let embedder_key = scope.embedder_key(pool).await;
        match nlp.embed_for_mode(&embedder_key, content).await {
            Ok((dim, embedding)) => {
                let col = crate::routes::records::embedding_col_for_dim(dim);
                let sql = format!(
                    "INSERT INTO curated_embeddings (node_id, model, {col}) VALUES ($1, $2, $3) \
                     ON CONFLICT (node_id, model) DO NOTHING"
                );
                // col is a trusted constant from embedding_col_for_dim; values bound.
                let _ = sqlx::query(sqlx::AssertSqlSafe(sql))
                    .bind(node_id)
                    .bind(&embedder_key)
                    .bind(Vector::from(embedding.clone()))
                    .execute(pool)
                    .await;
                Some((dim, embedding))
            }
            Err(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Clustering: embedding-centroid first, entity overlap as the fallback.
//
// The retired greedy single-link entity-Jaccard had two failure modes this
// exists to kill: transitive CHAINING (a~b, b~c merged a and c even when they
// shared nothing) and the lexical miss (same topic, different words, zero
// shared entities — the exact case embeddings are stored for). Assignment is
// against a cluster's CENTROID, so drift is bounded and chains cannot form;
// the size cap bounds the distill prompt and stops hub topics swallowing the
// batch. Deterministic for a given input order.
// ---------------------------------------------------------------------------

pub(crate) fn l2_normalize(v: &[f32]) -> Option<Vec<f32>> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm > 0.0).then(|| v.iter().map(|x| x / norm).collect())
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

struct Cluster {
    members: Vec<usize>,
    /// Running mean of the EMBEDDED members' normalized embeddings; None
    /// until one joins. Renormalized at comparison time.
    centroid: Option<Vec<f32>>,
    /// How many members contributed to the centroid — the mean's denominator.
    embedded: usize,
    /// The seed member's index — the entity-fallback comparison surface.
    seed: usize,
}

/// Cluster items by cosine-to-centroid over normalized `embeddings`, falling
/// back to entity Jaccard against the cluster seed when either side has no
/// embedding. Returns index groups, each at most `max_size` long.

/// Distil one cluster of episodes into semantic facts. Extracted so the caller
/// reads as "load episodes, cluster them, distil each" — the loop body is the
/// only part that talks to the model and writes nodes, and it was burying that
/// shape under a hundred lines.
async fn distill_one_cluster(
    pool: &PgPool,
    nlp: &dyn NlpService,
    scope: &Scope,
    cluster: &[usize],
    episodes: &[EpisodeNode],
    embedder_key: &str,
    stats: &mut CurationStats,
) -> AppResult<()> {
    // One EpisodeRef per EPISODE — the transcript, the unioned entities,
    // and when it happened, so the distiller can weigh recency.
    let refs: Vec<EpisodeRef> = cluster
        .iter()
        .map(|&i| EpisodeRef {
            id: episodes[i].node_id,
            content: episodes[i].content.clone(),
            topic: None,
            entities: episodes[i].entities.clone(),
            when: episodes[i].when(),
        })
        .collect();

    let facts = match nlp.distill_facts(&refs).await {
        Ok(f) => f,
        Err(first) => match nlp.distill_facts(&refs).await {
            Ok(f) => f,
            Err(e) => {
                // NOT marked as distilled — a failed cluster is retried
                // next pass, and the failure is a counted stat, not a
                // silently dropped topic.
                stats.clusters_failed += 1;
                tracing::warn!(
                    "curation: distillation failed on cluster twice ({first}; then {e})"
                );
                return Ok(());
            }
        },
    };

    // The provider answered, so this cluster is consumed — including a
    // legitimate empty answer ("too noisy to distill"). The marker is what
    // makes incremental passes append-only instead of re-distilling every
    // episode forever; a session that later grows has its meta rewritten,
    // which clears it. Singletons never reach here and stay eligible, so a
    // future related episode can still pull them into a cluster.
    let cluster_node_ids: Vec<Uuid> = cluster
        .iter()
        .filter(|&&i| !episodes[i].already_distilled)
        .map(|&i| episodes[i].node_id)
        .collect();
    sqlx::query(
        "UPDATE curated_nodes \
         SET meta = COALESCE(meta, '{}'::jsonb) || jsonb_build_object('distilled_at', NOW()) \
         WHERE id = ANY($1)",
    )
    .bind(&cluster_node_ids)
    .execute(pool)
    .await?;
    if facts.is_empty() {
        return Ok(());
    }

    let in_cluster: HashMap<Uuid, &EpisodeNode> = cluster
        .iter()
        .map(|&i| (episodes[i].node_id, &episodes[i]))
        .collect();

    for fact in facts {
        // Evidence-granularity lineage: the distiller cites the episodes a
        // fact came from, and the fact's edges point at THOSE episodes' raw
        // rows. A fact citing nothing valid falls back to the whole cluster
        // rather than dropping lineage.
        let cited: Vec<&EpisodeNode> = fact
            .source_episode_ids
            .iter()
            .filter_map(|id| in_cluster.get(id).copied())
            .collect();
        let evidence: Vec<&EpisodeNode> = if cited.is_empty() {
            in_cluster.values().copied().collect()
        } else {
            cited
        };
        let raw_ids: Vec<Uuid> = evidence
            .iter()
            .flat_map(|e| e.source_raw_ids.iter().copied())
            .collect();
        // A fact is dated by the newest evidence that supports it — never
        // by the moment curation happened to run, which made every old
        // fact look freshly minted to recency ranking.
        let fact_time = evidence.iter().filter_map(|e| e.ended.or(e.started)).max();

        let node_id = Uuid::new_v4();
        insert_node(
            pool,
            node_id,
            "semantic",
            &fact.content,
            scope,
            None,
            fact_time,
        )
        .await?;
        for raw_id in &raw_ids {
            add_edge(pool, node_id, *raw_id, "derived_from").await?;
        }
        // Episode-level lineage beside the raw-level edges: which curated
        // episodes this fact was distilled from, as the distiller cited.
        for ep in &evidence {
            add_edge(pool, node_id, ep.node_id, "distilled_from").await?;
        }
        let fact_emb = embed_node(pool, nlp, scope, node_id, &fact.content).await;
        // Temporal invalidation: the newer fact closes the validity window
        // of any near-same-topic older fact. A closed window, never a
        // delete — every filter already excludes superseded nodes, and a
        // rebuild reproduces the same closures from raw.
        if let (Some(t), Some((fdim, femb))) = (fact_time, fact_emb.as_ref()) {
            stats.superseded +=
                supersede_older_facts(pool, scope, embedder_key, node_id, t, *fdim, femb).await?;
        }
        stats.distilled += 1;
    }

    Ok(())
}

pub(crate) fn cluster_for_distill(
    embeddings: &[Option<Vec<f32>>],
    entity_sets: &[&[String]],
    cosine_threshold: f32,
    jaccard_threshold: f32,
    max_size: usize,
) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Cluster> = Vec::new();
    for i in 0..entity_sets.len() {
        match best_cluster_for(
            i,
            &clusters,
            embeddings,
            entity_sets,
            cosine_threshold,
            jaccard_threshold,
            max_size,
        ) {
            Some(ci) => absorb_into_cluster(&mut clusters[ci], i, &embeddings[i]),
            None => clusters.push(Cluster {
                embedded: usize::from(embeddings[i].is_some()),
                members: vec![i],
                centroid: embeddings[i].clone(),
                seed: i,
            }),
        }
    }
    clusters.into_iter().map(|c| c.members).collect()
}

/// The best cluster item `i` could join, or None to seed its own. Cosine against
/// the running centroid when both sides are embedded; entity overlap against the
/// seed otherwise, so a record whose embedding failed still clusters.
#[allow(clippy::too_many_arguments)]
fn best_cluster_for(
    i: usize,
    clusters: &[Cluster],
    embeddings: &[Option<Vec<f32>>],
    entity_sets: &[&[String]],
    cosine_threshold: f32,
    jaccard_threshold: f32,
    max_size: usize,
) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (ci, c) in clusters.iter().enumerate() {
        if c.members.len() >= max_size {
            continue;
        }
        let score = match (&embeddings[i], &c.centroid) {
            (Some(e), Some(cen)) => l2_normalize(cen)
                .map(|n| dot(e, &n))
                .filter(|cos| *cos >= cosine_threshold),
            _ => Some(jaccard(entity_sets[i], entity_sets[c.seed]))
                .filter(|j| *j >= jaccard_threshold),
        };
        if let Some(sc) = score {
            if best.is_none_or(|(_, b)| sc > b) {
                best = Some((ci, sc));
            }
        }
    }
    best.map(|(ci, _)| ci)
}

/// Add the item and fold its embedding into the running mean. Only embedded
/// members move the centroid, so `embedded` counts those rather than all members.
fn absorb_into_cluster(c: &mut Cluster, i: usize, embedding: &Option<Vec<f32>>) {
    if let Some(e) = embedding {
        c.centroid = Some(match c.centroid.take() {
            Some(mut cen) => {
                let k = c.embedded as f32;
                for (m, x) in cen.iter_mut().zip(e) {
                    *m = (*m * k + x) / (k + 1.0);
                }
                cen
            }
            None => e.clone(),
        });
        c.embedded += 1;
    }
    c.members.push(i);
}

fn cluster_by_entities(episodes: &[EpisodeNode]) -> Vec<Vec<usize>> {
    let embeddings: Vec<Option<Vec<f32>>> = episodes
        .iter()
        .map(|e| e.embedding.as_deref().and_then(l2_normalize))
        .collect();
    let entity_sets: Vec<&[String]> = episodes.iter().map(|e| e.entities.as_slice()).collect();
    cluster_for_distill(
        &embeddings,
        &entity_sets,
        CLUSTER_COSINE,
        CLUSTER_JACCARD,
        CLUSTER_MAX,
    )
}

pub(crate) fn jaccard(a: &[String], b: &[String]) -> f32 {
    let sa: HashSet<&str> = a.iter().map(String::as_str).collect();
    let sb: HashSet<&str> = b.iter().map(String::as_str).collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Test embedder: a hashed bag-of-words vector, so cosine reflects lexical
/// overlap the way a real embedder's reflects semantic overlap. Same-topic
/// test strings cluster; disjoint-topic strings don't — which is the geometry
/// the clustering tests exist to exercise.
#[cfg(test)]
pub(crate) fn bow_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0_f32; 384];
    for w in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        if w.len() < 3 {
            continue;
        }
        let mut h = 5381_u32;
        for b in w.bytes() {
            h = h.wrapping_mul(33) ^ b as u32;
        }
        v[(h % 384) as usize] += 1.0;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AppError;
    use async_trait::async_trait;
    use flashback_nlp::{DistilledFact, Extraction, ProviderError};

    /// Serializes the one test that mutates the process-global batch-cap env var.
    static BATCH_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // -- jaccard / clustering (pure) --------------------------------------

    #[test]
    fn jaccard_known_values() {
        assert_eq!(jaccard(&[], &[]), 0.0);
        let a = ["x".to_string(), "y".to_string()];
        assert!((jaccard(&a, &a) - 1.0).abs() < 1e-6);
        let b = ["y".to_string(), "z".to_string()];
        assert!((jaccard(&a, &b) - (1.0 / 3.0)).abs() < 1e-6);
    }

    fn ep(entities: &[&str]) -> EpisodeNode {
        EpisodeNode {
            node_id: Uuid::new_v4(),
            source_raw_ids: vec![Uuid::new_v4()],
            content: String::new(),
            entities: entities.iter().map(|s| s.to_string()).collect(),
            embedding: None,
            already_distilled: false,
            started: None,
            ended: None,
        }
    }

    fn nvec(v: &[f32]) -> Option<Vec<f32>> {
        l2_normalize(v)
    }

    #[test]
    fn centroid_clustering_kills_the_single_link_chain() {
        // a~b and b~c, but a and c share nothing. Single-link merged all
        // three; against the a+b centroid, c falls short and stays out.
        let e: Vec<Option<Vec<f32>>> =
            vec![nvec(&[1.0, 0.0]), nvec(&[0.8, 0.6]), nvec(&[0.0, 1.0])];
        let ents: Vec<&[String]> = vec![&[], &[], &[]];
        let clusters = cluster_for_distill(&e, &ents, 0.6, 0.4, 12);
        assert!(clusters.iter().any(|c| c == &vec![0, 1]));
        assert!(clusters.iter().any(|c| c == &vec![2]), "{clusters:?}");
    }

    #[test]
    fn embeddings_cluster_what_entities_miss() {
        // Same topic, zero shared entity strings — the lexical miss the old
        // clusterer could never bridge. Embeddings carry it.
        let e: Vec<Option<Vec<f32>>> = vec![nvec(&[0.9, 0.1]), nvec(&[0.85, 0.15])];
        let a = ["knee".to_string()];
        let b = ["joint pain".to_string()];
        let ents: Vec<&[String]> = vec![&a, &b];
        let clusters = cluster_for_distill(&e, &ents, 0.6, 0.4, 12);
        assert_eq!(clusters, vec![vec![0, 1]]);
    }

    #[test]
    fn cluster_size_cap_bounds_the_distill_prompt() {
        let e: Vec<Option<Vec<f32>>> = (0..5).map(|_| nvec(&[1.0, 0.0])).collect();
        let empty: [String; 0] = [];
        let ents: Vec<&[String]> = (0..5).map(|_| empty.as_slice()).collect();
        let clusters = cluster_for_distill(&e, &ents, 0.6, 0.4, 2);
        assert!(clusters.iter().all(|c| c.len() <= 2), "{clusters:?}");
        assert_eq!(clusters.iter().map(Vec::len).sum::<usize>(), 5);
    }

    #[test]
    fn missing_embeddings_fall_back_to_entity_overlap() {
        let e: Vec<Option<Vec<f32>>> = vec![None, None, None];
        let a = ["x".to_string(), "y".to_string()];
        let b = ["x".to_string(), "y".to_string(), "z".to_string()];
        let c = ["p".to_string()];
        let ents: Vec<&[String]> = vec![&a, &b, &c];
        let clusters = cluster_for_distill(&e, &ents, 0.6, 0.4, 12);
        assert!(clusters.iter().any(|cl| cl == &vec![0, 1]));
        assert!(clusters.iter().any(|cl| cl == &vec![2]));
    }

    #[test]
    fn cluster_groups_overlapping_singletons_apart() {
        let eps = vec![ep(&["a", "b"]), ep(&["a", "b", "c"]), ep(&["p", "q"])];
        let clusters = cluster_by_entities(&eps);
        // First two share jaccard 2/3 ≥ 0.4 → one cluster; third is alone.
        let big = clusters.iter().find(|c| c.len() == 2).unwrap();
        assert!(big.contains(&0) && big.contains(&1));
        assert!(clusters.iter().any(|c| c == &vec![2]));
    }

    // -- test doubles ------------------------------------------------------

    /// Provider stub that CAN distill: it returns a single fact whose sources
    /// are exactly the raw ids handed in. Lets the semantic path run without a
    /// real LLM. Entities are extracted via the real heuristic so clustering is
    /// exercised for real.
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
        async fn embed_one(&self, text: &str) -> Result<Vec<f32>, AppError> {
            Ok(bow_embed(text))
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok(texts.iter().map(|t| bow_embed(t)).collect())
        }
        fn extract_entities(&self, text: &str) -> Vec<String> {
            flashback_nlp::extract_entities(text)
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            episodes: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Ok(vec![DistilledFact {
                content: "distilled fact".to_string(),
                topic: None,
                source_episode_ids: episodes.iter().map(|e| e.id).collect(),
                confidence: 0.9,
            }])
        }
    }

    /// Distilling stub that records the exact `EpisodeRef`s handed to it, so a
    /// test can assert what distillation actually saw — the regression here was
    /// an episode arriving as one arbitrary raw turn instead of its transcript.
    #[derive(Clone, Default)]
    struct CapturingNlp {
        seen: std::sync::Arc<std::sync::Mutex<Vec<EpisodeRef>>>,
    }

    #[async_trait]
    impl NlpService for CapturingNlp {
        fn provider_name(&self) -> &'static str {
            "test-capture"
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
        async fn embed_one(&self, text: &str) -> Result<Vec<f32>, AppError> {
            Ok(bow_embed(text))
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok(texts.iter().map(|t| bow_embed(t)).collect())
        }
        fn extract_entities(&self, text: &str) -> Vec<String> {
            flashback_nlp::extract_entities(text)
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            episodes: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            self.seen.lock().unwrap().extend(episodes.iter().cloned());
            Ok(vec![DistilledFact {
                content: "distilled fact".to_string(),
                topic: None,
                source_episode_ids: episodes.iter().map(|e| e.id).collect(),
                confidence: 0.9,
            }])
        }
    }

    /// Heuristic-style stub that CANNOT distill (models the no-LLM deployment).
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
        async fn embed_one(&self, text: &str) -> Result<Vec<f32>, AppError> {
            Ok(bow_embed(text))
        }
        async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
            Ok(texts.iter().map(|t| bow_embed(t)).collect())
        }
        fn extract_entities(&self, text: &str) -> Vec<String> {
            flashback_nlp::extract_entities(text)
        }
        async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
            Ok(Extraction::empty())
        }
        async fn distill_facts(
            &self,
            _episodes: &[EpisodeRef],
        ) -> Result<Vec<DistilledFact>, ProviderError> {
            Err(ProviderError::NotConfigured("heuristic".into()))
        }
    }

    // -- DB helpers --------------------------------------------------------

    /// Insert a standalone raw record directly (bypassing the door — we don't
    /// need the door here, and raw is append-only so a plain INSERT is fine).
    /// `document` is the one-record-one-episode shape.
    async fn insert_document_raw(
        pool: &PgPool,
        user_id: &str,
        topic_id: Option<&str>,
        mode: Option<&str>,
        content: &str,
    ) -> Uuid {
        let id = Uuid::new_v4();
        sqlx::query(
            r#"INSERT INTO raw_records
               (id, type, content, event_time, source, user_id, topic_id)
               VALUES ($1, 'document', $2, NOW(), 'test', $3, $4)"#,
        )
        .bind(id)
        .bind(content)
        .bind(user_id)
        .bind(topic_id)
        .execute(pool)
        .await
        .unwrap();
        if let Some(m) = mode {
            crate::modes::ensure_builtin_modes(pool, user_id).await.ok();
            sqlx::query(
                "INSERT INTO derived_record_mode
                    (record_id, user_id, mode, embedder, event_time, origin, decided_by)
                 SELECT $1, $2, $3, m.embedder, NOW(), 'writer', 'test'
                 FROM modes m WHERE m.user_id = $2 AND m.name = $3",
            )
            .bind(id)
            .bind(user_id)
            .bind(m)
            .execute(pool)
            .await
            .unwrap();
        }
        id
    }

    /// A standalone document dated `days_ago` in the past — for temporal tests.
    async fn insert_document_raw_at(
        pool: &PgPool,
        user_id: &str,
        content: &str,
        days_ago: i64,
    ) -> Uuid {
        let id = Uuid::new_v4();
        sqlx::query(
            r#"INSERT INTO raw_records
               (id, type, content, event_time, source, user_id)
               VALUES ($1, 'document', $2, NOW() - ($3 || ' days')::interval, 'test', $4)"#,
        )
        .bind(id)
        .bind(content)
        .bind(days_ago.to_string())
        .bind(user_id)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    /// One conversation turn. `offset_secs` orders turns inside a session so a
    /// test can assert the transcript came out in the order it was said.
    async fn insert_turn(
        pool: &PgPool,
        user_id: &str,
        thread_id: &str,
        source: &str,
        content: &str,
        offset_secs: i64,
    ) -> Uuid {
        let id = Uuid::new_v4();
        sqlx::query(
            r#"INSERT INTO raw_records
               (id, type, content, event_time, source, user_id, thread_id)
               VALUES ($1, 'conversation', $2, NOW() + ($3 || ' seconds')::interval, $4, $5, $6)"#,
        )
        .bind(id)
        .bind(content)
        .bind(offset_secs.to_string())
        .bind(source)
        .bind(user_id)
        .bind(thread_id)
        .execute(pool)
        .await
        .unwrap();
        id
    }

    async fn node_meta(pool: &PgPool, user_id: &str) -> serde_json::Value {
        sqlx::query_scalar::<_, Option<serde_json::Value>>(
            "SELECT meta FROM curated_nodes WHERE user_id = $1 AND kind = 'episodic' LIMIT 1",
        )
        .bind(user_id)
        .fetch_one(pool)
        .await
        .unwrap()
        .unwrap_or(serde_json::Value::Null)
    }

    async fn count_nodes(pool: &PgPool, user_id: &str, kind: &str) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM curated_nodes WHERE user_id = $1 AND kind = $2")
            .bind(user_id)
            .bind(kind)
            .fetch_one(pool)
            .await
            .unwrap()
    }

    // -- promote -----------------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn promote_document_creates_one_episodic_with_edge(pool: PgPool) {
        let raw_id = insert_document_raw(&pool, "alice", None, None, "took 5mg lisinopril").await;
        let scope = Scope::new("alice");

        let stats = promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        assert_eq!(stats.promoted, 1);
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 1);

        // The derived_from edge points node -> raw id.
        let edge: (Uuid, Uuid) =
            sqlx::query_as("SELECT from_id, to_id FROM curated_edges WHERE kind = 'derived_from'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(edge.1, raw_id);

        // The episodic node was embedded.
        let n_emb: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_embeddings e JOIN curated_nodes n ON n.id = e.node_id \
             WHERE n.kind = 'episodic'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(n_emb, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn promote_is_idempotent(pool: PgPool) {
        insert_document_raw(&pool, "alice", None, None, "took 5mg lisinopril").await;
        let scope = Scope::new("alice");

        let first = promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        assert_eq!(first.promoted, 1);

        // Re-run: the derived_from edge already exists → nothing new.
        let second = promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        assert_eq!(second.promoted, 0);
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 1);
    }

    /// The unit of an episode is the conversation, not the turn: four turns in
    /// one session collapse to a single node, not four.
    #[sqlx::test(migrations = "../../migrations")]
    async fn a_conversation_becomes_one_episode_not_one_per_turn(pool: PgPool) {
        let a = insert_turn(
            &pool,
            "alice",
            "conv-1",
            "host:helper:user",
            "what's due friday",
            0,
        )
        .await;
        let b = insert_turn(
            &pool,
            "alice",
            "conv-1",
            "host:helper:assistant",
            "the quarterly report",
            1,
        )
        .await;
        let c = insert_turn(
            &pool,
            "alice",
            "conv-1",
            "host:helper:user",
            "and after that",
            2,
        )
        .await;
        let d = insert_turn(
            &pool,
            "alice",
            "conv-1",
            "host:helper:assistant",
            "the tax filing",
            3,
        )
        .await;

        let scope = Scope::new("alice");
        let stats = promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();

        assert_eq!(stats.promoted, 1);
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 1);

        // Lineage still resolves to every individual turn.
        let edges: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_edges WHERE kind = 'derived_from' AND to_id = ANY($1)",
        )
        .bind(vec![a, b, c, d].as_slice())
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(edges, 4);

        let meta = node_meta(&pool, "alice").await;
        assert_eq!(meta["thread_id"], "conv-1");
        assert_eq!(meta["turns"], 4);
        assert_eq!(meta["title"], "what's due friday");

        // The transcript is ordered and speaker-labelled from the turns' own text.
        let content: String = sqlx::query_scalar(
            "SELECT content FROM curated_nodes WHERE user_id = 'alice' AND kind = 'episodic'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(content.starts_with("user: what's due friday"));
        assert!(content.contains("assistant: the quarterly report"));
        assert!(content.find("and after that").unwrap() < content.find("the tax filing").unwrap());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn separate_sessions_stay_separate_episodes(pool: PgPool) {
        insert_turn(&pool, "alice", "conv-1", "chatgpt", "about the kernel", 0).await;
        insert_turn(&pool, "alice", "conv-2", "chatgpt", "about the taxes", 0).await;

        let scope = Scope::new("alice");
        let stats = promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();

        assert_eq!(stats.promoted, 2);
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 2);
    }

    /// Re-running promotion must not mint a second episode for a session that
    /// already has one — the session, not the turn, is the idempotency key.
    #[sqlx::test(migrations = "../../migrations")]
    async fn conversation_promotion_is_idempotent_per_session(pool: PgPool) {
        insert_turn(&pool, "alice", "conv-1", "host:helper:user", "first", 0).await;
        insert_turn(
            &pool,
            "alice",
            "conv-1",
            "host:helper:assistant",
            "second",
            1,
        )
        .await;
        let scope = Scope::new("alice");

        promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        let again = promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();

        assert_eq!(again.promoted, 0);
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 1);
    }

    /// An import that carried the conversation's own title uses it rather than
    /// falling back to the opening line.
    #[sqlx::test(migrations = "../../migrations")]
    async fn imported_title_wins_over_the_opening_turn(pool: PgPool) {
        sqlx::query(
            r#"INSERT INTO raw_records
               (id, type, content, event_time, source, user_id, thread_id, payload)
               VALUES ($1,'conversation','hey can you help',NOW(),'chatgpt','alice','conv-9',$2)"#,
        )
        .bind(Uuid::new_v4())
        .bind(serde_json::json!({ "conversation_title": "FPE export planning" }))
        .execute(&pool)
        .await
        .unwrap();

        promote_raw_to_episodic(&pool, &DistillingNlp, &Scope::new("alice"))
            .await
            .unwrap();
        assert_eq!(
            node_meta(&pool, "alice").await["title"],
            "FPE export planning"
        );
    }

    #[test]
    fn transcript_and_title_are_extractive_and_bounded() {
        let long = "x".repeat(TITLE_BUDGET + 40);
        assert_eq!(
            truncate_chars(&long, TITLE_BUDGET).chars().count(),
            TITLE_BUDGET + 1 // + the ellipsis
        );
        // Multi-byte input must not panic or split a character.
        assert_eq!(truncate_chars("héllo wörld", 4), "héll…");
        assert_eq!(speaker_of("host:agent-alpha:user"), "user");
        assert_eq!(speaker_of("chatgpt"), "chatgpt");
    }

    #[test]
    fn curation_batch_cap_defaults_and_rejects_bad_env() {
        // Serialize the process-global env mutation and restore it after.
        let _lock = BATCH_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = std::env::var("FLASHBACK_CURATION_BATCH").ok();

        // SAFETY: this test holds BATCH_ENV_LOCK for the whole mutation window.
        unsafe { std::env::remove_var("FLASHBACK_CURATION_BATCH") };
        assert_eq!(curation_batch_cap(), 5_000); // default when unset

        // A positive override is honored; zero/negative/garbage fall back.
        unsafe { std::env::set_var("FLASHBACK_CURATION_BATCH", "10") };
        assert_eq!(curation_batch_cap(), 10);
        for bad in ["0", "-4", "nan"] {
            unsafe { std::env::set_var("FLASHBACK_CURATION_BATCH", bad) };
            assert_eq!(curation_batch_cap(), 5_000, "'{bad}' must fall back");
        }

        // Restore the prior value so no other test observes our mutation.
        match prev {
            Some(v) => unsafe { std::env::set_var("FLASHBACK_CURATION_BATCH", v) },
            None => unsafe { std::env::remove_var("FLASHBACK_CURATION_BATCH") },
        }
    }

    // -- distill -----------------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn distill_entity_overlap_makes_one_semantic_with_all_sources(pool: PgPool) {
        // Two working records that extract the same entity set ({deploy target,
        // pgvector service}) → jaccard 1.0 → one cluster.
        let r1 = insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        let r2 = insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;
        let scope = Scope::new("alice");

        promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        let stats = distill_semantic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();

        assert!(!stats.skipped_distill);
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, 1);
        assert_eq!(stats.distilled, 1);

        // The semantic node has derived_from edges to BOTH source raw ids.
        let semantic_id: Uuid =
            sqlx::query_scalar("SELECT id FROM curated_nodes WHERE kind = 'semantic'")
                .fetch_one(&pool)
                .await
                .unwrap();
        let mut targets: Vec<Uuid> = sqlx::query_scalar(
            "SELECT to_id FROM curated_edges WHERE from_id = $1 AND kind = 'derived_from'",
        )
        .bind(semantic_id)
        .fetch_all(&pool)
        .await
        .unwrap();
        targets.sort();
        let mut want = vec![r1, r2];
        want.sort();
        assert_eq!(targets, want);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn distill_sees_whole_transcripts_spans_and_dates_the_fact(pool: PgPool) {
        // Two sessions on the same topic, two turns each. The distiller must
        // receive one ref per EPISODE carrying the transcript and the session
        // span — not one arbitrary turn per session, which is what a
        // join-ordering accident used to hand it.
        let s1a = insert_turn(
            &pool,
            "alice",
            "s1",
            "host:agent:user",
            "the deploy target for the pgvector service is staging",
            0,
        )
        .await;
        let s1b = insert_turn(
            &pool,
            "alice",
            "s1",
            "host:agent:assistant",
            "confirmed, the deploy target for the pgvector service stays staging",
            60,
        )
        .await;
        let s2a = insert_turn(
            &pool,
            "alice",
            "s2",
            "host:agent:user",
            "the deploy target for the pgvector service moved to production",
            0,
        )
        .await;
        let s2b = insert_turn(
            &pool,
            "alice",
            "s2",
            "host:agent:assistant",
            "updating the deploy target for the pgvector service to production",
            60,
        )
        .await;
        let scope = Scope::new("alice");

        let nlp = CapturingNlp::default();
        promote_raw_to_episodic(&pool, &nlp, &scope).await.unwrap();
        let stats = distill_semantic(&pool, &nlp, &scope).await.unwrap();
        assert_eq!(stats.distilled, 1);

        let refs = nlp.seen.lock().unwrap().clone();
        assert_eq!(refs.len(), 2, "one ref per episode, not per raw turn");
        for r in &refs {
            assert!(
                r.content.contains("user:"),
                "transcript, not one turn: {}",
                r.content
            );
            assert!(
                r.content.contains("assistant:"),
                "both speakers: {}",
                r.content
            );
            let when = r.when.as_deref().expect("episode carries when");
            assert!(when.contains(".."), "a session spans time, got {when}");
        }

        // The fact is dated by its newest evidence, never by the curation run,
        // and its lineage reaches every cited episode's raw rows.
        let (event_time, semantic_id): (Option<DateTime<Utc>>, Uuid) =
            sqlx::query_as("SELECT event_time, id FROM curated_nodes WHERE kind = 'semantic'")
                .fetch_one(&pool)
                .await
                .unwrap();
        assert!(
            event_time.is_some(),
            "semantic fact must carry evidence time"
        );
        let mut targets: Vec<Uuid> = sqlx::query_scalar(
            "SELECT to_id FROM curated_edges WHERE from_id = $1 AND kind = 'derived_from'",
        )
        .bind(semantic_id)
        .fetch_all(&pool)
        .await
        .unwrap();
        targets.sort();
        let mut want = vec![s1a, s1b, s2a, s2b];
        want.sort();
        assert_eq!(targets, want);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn distill_skips_gracefully_without_capability(pool: PgPool) {
        insert_document_raw(&pool, "alice", None, None, "the deploy target is staging").await;
        insert_document_raw(&pool, "alice", None, None, "switched the deploy target now").await;
        let scope = Scope::new("alice");

        promote_raw_to_episodic(&pool, &HeuristicNlp, &scope)
            .await
            .unwrap();
        let stats = distill_semantic(&pool, &HeuristicNlp, &scope)
            .await
            .unwrap();

        assert!(stats.skipped_distill);
        assert_eq!(stats.distilled, 0);
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, 0);
    }

    // -- curate (the incremental hot path) ---------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn curate_twice_does_no_new_work_and_never_duplicates(pool: PgPool) {
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;

        let first = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(first.promoted, 2);
        assert_eq!(first.distilled, 1);
        assert_eq!(first.pending_sessions + first.pending_solos, 0);

        // Same call again with nothing new: no promotion, no distillation, no
        // duplicate facts — the regression was every pass re-distilling every
        // episode forever.
        let second = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(second.promoted, 0);
        assert_eq!(second.refreshed, 0);
        assert_eq!(second.distilled, 0);
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, 1);

        // New same-topic records arrive: only THEY get distilled, appending a
        // second fact while the first survives untouched.
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "note 8: the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "note 9: the deploy target for the pgvector service is staging",
        )
        .await;
        let third = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(third.promoted, 2);
        assert_eq!(third.distilled, 1);
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, 2);
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 4);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn grown_session_refreshes_in_place_and_becomes_distillable_again(pool: PgPool) {
        insert_turn(
            &pool,
            "alice",
            "s1",
            "host:agent:user",
            "first turn about the plan",
            0,
        )
        .await;
        insert_turn(
            &pool,
            "alice",
            "s1",
            "host:agent:assistant",
            "noted the plan",
            60,
        )
        .await;
        let first = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(first.promoted, 1);
        let (node_id, meta): (Uuid, serde_json::Value) = sqlx::query_as(
            "SELECT id, meta FROM curated_nodes WHERE kind = 'episodic' AND user_id = 'alice'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(meta["turns"], 2);

        insert_turn(
            &pool,
            "alice",
            "s1",
            "host:agent:user",
            "third turn changes the plan",
            120,
        )
        .await;
        let second = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(second.promoted, 0, "no new episode — the session grew");
        assert_eq!(second.refreshed, 1);

        let (same_id, content, meta): (Uuid, String, serde_json::Value) = sqlx::query_as(
            "SELECT id, content, meta FROM curated_nodes WHERE kind = 'episodic' AND user_id = 'alice'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            same_id, node_id,
            "refresh keeps the node id (decay state survives)"
        );
        assert!(
            content.contains("third turn changes the plan"),
            "transcript grew: {content}"
        );
        assert_eq!(meta["turns"], 3);
        assert!(
            meta.get("distilled_at").is_none(),
            "a grown session must be distillable again"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn curate_skips_summary_rebuild_when_nothing_changed(pool: PgPool) {
        // Enough level-0 nodes to clear SUMMARY_FANOUT_STOP, or no tree forms.
        for i in 0..8 {
            insert_document_raw(
                &pool,
                "alice",
                None,
                None,
                &format!("note {i}: the deploy target for the pgvector service is staging"),
            )
            .await;
        }
        curate(&pool, &DistillingNlp, "alice").await.unwrap();
        let mut before: Vec<Uuid> = sqlx::query_scalar(
            "SELECT id FROM curated_nodes WHERE kind = 'summary' AND user_id = 'alice'",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        before.sort();
        assert!(!before.is_empty(), "test needs a summary tree");

        // An idle pass must not touch the tree — same node ids afterwards.
        curate(&pool, &DistillingNlp, "alice").await.unwrap();
        let mut after: Vec<Uuid> = sqlx::query_scalar(
            "SELECT id FROM curated_nodes WHERE kind = 'summary' AND user_id = 'alice'",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        after.sort();
        assert_eq!(
            before, after,
            "summary tree must be untouched by an idle pass"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn concurrent_curation_is_single_flight_per_user(pool: PgPool) {
        insert_document_raw(&pool, "alice", None, None, "the deploy target is staging").await;

        // Hold alice's curation lock on a separate connection — exactly what a
        // racing scheduler tick or rebuild holds.
        let mut holder = pool.acquire().await.unwrap();
        let got: bool = sqlx::query_scalar("SELECT pg_try_advisory_lock($1, hashtext('alice'))")
            .bind(CURATION_LOCK_NS)
            .fetch_one(&mut *holder)
            .await
            .unwrap();
        assert!(got);

        let stats = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert!(stats.locked_out, "a held lock must skip the pass");
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 0);
        let stats = rebuild(&pool, &DistillingNlp, "alice").await.unwrap();
        assert!(stats.locked_out, "rebuild honors the same lock");

        sqlx::query("SELECT pg_advisory_unlock($1, hashtext('alice'))")
            .bind(CURATION_LOCK_NS)
            .execute(&mut *holder)
            .await
            .unwrap();
        let stats = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert!(!stats.locked_out);
        assert_eq!(stats.promoted, 1, "released lock, pass proceeds");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn singletons_bump_attempts_and_undistilled_backlog_is_counted(pool: PgPool) {
        // Two topics with disjoint entities: neither clusters, both stay
        // eligible, and each pass must bump their attempt counter — the
        // rotation key that keeps head-of-line singletons from starving the
        // distill batch — while the undistilled backlog stays visible.
        insert_document_raw(&pool, "alice", None, None, "the rocket launch window moved").await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "repotted the garden basil today",
        )
        .await;

        let first = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(first.promoted, 2);
        assert_eq!(first.distilled, 0, "disjoint topics must not cluster");
        assert_eq!(first.pending_undistilled, 2);
        let attempts: Vec<i64> = sqlx::query_scalar(
            "SELECT (meta->>'distill_attempts')::bigint FROM curated_nodes \
             WHERE kind = 'episodic' AND user_id = 'alice' ORDER BY id",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(attempts, vec![1, 1]);

        let second = curate(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(second.pending_undistilled, 2);
        let attempts: Vec<i64> = sqlx::query_scalar(
            "SELECT (meta->>'distill_attempts')::bigint FROM curated_nodes \
             WHERE kind = 'episodic' AND user_id = 'alice' ORDER BY id",
        )
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(attempts, vec![2, 2], "each pass rotates the counter");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn new_evidence_joins_old_context_and_supersedes_the_stale_fact(pool: PgPool) {
        let nlp = CapturingNlp::default();
        // Two same-topic documents from ten days ago produce the first fact.
        insert_document_raw_at(
            &pool,
            "alice",
            "the deploy target for the pgvector service is staging",
            10,
        )
        .await;
        insert_document_raw_at(
            &pool,
            "alice",
            "note: the deploy target for the pgvector service is staging",
            10,
        )
        .await;
        let first = curate(&pool, &nlp, "alice").await.unwrap();
        assert_eq!(first.distilled, 1);
        assert_eq!(first.superseded, 0);

        // A single NEWER document on the same topic: it must cluster with the
        // already-distilled context (a lone new episode used to wait forever),
        // and its fact must close the old fact's validity window.
        insert_document_raw_at(
            &pool,
            "alice",
            "update: the deploy target for the pgvector service is production",
            0,
        )
        .await;
        let second = curate(&pool, &nlp, "alice").await.unwrap();
        assert_eq!(
            second.distilled, 1,
            "one new episode + context must distill"
        );
        assert_eq!(second.superseded, 1, "the stale fact's window must close");

        // The second distill call saw all three episodes, not the newcomer alone.
        let seen = nlp.seen.lock().unwrap().len();
        assert_eq!(seen, 2 + 3, "context episodes ride into the second cluster");

        // Two facts exist; exactly one is still active, and it is the newer.
        let (total, active): (i64, i64) = sqlx::query_as(
            "SELECT COUNT(*), \
                    COUNT(*) FILTER (WHERE NOT EXISTS ( \
                        SELECT 1 FROM curated_edges s \
                        WHERE s.to_id = curated_nodes.id AND s.kind = 'supersedes')) \
             FROM curated_nodes WHERE kind = 'semantic' AND user_id = 'alice'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!((total, active), (2, 1));

        // Nothing left to do: a third pass is a no-op.
        let third = curate(&pool, &nlp, "alice").await.unwrap();
        assert_eq!(third.distilled, 0);
        assert_eq!(third.superseded, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn unrelated_newer_facts_do_not_retire_older_ones(pool: PgPool) {
        // A stub whose fact text derives from the cluster, so facts about
        // different topics are genuinely dissimilar — which is what makes the
        // supersede cosine threshold testable in the refusing direction.
        struct TopicFactNlp;
        #[async_trait]
        impl NlpService for TopicFactNlp {
            fn provider_name(&self) -> &'static str {
                "test-topic"
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
            async fn embed_one(&self, text: &str) -> Result<Vec<f32>, AppError> {
                Ok(bow_embed(text))
            }
            async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
                Ok(texts.iter().map(|t| bow_embed(t)).collect())
            }
            fn extract_entities(&self, text: &str) -> Vec<String> {
                flashback_nlp::extract_entities(text)
            }
            async fn extract_full(&self, _t: &str) -> Result<Extraction, AppError> {
                Ok(Extraction::empty())
            }
            async fn distill_facts(
                &self,
                e: &[EpisodeRef],
            ) -> Result<Vec<DistilledFact>, ProviderError> {
                Ok(vec![DistilledFact {
                    content: format!("summary: {}", e[0].content),
                    topic: None,
                    source_episode_ids: e.iter().map(|x| x.id).collect(),
                    confidence: 0.9,
                }])
            }
        }

        insert_document_raw_at(
            &pool,
            "alice",
            "the deploy target for the pgvector service is staging",
            10,
        )
        .await;
        insert_document_raw_at(
            &pool,
            "alice",
            "note: the deploy target for the pgvector service is staging",
            10,
        )
        .await;
        let first = curate(&pool, &TopicFactNlp, "alice").await.unwrap();
        assert_eq!(first.distilled, 1);

        // A newer, UNRELATED topic must not close the deploy fact's window.
        insert_document_raw_at(
            &pool,
            "alice",
            "repotted the garden basil into fresh soil",
            0,
        )
        .await;
        insert_document_raw_at(
            &pool,
            "alice",
            "note: repotted the garden basil with fresh soil",
            0,
        )
        .await;
        let second = curate(&pool, &TopicFactNlp, "alice").await.unwrap();
        assert_eq!(second.distilled, 1);
        assert_eq!(second.superseded, 0, "dissimilar facts must coexist");
        let active: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes n WHERE n.kind = 'semantic' \
             AND n.user_id = 'alice' AND NOT EXISTS ( \
                 SELECT 1 FROM curated_edges s \
                 WHERE s.to_id = n.id AND s.kind = 'supersedes')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(active, 2);
    }

    // -- rebuild -----------------------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn rebuild_reproduces_the_same_derived_set(pool: PgPool) {
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;

        // First build.
        let first = rebuild(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(first.promoted, 2);
        assert_eq!(first.distilled, 1);
        let ep1 = count_nodes(&pool, "alice", "episodic").await;
        let sem1 = count_nodes(&pool, "alice", "semantic").await;

        // Rebuild — wipe + re-derive. Same counts, no duplication.
        let second = rebuild(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(second.promoted, 2);
        assert_eq!(second.distilled, 1);
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, ep1);
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, sem1);
    }

    // -- immutability + scope ---------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn curation_never_mutates_raw(pool: PgPool) {
        let raw_id = insert_document_raw(&pool, "alice", None, None, "took 5mg lisinopril").await;
        let scope = Scope::new("alice");
        promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        distill_semantic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        rebuild(&pool, &DistillingNlp, "alice").await.unwrap();

        // The raw row is byte-for-byte unchanged, and the immutability trigger
        // still fires on a direct write attempt.
        let content: String = sqlx::query_scalar("SELECT content FROM raw_records WHERE id = $1")
            .bind(raw_id)
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(content, "took 5mg lisinopril");
        let upd = sqlx::query("UPDATE raw_records SET content = 'x' WHERE id = $1")
            .bind(raw_id)
            .execute(&pool)
            .await;
        assert!(upd.is_err(), "raw_records must remain append-only");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn curation_is_scope_isolated_by_user(pool: PgPool) {
        insert_document_raw(&pool, "alice", None, None, "alice deploy target staging").await;
        insert_document_raw(&pool, "bob", None, None, "bob deploy target staging").await;

        // Curate only alice.
        let scope = Scope::new("alice");
        promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();

        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 1);
        assert_eq!(count_nodes(&pool, "bob", "episodic").await, 0);

        // Bob's rebuild only touches bob; alice's node survives.
        rebuild(&pool, &DistillingNlp, "bob").await.unwrap();
        assert_eq!(count_nodes(&pool, "alice", "episodic").await, 1);
        assert_eq!(count_nodes(&pool, "bob", "episodic").await, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_topic_scoped_node_is_found_through_the_evidence_it_cites(pool: PgPool) {
        // A curated node spans topics, so it carries none of its own. Scoping by
        // topic must therefore ask "does this node cite evidence from there" —
        // filtering the node's own (always null) topic silently returns nothing.
        insert_document_raw(
            &pool,
            "alice",
            Some("health"),
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            Some("taxes"),
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;
        rebuild(&pool, &DistillingNlp, "alice").await.unwrap();

        let scoped: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes n
             WHERE n.user_id = 'alice' AND EXISTS (
                 SELECT 1 FROM curated_edges de
                 JOIN raw_records dr ON dr.id = de.to_id
                 WHERE de.from_id = n.id AND de.kind = 'derived_from' AND dr.topic_id = $1
             )",
        )
        .bind("taxes")
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(
            scoped > 0,
            "a topic-scoped request must still reach curated facts"
        );

        let absent: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes n
             WHERE n.user_id = 'alice' AND EXISTS (
                 SELECT 1 FROM curated_edges de
                 JOIN raw_records dr ON dr.id = de.to_id
                 WHERE de.from_id = n.id AND de.kind = 'derived_from' AND dr.topic_id = $1
             )",
        )
        .bind("gardening")
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(absent, 0, "an unrelated topic must still narrow");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_parent_that_arrives_late_gets_linked_on_rebuild(pool: PgPool) {
        // The child names a parent that does not exist yet. That is a gap, not a
        // loss — and the gap has to close when the parent shows up, or moving
        // resolution out of raw bought nothing.
        let child: Uuid = sqlx::query_scalar(
            "INSERT INTO raw_records (type, content, event_time, source, source_ref,
                                      user_id, prev_source_ref)
             VALUES ('conversation','the reply',NOW(),'t','ref:2','alice','ref:1')
             RETURNING id",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        rederive_edges(&pool, "alice").await.unwrap();
        let linked: Option<Uuid> =
            sqlx::query_scalar("SELECT prev_id FROM derived_link WHERE record_id = $1")
                .bind(child)
                .fetch_optional(&pool)
                .await
                .unwrap()
                .flatten();
        assert_eq!(linked, None, "nothing to link to yet");

        let parent: Uuid = sqlx::query_scalar(
            "INSERT INTO raw_records (type, content, event_time, source, source_ref, user_id)
             VALUES ('conversation','the question',NOW(),'t','ref:1','alice')
             RETURNING id",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        rederive_edges(&pool, "alice").await.unwrap();
        let healed: Option<Uuid> =
            sqlx::query_scalar("SELECT prev_id FROM derived_link WHERE record_id = $1")
                .bind(child)
                .fetch_optional(&pool)
                .await
                .unwrap()
                .flatten();
        assert_eq!(healed, Some(parent), "the late parent must get linked");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn rederiving_never_destroys_a_register(pool: PgPool) {
        // The register is the one edge nothing can recompute from the row, so a
        // rebuild must leave it alone and only fill records that have none.
        crate::modes::ensure_builtin_modes(&pool, "alice")
            .await
            .unwrap();
        let classified: Uuid = sqlx::query_scalar(
            "INSERT INTO raw_records (type, content, event_time, source, user_id)
             VALUES ('document','rust borrow checker',NOW(),'t','alice') RETURNING id",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        sqlx::query(
            "INSERT INTO derived_record_mode
                (record_id, user_id, mode, embedder, event_time, origin, decided_by)
             SELECT $1, 'alice', 'code', m.embedder, NOW(), 'llm', 'classifier-v1'
             FROM modes m WHERE m.user_id = 'alice' AND m.name = 'code'",
        )
        .bind(classified)
        .execute(&pool)
        .await
        .unwrap();

        let bare: Uuid = sqlx::query_scalar(
            "INSERT INTO raw_records (type, content, event_time, source, user_id)
             VALUES ('document','no register at all',NOW(),'t','alice') RETURNING id",
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        rederive_edges(&pool, "alice").await.unwrap();

        let kept: (String, String) =
            sqlx::query_as("SELECT mode, origin FROM derived_record_mode WHERE record_id = $1")
                .bind(classified)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(
            kept,
            ("code".into(), "llm".into()),
            "a real classification survives"
        );

        let filled: (String, String) =
            sqlx::query_as("SELECT mode, origin FROM derived_record_mode WHERE record_id = $1")
                .bind(bare)
                .fetch_one(&pool)
                .await
                .unwrap();
        assert_eq!(
            filled.1, "default",
            "a record with no register gets the default"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn curation_crosses_topics(pool: PgPool) {
        // Same shape as the mode test, but split by topic instead. Topics do not
        // partition, so these MUST merge.
        insert_document_raw(
            &pool,
            "alice",
            Some("health"),
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            Some("taxes"),
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;

        rebuild(&pool, &DistillingNlp, "alice").await.unwrap();

        let semantic: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes WHERE user_id='alice' AND kind='semantic'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            semantic, 1,
            "records in different topics must still cluster"
        );

        let sources: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_edges e \
             JOIN curated_nodes n ON n.id = e.from_id \
             WHERE n.user_id='alice' AND n.kind='semantic' AND e.kind='derived_from'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(
            sources, 2,
            "the semantic fact must cite both topics' records"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn curation_never_crosses_a_mode_boundary(pool: PgPool) {
        // Identical entity sets that would cluster — except different modes.
        insert_document_raw(
            &pool,
            "alice",
            None,
            Some("code"),
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            Some("journal"),
            "the deploy target for the pgvector service moved to production",
        )
        .await;

        // A rebuild processes each mode bucket separately.
        rebuild(&pool, &DistillingNlp, "alice").await.unwrap();

        // Each mode promoted its own episodic node...
        let code_ep: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes WHERE user_id='alice' AND kind='episodic' AND mode='code'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        let journal_ep: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes WHERE user_id='alice' AND kind='episodic' AND mode='journal'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!((code_ep, journal_ep), (1, 1));

        // ...and because each mode's scope saw only ONE episode, neither could
        // form a >=2 cluster, so ZERO semantic facts were distilled. The two
        // modes never merged into a cross-register fact.
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, 0);

        // Every curated node carries the mode of its source — never a blend.
        let leaked: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM curated_nodes WHERE user_id='alice' AND mode NOT IN ('code','journal')",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(leaked, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn same_mode_still_clusters_together(pool: PgPool) {
        // Control for the boundary test: the SAME two records in the SAME mode
        // DO cluster into one semantic fact — proving the boundary above is the
        // mode split, not something else blocking the merge.
        insert_document_raw(
            &pool,
            "alice",
            None,
            Some("code"),
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            Some("code"),
            "the deploy target for the pgvector service moved to production",
        )
        .await;
        rebuild(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, 1);
    }

    // -- entity index ------------------------------------------------------

    async fn count_entity_index(pool: &PgPool, user_id: &str) -> i64 {
        sqlx::query_scalar("SELECT COUNT(*) FROM entity_index WHERE user_id = $1")
            .bind(user_id)
            .fetch_one(pool)
            .await
            .unwrap()
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn promote_populates_entity_index_and_entity_edges(pool: PgPool) {
        let raw_id = insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        let scope = Scope::new("alice");
        promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();

        // entity_index has the entities extracted from the raw content, all
        // pointing at that raw id.
        let entities: Vec<String> = sqlx::query_scalar(
            "SELECT entity FROM entity_index WHERE user_id = 'alice' AND record_id = $1 ORDER BY entity",
        )
        .bind(raw_id)
        .fetch_all(&pool)
        .await
        .unwrap();
        let expected = flashback_nlp::extract_entities(
            "the deploy target for the pgvector service is staging",
        );
        assert!(!entities.is_empty());
        assert_eq!(entities.len(), expected.len());

        // A single labelled 'entity' edge (node -> raw id) marks the lineage;
        // the per-entity fanout is the entity_index (asserted above). The edge
        // PK is (from_id, to_id, kind), so one edge per node/raw pair carrying
        // one of the entities as its label.
        let edge_labels: Vec<Option<String>> = sqlx::query_scalar(
            "SELECT label FROM curated_edges WHERE kind = 'entity' AND to_id = $1",
        )
        .bind(raw_id)
        .fetch_all(&pool)
        .await
        .unwrap();
        assert_eq!(edge_labels.len(), 1);
        let label = edge_labels[0].as_ref().unwrap();
        assert!(
            expected.contains(label),
            "edge label is one of the entities"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn distill_clusters_using_the_entity_index(pool: PgPool) {
        // Two records with identical entity sets → jaccard 1.0 → one cluster.
        // Promotion fills the index; distill must read it (no on-the-fly needed).
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;
        let scope = Scope::new("alice");
        promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();
        assert!(count_entity_index(&pool, "alice").await > 0);

        // A NlpService whose extract_entities panics if called — proves distill
        // read the index, not the on-the-fly fallback.
        #[derive(Clone)]
        struct IndexOnlyNlp;
        #[async_trait]
        impl NlpService for IndexOnlyNlp {
            fn provider_name(&self) -> &'static str {
                "index-only"
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
            async fn embed_batch(&self, t: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
                Ok((0..t.len()).map(|_| vec![0.1_f32; 384]).collect())
            }
            fn extract_entities(&self, _t: &str) -> Vec<String> {
                panic!("distill must cluster from entity_index, not on-the-fly extraction");
            }
            async fn extract_full(&self, _t: &str) -> Result<Extraction, AppError> {
                Ok(Extraction::empty())
            }
            async fn distill_facts(
                &self,
                episodes: &[EpisodeRef],
            ) -> Result<Vec<DistilledFact>, ProviderError> {
                Ok(vec![DistilledFact {
                    content: "distilled".into(),
                    topic: None,
                    source_episode_ids: episodes.iter().map(|e| e.id).collect(),
                    confidence: 0.9,
                }])
            }
        }

        let stats = distill_semantic(&pool, &IndexOnlyNlp, &scope)
            .await
            .unwrap();
        assert_eq!(
            stats.distilled, 1,
            "the two records clustered into one fact"
        );
        assert_eq!(count_nodes(&pool, "alice", "semantic").await, 1);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn distill_falls_back_when_index_empty(pool: PgPool) {
        // Promote via a path that leaves entity_index empty (insert episodic
        // nodes directly, bypassing promote's indexing), then distill — the
        // fallback on-the-fly extraction must still cluster the two records.
        let r1 = insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        let r2 = insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;
        for raw in [r1, r2] {
            let node = Uuid::new_v4();
            let content: String =
                sqlx::query_scalar("SELECT content FROM raw_records WHERE id = $1")
                    .bind(raw)
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            insert_node(
                &pool,
                node,
                "episodic",
                &content,
                &Scope::new("alice"),
                None,
                None,
            )
            .await
            .unwrap();
            add_edge(&pool, node, raw, "derived_from").await.unwrap();
        }
        assert_eq!(count_entity_index(&pool, "alice").await, 0);

        let stats = distill_semantic(&pool, &DistillingNlp, &Scope::new("alice"))
            .await
            .unwrap();
        assert_eq!(
            stats.distilled, 1,
            "fallback extraction clustered the records"
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn rebuild_repopulates_entity_index_deterministically(pool: PgPool) {
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service is staging",
        )
        .await;
        insert_document_raw(
            &pool,
            "alice",
            None,
            None,
            "the deploy target for the pgvector service moved to production",
        )
        .await;

        rebuild(&pool, &DistillingNlp, "alice").await.unwrap();
        let first = count_entity_index(&pool, "alice").await;
        assert!(first > 0);

        // Rebuild wipes + re-derives; the index count is identical (no dupes,
        // no leaks) — deterministic repopulation.
        rebuild(&pool, &DistillingNlp, "alice").await.unwrap();
        assert_eq!(count_entity_index(&pool, "alice").await, first);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn derivations_lists_curated_nodes_for_a_raw_id(pool: PgPool) {
        let raw_id = insert_document_raw(&pool, "alice", None, None, "took 5mg lisinopril").await;
        let scope = Scope::new("alice");
        promote_raw_to_episodic(&pool, &DistillingNlp, &scope)
            .await
            .unwrap();

        let derived = derivations_of(&pool, "alice", raw_id).await.unwrap();
        assert_eq!(derived.len(), 1);
        assert_eq!(derived[0].kind, "episodic");

        // Wrong user sees nothing.
        let none = derivations_of(&pool, "bob", raw_id).await.unwrap();
        assert!(none.is_empty());
    }
}
