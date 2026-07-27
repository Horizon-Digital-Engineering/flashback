//! Proposals — the lake proposes, it never acts.
//!
//! The lake surfaces a decision or an action as a PROPOSAL that cites the
//! evidence (raw record ids) justifying it. The operator/host decides; the host
//! executes. The lake has no code path by which it carries out an action — the
//! absence of an execute endpoint is the enforcement of "propose, don't act".
//!
//! Lifecycle:  proposed → approved → executed   (with proposed → denied)
//!   POST   /proposals               create a 'proposed' row (validates evidence)
//!   GET    /proposals?status=       list, optionally filtered by status
//!   GET    /proposals/:id           one proposal
//!   POST   /proposals/:id/approve   operator approves    (proposed → approved)
//!   POST   /proposals/:id/deny      operator denies      (proposed → denied)
//!   POST   /proposals/:id/executed  host reports done    (approved → executed)
//!
//! There is intentionally no /proposals/:id/execute — the lake never runs the
//! action; `executed` is only the host REPORTING that IT completed the action.

use axum::{
    extract::{Path, Query, State},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{FromRow, PgPool};
use uuid::Uuid;

use crate::{
    auth::AuthUser,
    error::{AppError, AppResult},
    AppState,
};

pub fn router(state: AppState) -> Router<AppState> {
    Router::new()
        .route("/", get(list_proposals).post(create_proposal))
        .route("/{id}", get(get_proposal))
        .route("/{id}/approve", post(approve))
        .route("/{id}/deny", post(deny))
        // The host REPORTS completion; the lake never executes. No /execute route.
        .route("/{id}/executed", post(executed))
        .with_state(state)
}

#[derive(Debug, Clone, FromRow, Serialize)]
pub struct ProposalRow {
    pub id: Uuid,
    pub user_id: String,
    pub kind: String,
    pub title: String,
    pub body: Value,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub decided_at: Option<DateTime<Utc>>,
    pub decided_by: Option<String>,
}

// ---------------------------------------------------------------------------
// POST /proposals — create a proposed row citing evidence
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateProposalRequest {
    /// 'action' | 'insight'. Defaults to 'action'.
    #[serde(default)]
    pub kind: Option<String>,
    pub title: String,
    /// The proposed action / the insight statement.
    pub action: String,
    #[serde(default)]
    pub rationale: Option<String>,
    /// Raw record ids that justify the proposal. Each must exist + be the
    /// caller's; a foreign or missing id is rejected.
    #[serde(default)]
    pub evidence: Vec<Uuid>,
}

async fn create_proposal(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Json(req): Json<CreateProposalRequest>,
) -> AppResult<Json<ProposalRow>> {
    Ok(Json(
        create_proposal_inner(&state.pool, &auth_user.user_id, req).await?,
    ))
}

pub(crate) async fn create_proposal_inner(
    pool: &PgPool,
    user_id: &str,
    req: CreateProposalRequest,
) -> AppResult<ProposalRow> {
    if req.title.trim().is_empty() {
        return Err(AppError::bad_request("title must not be empty"));
    }
    if req.action.trim().is_empty() {
        return Err(AppError::bad_request("action must not be empty"));
    }
    let kind = req.kind.as_deref().unwrap_or("action");
    if !matches!(kind, "action" | "insight") {
        return Err(AppError::bad_request("kind must be 'action' or 'insight'"));
    }

    // Every cited evidence id must exist AND belong to the caller.
    validate_evidence(pool, user_id, &req.evidence).await?;

    let body = json!({
        "action": req.action,
        "rationale": req.rationale,
        "evidence": req.evidence,
    });

    let row = sqlx::query_as::<_, ProposalRow>(
        r#"
        INSERT INTO proposals (user_id, kind, title, body, status)
        VALUES ($1, $2, $3, $4, 'proposed')
        RETURNING id, user_id, kind, title, body, status, created_at, decided_at, decided_by
        "#,
    )
    .bind(user_id)
    .bind(kind)
    .bind(req.title.trim())
    .bind(&body)
    .fetch_one(pool)
    .await?;
    Ok(row)
}

/// Reject the proposal unless every cited evidence id is an existing raw record
/// owned by the caller. Empty evidence is allowed (an insight can stand alone),
/// but any id that is missing or foreign fails the whole create.
async fn validate_evidence(pool: &PgPool, user_id: &str, evidence: &[Uuid]) -> AppResult<()> {
    if evidence.is_empty() {
        return Ok(());
    }
    let owned: Vec<Uuid> = sqlx::query_scalar(
        "SELECT id FROM raw_records WHERE ($1 = '*' OR user_id = $1) AND id = ANY($2)",
    )
    .bind(user_id)
    .bind(evidence)
    .fetch_all(pool)
    .await?;
    let owned_set: std::collections::HashSet<Uuid> = owned.into_iter().collect();
    for id in evidence {
        if !owned_set.contains(id) {
            return Err(AppError::bad_request(format!(
                "evidence record {id} not found"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// GET /proposals?status=  /  GET /proposals/:id
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    #[serde(default)]
    pub status: Option<String>,
}

async fn list_proposals(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Query(q): Query<ListQuery>,
) -> AppResult<Json<Vec<ProposalRow>>> {
    Ok(Json(
        list_proposals_inner(&state.pool, &auth_user.user_id, q.status.as_deref()).await?,
    ))
}

pub(crate) async fn list_proposals_inner(
    pool: &PgPool,
    user_id: &str,
    status: Option<&str>,
) -> AppResult<Vec<ProposalRow>> {
    let rows = sqlx::query_as::<_, ProposalRow>(
        r#"
        SELECT id, user_id, kind, title, body, status, created_at, decided_at, decided_by
        FROM proposals
        WHERE ($1 = '*' OR user_id = $1)
          AND ($2::text IS NULL OR status = $2)
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_id)
    .bind(status)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

async fn get_proposal(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
) -> AppResult<Json<ProposalRow>> {
    Ok(Json(
        fetch_owned(&state.pool, &auth_user.user_id, id).await?,
    ))
}

async fn fetch_owned(pool: &PgPool, user_id: &str, id: Uuid) -> AppResult<ProposalRow> {
    let row = sqlx::query_as::<_, ProposalRow>(
        r#"
        SELECT id, user_id, kind, title, body, status, created_at, decided_at, decided_by
        FROM proposals WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| AppError::not_found(format!("proposal {id}")))?;
    if row.user_id != user_id {
        return Err(AppError::not_found(format!("proposal {id}")));
    }
    Ok(row)
}

// ---------------------------------------------------------------------------
// Decisions + host completion report — the only status transitions.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
pub struct DecisionRequest {
    /// Who made the decision (operator/host identity). Recorded on the row.
    #[serde(default)]
    pub decided_by: Option<String>,
}

async fn approve(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
    body: Option<Json<DecisionRequest>>,
) -> AppResult<Json<ProposalRow>> {
    let who = body.and_then(|Json(b)| b.decided_by);
    Ok(Json(
        approve_inner(&state.pool, &auth_user.user_id, id, who.as_deref()).await?,
    ))
}

async fn deny(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
    body: Option<Json<DecisionRequest>>,
) -> AppResult<Json<ProposalRow>> {
    let who = body.and_then(|Json(b)| b.decided_by);
    Ok(Json(
        deny_inner(&state.pool, &auth_user.user_id, id, who.as_deref()).await?,
    ))
}

async fn executed(
    State(state): State<AppState>,
    auth_user: AuthUser,
    Path(id): Path<Uuid>,
    body: Option<Json<DecisionRequest>>,
) -> AppResult<Json<ProposalRow>> {
    let who = body.and_then(|Json(b)| b.decided_by);
    Ok(Json(
        executed_inner(&state.pool, &auth_user.user_id, id, who.as_deref()).await?,
    ))
}

/// Approve a proposal: proposed → approved. Only a 'proposed' row may be
/// approved; any other current status is a 409 (approve/deny are terminal
/// decisions and executed is terminal, so none can be re-approved).
pub(crate) async fn approve_inner(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
    decided_by: Option<&str>,
) -> AppResult<ProposalRow> {
    transition(pool, user_id, id, "proposed", "approved", decided_by, true).await
}

/// Deny a proposal: proposed → denied (terminal).
pub(crate) async fn deny_inner(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
    decided_by: Option<&str>,
) -> AppResult<ProposalRow> {
    transition(pool, user_id, id, "proposed", "denied", decided_by, true).await
}

/// The host REPORTS it executed an approved proposal: approved → executed
/// (terminal). This is a completion report, not the lake acting — the lake has
/// no path that performs the action itself.
pub(crate) async fn executed_inner(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
    decided_by: Option<&str>,
) -> AppResult<ProposalRow> {
    transition(pool, user_id, id, "approved", "executed", decided_by, false).await
}

/// Apply a single guarded status transition. `from` is the ONLY status the
/// proposal may currently hold; anything else is a 409 Conflict describing the
/// actual state. `set_decision` stamps `decided_at`/`decided_by` (the operator
/// decisions do; the host completion report reuses the field to record who).
async fn transition(
    pool: &PgPool,
    user_id: &str,
    id: Uuid,
    from: &str,
    to: &str,
    decided_by: Option<&str>,
    set_decision: bool,
) -> AppResult<ProposalRow> {
    let current = fetch_owned(pool, user_id, id).await?;
    if current.status != from {
        return Err(AppError::Conflict(format!(
            "proposal {id} is '{}', cannot move to '{to}' (requires '{from}')",
            current.status
        )));
    }
    // decided_at/decided_by carry the decision timestamp + actor on every
    // transition, so the audit trail records who moved it and when.
    let _ = set_decision;
    let row = sqlx::query_as::<_, ProposalRow>(
        r#"
        UPDATE proposals
           SET status = $3, decided_at = NOW(), decided_by = $4
         WHERE id = $1 AND ($2 = '*' OR user_id = $2) AND status = $5
        RETURNING id, user_id, kind, title, body, status, created_at, decided_at, decided_by
        "#,
    )
    .bind(id)
    .bind(user_id)
    .bind(to)
    .bind(decided_by)
    .bind(from)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| AppError::Conflict(format!("proposal {id} changed status concurrently")))?;
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn ingest_raw(pool: &PgPool, user_id: &str, content: &str) -> Uuid {
        use async_trait::async_trait;
        use flashback_nlp::{DistilledFact, EpisodeRef, Extraction, ProviderError};

        #[derive(Clone)]
        struct StubNlp;
        #[async_trait]
        impl crate::nlp::NlpService for StubNlp {
            fn provider_name(&self) -> &'static str {
                "stub"
            }
            fn provider_can_distill(&self) -> bool {
                false
            }
            fn embedder_model_name(&self) -> &str {
                "stub-embedder"
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
                Vec::new()
            }
            async fn extract_full(&self, _t: &str) -> Result<Extraction, AppError> {
                Ok(Extraction::empty())
            }
            async fn distill_facts(
                &self,
                _e: &[EpisodeRef],
            ) -> Result<Vec<DistilledFact>, ProviderError> {
                Err(ProviderError::NotConfigured("stub".into()))
            }
        }

        crate::routes::records::ingest_record(
            pool,
            &StubNlp,
            user_id,
            crate::routes::records::IngestRecordRequest {
                r#type: "document".into(),
                content: content.into(),
                event_time: None,
                source: "test".into(),
                source_ref: None,
                project_id: None,
                container_id: None,
                mode: None,
                importance: None,
                supersedes: None,
                payload: None,
            },
        )
        .await
        .unwrap()
        .id
    }

    fn req(title: &str, action: &str, evidence: Vec<Uuid>) -> CreateProposalRequest {
        CreateProposalRequest {
            kind: None,
            title: title.into(),
            action: action.into(),
            rationale: Some("because the records say so".into()),
            evidence,
        }
    }

    // ---- create validates evidence ---------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_with_valid_evidence_makes_proposed_row(pool: PgPool) {
        let e1 = ingest_raw(&pool, "alice", "spent too much on coffee").await;
        let p = create_proposal_inner(
            &pool,
            "alice",
            req("cut coffee budget", "reduce coffee spend by 20%", vec![e1]),
        )
        .await
        .unwrap();
        assert_eq!(p.status, "proposed");
        assert_eq!(p.kind, "action");
        assert_eq!(p.body["action"], "reduce coffee spend by 20%");
        assert_eq!(p.body["evidence"][0], json!(e1));
        assert!(p.decided_at.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_rejects_missing_evidence(pool: PgPool) {
        let ghost = Uuid::new_v4();
        let err = create_proposal_inner(&pool, "alice", req("x", "do y", vec![ghost]))
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_rejects_foreign_evidence(pool: PgPool) {
        // A raw record owned by bob is not valid evidence for alice's proposal.
        let bobs = ingest_raw(&pool, "bob", "bob's record").await;
        let err = create_proposal_inner(&pool, "alice", req("x", "do y", vec![bobs]))
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn create_allows_empty_evidence_for_insight(pool: PgPool) {
        let mut r = req("a pattern", "you tend to overspend on weekends", vec![]);
        r.kind = Some("insight".into());
        let p = create_proposal_inner(&pool, "alice", r).await.unwrap();
        assert_eq!(p.kind, "insight");
        assert_eq!(p.status, "proposed");
    }

    // ---- decision transitions are terminal-correct ------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn approve_then_executed_flow(pool: PgPool) {
        let p = create_proposal_inner(&pool, "alice", req("t", "act", vec![]))
            .await
            .unwrap();
        let approved = approve_inner(&pool, "alice", p.id, Some("operator"))
            .await
            .unwrap();
        assert_eq!(approved.status, "approved");
        assert_eq!(approved.decided_by.as_deref(), Some("operator"));
        assert!(approved.decided_at.is_some());

        let done = executed_inner(&pool, "alice", p.id, Some("host"))
            .await
            .unwrap();
        assert_eq!(done.status, "executed");

        // executed is terminal — can't approve/deny/re-execute it.
        assert!(approve_inner(&pool, "alice", p.id, None).await.is_err());
        assert!(deny_inner(&pool, "alice", p.id, None).await.is_err());
        assert!(executed_inner(&pool, "alice", p.id, None).await.is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn deny_is_terminal(pool: PgPool) {
        let p = create_proposal_inner(&pool, "alice", req("t", "act", vec![]))
            .await
            .unwrap();
        let denied = deny_inner(&pool, "alice", p.id, Some("operator"))
            .await
            .unwrap();
        assert_eq!(denied.status, "denied");
        // A denied proposal can't be approved or executed.
        assert!(approve_inner(&pool, "alice", p.id, None).await.is_err());
        assert!(executed_inner(&pool, "alice", p.id, None).await.is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn cannot_execute_without_approval(pool: PgPool) {
        // A freshly-proposed row cannot jump straight to executed — it must be
        // approved first. This is the guard behind "propose, don't act": even
        // the host-completion report only applies to something an operator OK'd.
        let p = create_proposal_inner(&pool, "alice", req("t", "act", vec![]))
            .await
            .unwrap();
        let err = executed_inner(&pool, "alice", p.id, None)
            .await
            .unwrap_err();
        assert!(matches!(err, AppError::Conflict(_)));
    }

    // ---- list + scope isolation ------------------------------------------

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_filters_by_status_and_isolates_users(pool: PgPool) {
        let a1 = create_proposal_inner(&pool, "alice", req("a1", "act", vec![]))
            .await
            .unwrap();
        create_proposal_inner(&pool, "alice", req("a2", "act", vec![]))
            .await
            .unwrap();
        approve_inner(&pool, "alice", a1.id, None).await.unwrap();
        create_proposal_inner(&pool, "bob", req("b1", "act", vec![]))
            .await
            .unwrap();

        let alice_all = list_proposals_inner(&pool, "alice", None).await.unwrap();
        assert_eq!(alice_all.len(), 2);
        let alice_proposed = list_proposals_inner(&pool, "alice", Some("proposed"))
            .await
            .unwrap();
        assert_eq!(alice_proposed.len(), 1);
        let alice_approved = list_proposals_inner(&pool, "alice", Some("approved"))
            .await
            .unwrap();
        assert_eq!(alice_approved.len(), 1);

        // Bob sees only his own; can't fetch alice's.
        assert_eq!(
            list_proposals_inner(&pool, "bob", None)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(fetch_owned(&pool, "bob", a1.id).await.is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn decisions_scoped_to_owner(pool: PgPool) {
        let p = create_proposal_inner(&pool, "alice", req("t", "act", vec![]))
            .await
            .unwrap();
        // Bob can't approve alice's proposal (404, not a 409).
        let err = approve_inner(&pool, "bob", p.id, None).await.unwrap_err();
        assert!(matches!(err, AppError::NotFound(_)));
        // Alice's proposal is untouched.
        let still = fetch_owned(&pool, "alice", p.id).await.unwrap();
        assert_eq!(still.status, "proposed");
    }
}
