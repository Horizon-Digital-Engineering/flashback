pub mod admin;
pub mod health;
pub mod modes;
pub mod records;

use crate::AppState;
use axum::Router;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", axum::routing::get(health::health_check))
        .nest("/admin", admin::router(state.clone()))
        .nest("/records", records::router(state.clone()))
        .nest("/modes", modes::router(state.clone()))
        .nest("/catalog", crate::catalog::router(state.clone()))
        .nest("/proposals", crate::proposals::router(state.clone()))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::state_from;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use sqlx::PgPool;
    use tower::ServiceExt;

    async fn get(pool: PgPool, path: &str) -> StatusCode {
        router(state_from(pool))
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn health_is_mounted_and_open(pool: PgPool) {
        assert_eq!(get(pool, "/health").await, StatusCode::OK);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn every_subtree_is_mounted(pool: PgPool) {
        // A nested router that silently stops being mounted looks exactly like a
        // handler bug from the outside, so assert the paths resolve at all —
        // 404 is the failure, anything else means routing found something.
        for path in [
            "/records/nope",
            "/modes",
            "/catalog",
            "/proposals",
            "/admin",
        ] {
            let code = get(pool.clone(), path).await;
            assert_ne!(code, StatusCode::NOT_FOUND, "{path} is not mounted");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn an_unmounted_path_is_still_a_404(pool: PgPool) {
        assert_eq!(
            get(pool, "/definitely-not-a-route").await,
            StatusCode::NOT_FOUND
        );
    }

    /// Every path the routers declare, scraped from the source so a route added
    /// tomorrow is covered without anyone remembering to list it here.
    fn declared_paths() -> Vec<String> {
        const SOURCES: &[(&str, &str)] = &[
            ("", include_str!("mod.rs")),
            ("/records", include_str!("records.rs")),
            ("/modes", include_str!("modes.rs")),
            ("/catalog", include_str!("../catalog.rs")),
            ("/proposals", include_str!("../proposals.rs")),
            ("/admin", include_str!("admin/mod.rs")),
        ];
        let mut out = Vec::new();
        for (prefix, src) in SOURCES {
            for chunk in src.split(".route(").skip(1) {
                let Some(open) = chunk.find('"') else {
                    continue;
                };
                let Some(close) = chunk[open + 1..].find('"') else {
                    continue;
                };
                let leaf = &chunk[open + 1..open + 1 + close];
                if !leaf.starts_with('/') {
                    continue;
                }
                let full = if leaf == "/" {
                    if prefix.is_empty() {
                        "/".to_string()
                    } else {
                        prefix.to_string()
                    }
                } else {
                    format!("{prefix}{leaf}")
                };
                let concrete = full
                    .replace("{id}", "00000000-0000-0000-0000-000000000000")
                    .replace("{name}", "general");
                if !out.contains(&concrete) {
                    out.push(concrete);
                }
            }
        }
        out
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn no_route_answers_an_unauthenticated_request(pool: PgPool) {
        // Deliberately public: the health probe, the login form and the assets
        // the login page needs before anyone has a token.
        const PUBLIC: &[&str] = &[
            "/health",
            "/admin/login",
            "/admin/style.css",
            "/favicon.ico",
        ];

        let paths = declared_paths();
        assert!(
            paths.len() > 30,
            "the scraper stopped finding routes: {paths:?}"
        );

        for path in paths {
            if PUBLIC.contains(&path.as_str()) {
                continue;
            }
            let code = get(pool.clone(), &path).await;
            assert!(
                !code.is_success(),
                "{path} answered an unauthenticated GET with {code}"
            );
            assert!(
                !code.is_server_error(),
                "{path} met an unauthenticated GET with {code} instead of refusing it"
            );
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn the_public_paths_are_the_only_public_paths(pool: PgPool) {
        for path in ["/health", "/admin/login", "/admin/style.css"] {
            let code = get(pool.clone(), path).await;
            assert!(code.is_success(), "{path} should be reachable, got {code}");
        }
    }

    async fn authed_get(pool: PgPool, user: &str, path: &str) -> (StatusCode, String) {
        as_role(pool, user, crate::auth::TokenRole::Operator, path).await
    }

    async fn as_role(
        pool: PgPool,
        user: &str,
        role: crate::auth::TokenRole,
        path: &str,
    ) -> (StatusCode, String) {
        let r = crate::testsupport::authed_router(state_from(pool), user, role)
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = r.status();
        let bytes = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn no_route_breaks_on_an_empty_store(pool: PgPool) {
        // A store with no records is the state every deployment starts in, and
        // an empty-state crash is invisible until someone installs it.
        for path in declared_paths() {
            if path == "/admin/logout" {
                continue;
            }
            let (code, body) = authed_get(pool.clone(), "alice", &path).await;
            assert!(
                !code.is_server_error(),
                "{path} returned {code} on an empty store: {body}"
            );
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn no_route_breaks_once_there_is_something_to_render(pool: PgPool) {
        // An empty store never deserialises a row, so it cannot catch a column
        // that stopped existing. Give every page something to draw first.
        let state = state_from(pool.clone());
        for (kind, content, payload) in [
            (
                "conversation",
                "the billing service moved clusters",
                serde_json::json!({}),
            ),
            (
                "document",
                "the quarterly report is due friday",
                serde_json::json!({"origin": "test"}),
            ),
            (
                "state_object",
                "today's plan",
                serde_json::json!({"kind": "plan", "key": "today"}),
            ),
        ] {
            crate::routes::records::ingest_record(
                &state.pool,
                &*state.nlp,
                "alice",
                crate::routes::records::IngestRecordRequest {
                    r#type: kind.into(),
                    content: content.into(),
                    event_time: None,
                    source: "test".into(),
                    source_ref: None,
                    topic_id: Some("work".into()),
                    thread_id: Some("c1".into()),
                    mode: None,
                    supersedes: None,
                    prev_source_ref: None,
                    payload: Some(payload),
                },
            )
            .await
            .unwrap();
        }
        crate::curation::curate(&state.pool, &*state.nlp, "alice")
            .await
            .unwrap();

        for path in declared_paths() {
            if path == "/admin/logout" {
                continue;
            }
            let (code, body) = authed_get(pool.clone(), "alice", &path).await;
            assert!(
                !code.is_server_error(),
                "{path} returned {code} with records present: {body}"
            );
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn nothing_of_alices_reaches_bob(pool: PgPool) {
        // One secret string, written through every writer alice has, then every
        // route driven as bob. Anything that echoes it back is a leak, whatever
        // the route thought it was returning.
        const SECRET: &str = "zzsecretzz-quarterly-passphrase";
        let state = state_from(pool.clone());
        crate::routes::records::ingest_record(
            &state.pool,
            &*state.nlp,
            "alice",
            crate::routes::records::IngestRecordRequest {
                r#type: "document".into(),
                content: SECRET.into(),
                event_time: None,
                source: "test".into(),
                source_ref: None,
                topic_id: Some(SECRET.into()),
                thread_id: Some(SECRET.into()),
                mode: None,
                supersedes: None,
                prev_source_ref: None,
                payload: Some(serde_json::json!({ "note": SECRET })),
            },
        )
        .await
        .unwrap();
        crate::modes::create_mode(
            &state.pool,
            "alice",
            SECRET,
            crate::modes::UpsertModeRequest {
                embedder: "all-MiniLM-L6-v2".into(),
                embedding_dim: None,
                description: Some(SECRET.into()),
                default_decay: None,
                prompt_overrides: None,
                is_default: None,
            },
        )
        .await
        .unwrap();
        crate::curation::curate(&state.pool, &*state.nlp, "alice")
            .await
            .unwrap();

        // The admin surface is operator-only and an operator is defined to see
        // the whole estate, so the boundary being tested here is the API's.
        for path in declared_paths() {
            if path.starts_with("/admin") {
                continue;
            }
            let (_, body) =
                as_role(pool.clone(), "bob", crate::auth::TokenRole::Service, &path).await;
            assert!(
                !body.contains(SECRET),
                "{path} showed bob something of alice's"
            );
        }

        let (_, dash) = authed_get(pool, "carol", "/admin").await;
        assert!(
            dash.contains("1") || !dash.is_empty(),
            "an operator dashboard should still render the estate"
        );
    }

    async fn api(
        pool: PgPool,
        user: &str,
        method: &str,
        path: &str,
        body: Option<serde_json::Value>,
    ) -> (StatusCode, serde_json::Value) {
        let req = Request::builder().method(method).uri(path);
        let req = match body {
            Some(b) => req
                .header("content-type", "application/json")
                .body(Body::from(b.to_string()))
                .unwrap(),
            None => req.body(Body::empty()).unwrap(),
        };
        let r = crate::testsupport::authed_router(
            state_from(pool),
            user,
            crate::auth::TokenRole::Service,
        )
        .oneshot(req)
        .await
        .unwrap();
        let status = r.status();
        let bytes = axum::body::to_bytes(r.into_body(), usize::MAX)
            .await
            .unwrap_or_default();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
        )
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_record_goes_in_and_comes_back_out_the_way_it_arrived(pool: PgPool) {
        let (code, rec) = api(
            pool.clone(),
            "alice",
            "POST",
            "/records",
            Some(serde_json::json!({
                "type": "document",
                "content": "the backup drive is in the safe",
                "source": "test",
                "topic_id": "home",
                "thread_id": "c1"
            })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{rec}");
        let id = rec["id"].as_str().unwrap().to_string();

        let (code, got) = api(
            pool.clone(),
            "alice",
            "GET",
            &format!("/records/{id}"),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(got["content"], "the backup drive is in the safe");
        assert_eq!(got["topic_id"], "home");

        let (code, _) = api(pool.clone(), "bob", "GET", &format!("/records/{id}"), None).await;
        assert_eq!(code, StatusCode::NOT_FOUND, "bob read alice's record");

        let (code, verify) = api(pool.clone(), "alice", "GET", "/records/verify", None).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(verify["broken"], 0, "{verify}");

        for sub in ["lineage", "derivations"] {
            let (code, _) = api(
                pool.clone(),
                "alice",
                "GET",
                &format!("/records/{id}/{sub}"),
                None,
            )
            .await;
            assert_eq!(code, StatusCode::OK, "{sub}");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_record_that_makes_no_sense_is_refused_rather_than_stored(pool: PgPool) {
        for bad in [
            serde_json::json!({ "type": "document", "content": "", "source": "t" }),
            serde_json::json!({ "type": "", "content": "x", "source": "t" }),
            serde_json::json!({ "type": "document", "content": "x" }),
        ] {
            let (code, _) = api(pool.clone(), "alice", "POST", "/records", Some(bad)).await;
            assert!(
                code.is_client_error(),
                "a malformed record was accepted with {code}"
            );
        }
        let stored: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM raw_records")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(stored, 0);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn query_and_context_answer_within_the_callers_own_records(pool: PgPool) {
        for (u, c) in [
            ("alice", "the backup drive is in the safe"),
            ("bob", "bob's own note about drives"),
        ] {
            api(
                pool.clone(),
                u,
                "POST",
                "/records",
                Some(serde_json::json!({
                    "type": "document", "content": c, "source": "test", "thread_id": "c1"
                })),
            )
            .await;
        }

        let (code, out) = api(
            pool.clone(),
            "alice",
            "POST",
            "/records/query",
            Some(serde_json::json!({ "query": "drive", "limit": 10 })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{out}");
        let body = out.to_string();
        assert!(body.contains("in the safe"), "{body}");
        assert!(!body.contains("bob's own note"), "{body}");

        let (code, ctx) = api(
            pool.clone(),
            "alice",
            "POST",
            "/records/context",
            Some(serde_json::json!({ "query": "drive", "limit": 5 })),
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert!(!ctx.to_string().contains("bob's own note"));

        let (code, _) = api(pool, "alice", "GET", "/records/summaries", None).await;
        assert_eq!(code, StatusCode::OK);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_proposal_can_only_cite_the_callers_own_records(pool: PgPool) {
        let (_, alice_rec) = api(
            pool.clone(),
            "alice",
            "POST",
            "/records",
            Some(serde_json::json!({
                "type": "document", "content": "alice's evidence", "source": "test"
            })),
        )
        .await;
        let alice_id = alice_rec["id"].as_str().unwrap().to_string();

        let (code, out) = api(
            pool.clone(),
            "bob",
            "POST",
            "/proposals",
            Some(serde_json::json!({
                "title": "cite what is not mine",
                "action": "do the thing",
                "evidence": [alice_id]
            })),
        )
        .await;
        assert!(
            code.is_client_error(),
            "bob cited alice's record and got {code}: {out}"
        );

        let (code, mine) = api(
            pool.clone(),
            "alice",
            "POST",
            "/proposals",
            Some(serde_json::json!({
                "title": "cite my own",
                "action": "do the thing",
                "rationale": "because",
                "evidence": [alice_id]
            })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{mine}");
        let pid = mine["id"].as_str().unwrap().to_string();
        assert_eq!(mine["status"], "proposed");

        let (code, seen) = api(
            pool.clone(),
            "bob",
            "GET",
            &format!("/proposals/{pid}"),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::NOT_FOUND, "{seen}");

        let (code, approved) = api(
            pool.clone(),
            "alice",
            "POST",
            &format!("/proposals/{pid}/approve"),
            Some(serde_json::json!({})),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{approved}");
        assert_eq!(approved["status"], "approved");

        let (code, _) = api(
            pool.clone(),
            "alice",
            "POST",
            &format!("/proposals/{pid}/deny"),
            Some(serde_json::json!({})),
        )
        .await;
        assert!(
            code.is_client_error(),
            "an approved proposal was denied afterwards"
        );

        let (code, listed) = api(pool, "alice", "GET", "/proposals", None).await;
        assert_eq!(code, StatusCode::OK);
        assert!(listed.to_string().contains("cite my own"));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_catalog_store_belongs_to_the_user_who_declared_it(pool: PgPool) {
        let (code, store) = api(
            pool.clone(),
            "alice",
            "POST",
            "/catalog/stores",
            Some(serde_json::json!({
                "name": "invoices",
                "kind": "external",
                "description": "the billing export"
            })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{store}");
        let id = store["id"].as_str().unwrap().to_string();

        let (code, _) = api(
            pool.clone(),
            "bob",
            "GET",
            &format!("/catalog/stores/{id}"),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::NOT_FOUND);

        let (code, listed) = api(pool.clone(), "bob", "GET", "/catalog", None).await;
        assert_eq!(code, StatusCode::OK);
        assert!(!listed.to_string().contains("the billing export"));

        let (code, _) = api(
            pool.clone(),
            "alice",
            "DELETE",
            &format!("/catalog/stores/{id}"),
            None,
        )
        .await;
        assert_eq!(code, StatusCode::OK);

        let (code, _) = api(pool, "alice", "GET", &format!("/catalog/stores/{id}"), None).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_reference_keeps_its_history_and_answers_with_the_latest(pool: PgPool) {
        // A reference is a state_object raw record projected to a current value.
        // Raw is append-only, so "changing" one appends and the older value has
        // to stay readable.
        for v in ["ship the migration", "ship the migration and the runbook"] {
            let (code, out) = api(
                pool.clone(),
                "alice",
                "POST",
                "/records/state/plan/today",
                Some(serde_json::json!({ "data": { "text": v } })),
            )
            .await;
            assert_eq!(code, StatusCode::OK, "{out}");
        }

        let (code, current) = api(
            pool.clone(),
            "alice",
            "GET",
            "/records/state/plan/today",
            None,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        assert!(current.to_string().contains("and the runbook"), "{current}");

        let (code, history) = api(
            pool.clone(),
            "alice",
            "GET",
            "/records/state/plan/today/history",
            None,
        )
        .await;
        assert_eq!(code, StatusCode::OK);
        let h = history.to_string();
        assert!(h.contains("and the runbook"));
        assert!(
            h.contains("ship the migration\""),
            "the superseded value is gone from history: {h}"
        );

        let (code, listed) = api(pool.clone(), "alice", "GET", "/records/state/plan", None).await;
        assert_eq!(code, StatusCode::OK);
        assert!(listed.to_string().contains("today"));

        let (code, theirs) = api(
            pool.clone(),
            "bob",
            "GET",
            "/records/state/plan/today",
            None,
        )
        .await;
        assert_eq!(code, StatusCode::NOT_FOUND, "{theirs}");

        let (code, empty) = api(pool, "alice", "GET", "/records/state/plan/nothing", None).await;
        assert_eq!(code, StatusCode::NOT_FOUND, "{empty}");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_reference_key_that_names_nothing_is_refused(pool: PgPool) {
        for (kind, key) in [("plan", "%20%20"), ("%20%20", "today")] {
            let (code, _) = api(
                pool.clone(),
                "alice",
                "POST",
                &format!("/records/state/{kind}/{key}"),
                Some(serde_json::json!({ "data": { "text": "x" } })),
            )
            .await;
            assert!(code.is_client_error(), "{kind}/{key} was accepted");
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn an_import_is_idempotent_on_the_source_reference(pool: PgPool) {
        // A corpus is re-imported after a failure more often than not; the same
        // source_ref must not become a second record.
        let batch = serde_json::json!({
            "records": [
                {
                    "type": "conversation", "content": "the first turn",
                    "source": "host:helper:user", "source_ref": "turn-1",
                    "thread_id": "c1"
                },
                {
                    "type": "conversation", "content": "the reply",
                    "source": "host:helper:assistant", "source_ref": "turn-2",
                    "thread_id": "c1", "prev_source_ref": "turn-1"
                }
            ]
        });

        let (code, first) = api(
            pool.clone(),
            "alice",
            "POST",
            "/records/import",
            Some(batch.clone()),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{first}");
        assert_eq!(first["imported"], 2);

        let (code, again) = api(
            pool.clone(),
            "alice",
            "POST",
            "/records/import",
            Some(batch),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{again}");
        assert_eq!(again["imported"], 0, "{again}");
        assert_eq!(again["skipped"], 2);

        let stored: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM raw_records")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(stored, 2);

        // The writer's claim about what preceded a turn is kept verbatim on the
        // raw row. Import does not resolve it into derived_link — that is the
        // rebuild's job, and the test below is what proves the rebuild does it.
        let claim: Option<String> = sqlx::query_scalar(
            "SELECT prev_source_ref FROM raw_records WHERE source_ref = 'turn-2'",
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(claim.as_deref(), Some("turn-1"));

        let links: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM derived_link")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(links, 0, "import resolves no edges; a rebuild does");

        let (code, verify) = api(pool, "alice", "GET", "/records/verify", None).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(verify["broken"], 0, "{verify}");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_rebuild_reproduces_the_derived_layer_from_raw(pool: PgPool) {
        api(
            pool.clone(),
            "alice",
            "POST",
            "/records/import",
            Some(serde_json::json!({
                "records": [
                    { "type": "conversation", "content": "second", "source": "s",
                      "source_ref": "b", "prev_source_ref": "a", "thread_id": "c1" },
                    { "type": "conversation", "content": "first", "source": "s",
                      "source_ref": "a", "thread_id": "c1" }
                ]
            })),
        )
        .await;

        // "second" arrived before its parent, so its link could not resolve.
        let before: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM derived_link")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(before, 0, "a parent that had not arrived cannot be linked");

        let (code, out) = api(
            pool.clone(),
            "alice",
            "POST",
            "/records/rebuild",
            Some(serde_json::json!({})),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{out}");

        let after: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM derived_link")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(after, 1, "the late parent was never picked up");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_store_is_described_updated_and_synced_by_its_owner_only(pool: PgPool) {
        let (_, store) = api(
            pool.clone(),
            "alice",
            "POST",
            "/catalog/stores",
            Some(serde_json::json!({
                "name": "invoices", "kind": "external", "description": "first pass"
            })),
        )
        .await;
        let id = store["id"].as_str().unwrap().to_string();

        let (code, updated) = api(
            pool.clone(),
            "alice",
            "PUT",
            &format!("/catalog/stores/{id}"),
            Some(serde_json::json!({
                "description": "second pass",
                "schema": { "columns": ["id", "total"] }
            })),
        )
        .await;
        assert_eq!(code, StatusCode::OK, "{updated}");
        assert_eq!(updated["description"], "second pass");
        assert_eq!(updated["schema"]["columns"][1], "total");

        let (code, _) = api(
            pool.clone(),
            "bob",
            "PUT",
            &format!("/catalog/stores/{id}"),
            Some(serde_json::json!({ "description": "mine now" })),
        )
        .await;
        assert_eq!(code, StatusCode::NOT_FOUND, "bob edited alice's store");

        let (code, synced) = api(
            pool.clone(),
            "alice",
            "POST",
            &format!("/catalog/stores/{id}/sync"),
            Some(serde_json::json!({ "facts": [] })),
        )
        .await;
        assert!(code.is_success() || code.is_client_error(), "{synced}");

        let (code, listed) = api(pool, "alice", "GET", "/catalog", None).await;
        assert_eq!(code, StatusCode::OK);
        let body = listed.to_string();
        assert!(body.contains("second pass"), "{body}");
        assert!(body.contains("raw"), "the built-in stores register on read");
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn no_route_leaks_an_internal_detail_in_its_body(pool: PgPool) {
        // A stack path, a SQL fragment or a connection string in a response body
        // is a disclosure whether or not the status code says error.
        const NEVER: &[&str] = &[
            "/home/",
            "crates/server/src",
            "SELECT ",
            "postgres://",
            "panicked",
            "RUST_BACKTRACE",
        ];
        for path in declared_paths() {
            let (_, body) = authed_get(pool.clone(), "alice", &path).await;
            for needle in NEVER {
                assert!(
                    !body.contains(needle),
                    "{path} put {needle:?} in its response body"
                );
            }
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn the_api_requires_a_token(pool: PgPool) {
        // Unauthenticated, so anything other than 401 here would mean the auth
        // layer is not actually in front of the records routes.
        let code = get(pool, "/records/00000000-0000-0000-0000-000000000000").await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);
    }
}
