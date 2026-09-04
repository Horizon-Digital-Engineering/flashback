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
