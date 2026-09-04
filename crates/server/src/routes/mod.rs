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

    #[sqlx::test(migrations = "../../migrations")]
    async fn the_api_requires_a_token(pool: PgPool) {
        // Unauthenticated, so anything other than 401 here would mean the auth
        // layer is not actually in front of the records routes.
        let code = get(pool, "/records/00000000-0000-0000-0000-000000000000").await;
        assert_eq!(code, StatusCode::UNAUTHORIZED);
    }
}
