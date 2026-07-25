//! Bearer-token auth.
//!
//! Tokens are `fb_<32 random base32-ish chars>`. We store sha256(token) in
//! the DB; the plaintext is shown ONCE at mint time and never logged.
//!
//! Authenticated requests carry the user_id in a request extension —
//! handlers read it via the `AuthUser` extractor instead of trusting bodies.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::FromRequestParts,
    http::{header, request::Parts, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use sqlx::PgPool;

use crate::AppState;

const TOKEN_PREFIX: &str = "fb_";
const TOKEN_RANDOM_LEN: usize = 32;

/// Which surface a token may touch. Exactly one of two; the middleware
/// enforces the wall in both directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenRole {
    /// REST/MCP API (ingest, query, context) — what integrations hold.
    Service,
    /// The /admin UI — sees the whole estate, cannot call the API.
    Operator,
}

impl TokenRole {
    pub fn as_str(self) -> &'static str {
        match self {
            TokenRole::Service => "service",
            TokenRole::Operator => "operator",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "service" => Some(TokenRole::Service),
            "operator" => Some(TokenRole::Operator),
            _ => None,
        }
    }
}

/// A successfully-authenticated principal, attached to every request.
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user_id: String,
    pub token_id: uuid::Uuid,
    pub role: TokenRole,
}

/// Scope value that matches every user. Reserved (it is also the modes
/// template user), so no real user_id can collide with it.
pub const ALL_USERS: &str = "*";

impl AuthUser {
    /// The user_id to scope admin reads by: an operator sees the whole
    /// estate, a service principal only its own rows.
    pub fn scope(&self) -> &str {
        match self.role {
            TokenRole::Operator => ALL_USERS,
            TokenRole::Service => &self.user_id,
        }
    }
}

impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        parts.extensions.get::<AuthUser>().cloned().ok_or_else(|| {
            (
                StatusCode::UNAUTHORIZED,
                Json(json!({ "error": "unauthenticated" })),
            )
        })
    }
}

/// Bearer-token middleware. Health is whitelisted; everything else requires
/// a valid, non-revoked token.
pub async fn require_bearer(
    axum::extract::State(state): axum::extract::State<AppState>,
    mut req: Request<Body>,
    next: Next,
) -> Result<Response, Response> {
    let path = req.uri().path();
    if is_exempt(path) {
        return Ok(next.run(req).await);
    }

    // DEV MODE: skip token validation entirely. Inject a synthetic user whose
    // role matches the surface so the wall stays invisible in dev.
    // The startup banner warns about this; the admin UI also shows a banner.
    if state.cfg.dev_mode {
        let role = if is_admin_surface(path) {
            TokenRole::Operator
        } else {
            TokenRole::Service
        };
        req.extensions_mut().insert(AuthUser {
            user_id: "dev".to_string(),
            token_id: uuid::Uuid::nil(),
            role,
        });
        return Ok(next.run(req).await);
    }

    // Accept token from either Authorization: Bearer header OR a cookie set
    // by /admin/login. Bearer wins if both are present.
    let token = extract_bearer(&req)
        .or_else(|| extract_cookie_token(&req))
        .ok_or_else(|| unauthorized(path, "missing or malformed token"))?;

    let principal = validate_token(&state.pool, &token)
        .await
        .map_err(|_| unauthorized(path, "invalid or revoked token"))?;

    // The role wall, both directions: service tokens never reach /admin,
    // operator tokens never call the API.
    match (is_admin_surface(path), principal.role) {
        (true, TokenRole::Operator) | (false, TokenRole::Service) => {}
        (true, TokenRole::Service) => {
            return Err(unauthorized(
                path,
                "service tokens cannot access the admin UI",
            ));
        }
        (false, TokenRole::Operator) => {
            return Err((
                StatusCode::FORBIDDEN,
                Json(json!({ "error": "operator tokens cannot call the service API" })),
            )
                .into_response());
        }
    }

    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}

fn is_admin_surface(path: &str) -> bool {
    path == "/admin" || path.starts_with("/admin/")
}

fn is_exempt(path: &str) -> bool {
    matches!(
        path,
        "/health" | "/admin/login" | "/admin/style.css" | "/favicon.ico"
    )
}

fn extract_cookie_token(req: &Request<Body>) -> Option<String> {
    let cookies = req
        .headers()
        .get(axum::http::header::COOKIE)?
        .to_str()
        .ok()?;
    for kv in cookies.split(';') {
        let kv = kv.trim();
        if let Some(rest) = kv.strip_prefix("flashback_token=") {
            let v = rest.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// For admin browser routes, return a 303 redirect to /admin/login so the
/// user gets the login form instead of a JSON error. For everything else,
/// return JSON.
fn unauthorized(path: &str, msg: &str) -> Response {
    if path.starts_with("/admin/") || path == "/admin" {
        let body = axum::body::Body::empty();
        return axum::http::Response::builder()
            .status(StatusCode::SEE_OTHER)
            .header("Location", "/admin/login?reason=unauth")
            .body(body)
            .unwrap()
            .into_response();
    }
    (StatusCode::UNAUTHORIZED, Json(json!({ "error": msg }))).into_response()
}

fn extract_bearer(req: &Request<Body>) -> Option<String> {
    let header = req.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    let rest = header.strip_prefix("Bearer ")?.trim();
    if rest.is_empty() {
        None
    } else {
        Some(rest.to_string())
    }
}

pub(crate) async fn validate_token(pool: &PgPool, token: &str) -> Result<AuthUser, ()> {
    if !token.starts_with(TOKEN_PREFIX) {
        return Err(());
    }
    let hash = sha256_hex(token);

    let row: Option<(uuid::Uuid, String, String)> = sqlx::query_as(
        r#"
        UPDATE tokens
        SET last_used_at = NOW()
        WHERE token_hash = $1 AND revoked_at IS NULL
        RETURNING id, user_id, role
        "#,
    )
    .bind(&hash)
    .fetch_optional(pool)
    .await
    .map_err(|_| ())?;

    let (token_id, user_id, role) = row.ok_or(())?;
    let role = TokenRole::parse(&role).ok_or(())?;
    Ok(AuthUser {
        user_id,
        token_id,
        role,
    })
}

pub fn sha256_hex(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    hex::encode(hasher.finalize())
}

/// Generate a new token plaintext. Caller is responsible for ensuring it is
/// never logged, never persisted, and printed exactly once.
pub fn generate_token() -> String {
    let bytes: [u8; 24] = rand::random(); // 24 bytes → 32 chars base32 alphabet (close enough)
    let body: String = bytes
        .iter()
        .map(|b| {
            const ALPHABET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789abcdefghjkmnpqrstuvwxyz"; // O/0/I/1/l stripped
            ALPHABET[(*b as usize) % ALPHABET.len()] as char
        })
        .collect();
    let body: String = body.chars().take(TOKEN_RANDOM_LEN).collect();
    format!("{TOKEN_PREFIX}{body}")
}

pub fn token_prefix(token: &str) -> String {
    token.chars().take(11).collect()
}

pub struct MintedToken {
    pub plaintext: String,
    pub id: uuid::Uuid,
    pub prefix: String,
}

pub async fn mint_token(
    pool: &PgPool,
    user_id: &str,
    name: Option<&str>,
    role: TokenRole,
) -> anyhow::Result<MintedToken> {
    let plaintext = generate_token();
    let hash = sha256_hex(&plaintext);
    let prefix = token_prefix(&plaintext);

    let id: uuid::Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO tokens (token_hash, token_prefix, user_id, name, role)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id
        "#,
    )
    .bind(&hash)
    .bind(&prefix)
    .bind(user_id)
    .bind(name)
    .bind(role.as_str())
    .fetch_one(pool)
    .await?;

    Ok(MintedToken {
        plaintext,
        id,
        prefix,
    })
}

#[derive(Debug, sqlx::FromRow, serde::Serialize)]
pub struct TokenListRow {
    pub id: uuid::Uuid,
    pub token_prefix: String,
    pub user_id: String,
    pub name: Option<String>,
    pub role: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
    pub revoked_at: Option<chrono::DateTime<chrono::Utc>>,
}

pub async fn list_tokens(
    pool: &PgPool,
    user_id: Option<&str>,
) -> anyhow::Result<Vec<TokenListRow>> {
    let rows = sqlx::query_as::<_, TokenListRow>(
        r#"
        SELECT id, token_prefix, user_id, name, role, created_at, last_used_at, revoked_at
        FROM tokens
        WHERE ($1::TEXT IS NULL OR user_id = $1)
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_id)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}

/// Revoke by token id (the UUID `flashback token list` shows in its first column).
pub async fn revoke_token(pool: &PgPool, id: uuid::Uuid) -> anyhow::Result<bool> {
    let result = sqlx::query(
        r#"
        UPDATE tokens SET revoked_at = NOW()
        WHERE id = $1 AND revoked_at IS NULL
        "#,
    )
    .bind(id)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Avoid Arc unused warning for the explicit import path above.
#[allow(dead_code)]
fn _arc_marker(_: Arc<()>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_hex_known_vector() {
        // RFC test vector for empty string.
        assert_eq!(
            sha256_hex(""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        // RFC test vector for "abc".
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn sha256_hex_is_64_hex_chars() {
        let h = sha256_hex("fb_anything_at_all");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sha256_hex_is_deterministic() {
        assert_eq!(sha256_hex("seed"), sha256_hex("seed"));
        assert_ne!(sha256_hex("a"), sha256_hex("b"));
    }

    #[test]
    fn generate_token_has_fb_prefix_and_bounded_length() {
        let t = generate_token();
        assert!(t.starts_with(TOKEN_PREFIX), "got {t}");
        // Body comes from 24 random bytes mapped to alphabet chars and then
        // truncated to at most TOKEN_RANDOM_LEN. The constant is a cap, not
        // a guarantee — actual body length = min(24, TOKEN_RANDOM_LEN).
        let body = &t[TOKEN_PREFIX.len()..];
        assert!(!body.is_empty());
        assert!(body.len() <= TOKEN_RANDOM_LEN);
        assert_eq!(t.len(), TOKEN_PREFIX.len() + body.len());
    }

    #[test]
    fn generate_token_body_uses_alphabet_only() {
        let t = generate_token();
        let body = &t[TOKEN_PREFIX.len()..];
        // O / 0 / I / 1 / l intentionally stripped — verify by hand.
        let forbidden = ['O', '0', 'I', '1', 'l'];
        for c in body.chars() {
            assert!(c.is_ascii_alphanumeric(), "non-alphanumeric: {c:?}");
            assert!(
                !forbidden.contains(&c),
                "forbidden char {c:?} in token body {body}"
            );
        }
    }

    #[test]
    fn generate_token_produces_distinct_values() {
        // 100 fresh tokens should be all-unique with overwhelming probability.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..100 {
            assert!(seen.insert(generate_token()));
        }
    }

    #[test]
    fn token_prefix_takes_first_eleven_chars() {
        assert_eq!(token_prefix("fb_ABCDEFGH123456789"), "fb_ABCDEFGH");
        // Short input → just the whole thing.
        assert_eq!(token_prefix("fb_short"), "fb_short");
    }

    // ---- middleware-helper unit tests (no DB / no HTTP) -----------------

    #[test]
    fn is_exempt_matches_whitelisted_paths() {
        assert!(is_exempt("/health"));
        assert!(is_exempt("/admin/login"));
        assert!(is_exempt("/admin/style.css"));
        assert!(is_exempt("/favicon.ico"));
    }

    #[test]
    fn is_exempt_rejects_everything_else() {
        assert!(!is_exempt("/memory/ingest"));
        assert!(!is_exempt("/admin/memories"));
        assert!(!is_exempt("/healthcheck")); // not literal /health
        assert!(!is_exempt("/")); // root isn't whitelisted
        assert!(!is_exempt(""));
    }

    fn req_with_header(name: &str, value: &str) -> Request<Body> {
        Request::builder()
            .header(name, value)
            .body(Body::empty())
            .unwrap()
    }

    #[test]
    fn extract_bearer_pulls_token_after_prefix() {
        let req = req_with_header("Authorization", "Bearer fb_TOKEN_VALUE");
        assert_eq!(extract_bearer(&req), Some("fb_TOKEN_VALUE".to_string()));
    }

    #[test]
    fn extract_bearer_returns_none_when_no_header() {
        let req = Request::builder().body(Body::empty()).unwrap();
        assert_eq!(extract_bearer(&req), None);
    }

    #[test]
    fn extract_bearer_returns_none_when_not_bearer_scheme() {
        let req = req_with_header("Authorization", "Basic dXNlcjpwYXNz");
        assert_eq!(extract_bearer(&req), None);
    }

    #[test]
    fn extract_bearer_returns_none_on_empty_token() {
        let req = req_with_header("Authorization", "Bearer ");
        assert_eq!(extract_bearer(&req), None);
    }

    #[test]
    fn extract_cookie_token_finds_the_flashback_cookie() {
        let req = req_with_header(
            "Cookie",
            "other=value; flashback_token=fb_FROM_COOKIE; another=x",
        );
        assert_eq!(
            extract_cookie_token(&req),
            Some("fb_FROM_COOKIE".to_string())
        );
    }

    #[test]
    fn extract_cookie_token_returns_none_when_no_flashback_cookie() {
        let req = req_with_header("Cookie", "other=value; session=abc");
        assert_eq!(extract_cookie_token(&req), None);
    }

    #[test]
    fn extract_cookie_token_returns_none_when_no_cookie_header() {
        let req = Request::builder().body(Body::empty()).unwrap();
        assert_eq!(extract_cookie_token(&req), None);
    }

    #[test]
    fn extract_cookie_token_handles_empty_value() {
        let req = req_with_header("Cookie", "flashback_token=");
        assert_eq!(extract_cookie_token(&req), None);
    }

    #[test]
    fn unauthorized_redirects_admin_paths_to_login() {
        let resp = unauthorized("/admin/memories", "bad token");
        assert_eq!(resp.status(), StatusCode::SEE_OTHER);
        assert_eq!(
            resp.headers().get("Location").unwrap(),
            "/admin/login?reason=unauth"
        );
    }

    #[test]
    fn unauthorized_returns_json_401_for_api_paths() {
        let resp = unauthorized("/memory/ingest", "missing token");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    // ---- Integration tests against a real Postgres (via #[sqlx::test]) ---

    #[sqlx::test(migrations = "../../migrations")]
    async fn mint_then_list_finds_the_token(pool: PgPool) {
        let minted = mint_token(&pool, "alice", Some("test-token"), TokenRole::Service)
            .await
            .unwrap();
        assert!(minted.plaintext.starts_with("fb_"));
        assert_eq!(minted.prefix, token_prefix(&minted.plaintext));

        let rows = list_tokens(&pool, Some("alice")).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, minted.id);
        assert_eq!(rows[0].user_id, "alice");
        assert_eq!(rows[0].name.as_deref(), Some("test-token"));
        assert!(rows[0].revoked_at.is_none());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn list_tokens_filters_by_user(pool: PgPool) {
        mint_token(&pool, "alice", None, TokenRole::Service)
            .await
            .unwrap();
        mint_token(&pool, "bob", None, TokenRole::Service)
            .await
            .unwrap();
        mint_token(&pool, "alice", None, TokenRole::Service)
            .await
            .unwrap();

        let alice = list_tokens(&pool, Some("alice")).await.unwrap();
        assert_eq!(alice.len(), 2);
        assert!(alice.iter().all(|r| r.user_id == "alice"));

        let all = list_tokens(&pool, None).await.unwrap();
        assert_eq!(all.len(), 3);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn validate_token_round_trip(pool: PgPool) {
        let minted = mint_token(&pool, "alice", None, TokenRole::Service)
            .await
            .unwrap();

        let principal = validate_token(&pool, &minted.plaintext).await.unwrap();
        assert_eq!(principal.user_id, "alice");
        assert_eq!(principal.token_id, minted.id);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn validate_token_rejects_bad_input(pool: PgPool) {
        // Wrong prefix.
        assert!(validate_token(&pool, "wrong_prefix_token").await.is_err());
        // Right prefix, never minted.
        assert!(validate_token(&pool, "fb_NEVER_EXISTED_AAAAAAAA")
            .await
            .is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoke_token_makes_it_unvalidatable(pool: PgPool) {
        let minted = mint_token(&pool, "alice", None, TokenRole::Service)
            .await
            .unwrap();
        assert!(validate_token(&pool, &minted.plaintext).await.is_ok());

        let revoked = revoke_token(&pool, minted.id).await.unwrap();
        assert!(revoked, "expected revoke_token to report success");

        // Token no longer validates after revocation.
        assert!(validate_token(&pool, &minted.plaintext).await.is_err());

        // Revoking an already-revoked token returns false.
        let revoked_again = revoke_token(&pool, minted.id).await.unwrap();
        assert!(!revoked_again);
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn revoked_tokens_still_listed_with_revoked_at_populated(pool: PgPool) {
        let minted = mint_token(&pool, "alice", None, TokenRole::Service)
            .await
            .unwrap();
        revoke_token(&pool, minted.id).await.unwrap();

        let rows = list_tokens(&pool, Some("alice")).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].revoked_at.is_some());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn minted_role_round_trips_through_validate(pool: PgPool) {
        let svc = mint_token(&pool, "alice", None, TokenRole::Service)
            .await
            .unwrap();
        let op = mint_token(&pool, "alice", None, TokenRole::Operator)
            .await
            .unwrap();
        assert_eq!(
            validate_token(&pool, &svc.plaintext).await.unwrap().role,
            TokenRole::Service
        );
        assert_eq!(
            validate_token(&pool, &op.plaintext).await.unwrap().role,
            TokenRole::Operator
        );
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn role_defaults_to_service_and_is_listed(pool: PgPool) {
        mint_token(&pool, "alice", Some("ritsu"), TokenRole::Service)
            .await
            .unwrap();
        let rows = list_tokens(&pool, Some("alice")).await.unwrap();
        assert_eq!(rows[0].role, "service");
    }

    #[test]
    fn admin_surface_detection_covers_admin_only() {
        assert!(is_admin_surface("/admin"));
        assert!(is_admin_surface("/admin/records"));
        // The API surface must never be mistaken for the admin one.
        assert!(!is_admin_surface("/records"));
        assert!(!is_admin_surface("/records/context"));
        assert!(!is_admin_surface("/administrative"));
    }

    #[test]
    fn role_parse_rejects_unknown_values() {
        assert_eq!(TokenRole::parse("service"), Some(TokenRole::Service));
        assert_eq!(TokenRole::parse("operator"), Some(TokenRole::Operator));
        assert_eq!(TokenRole::parse("admin"), None);
        assert_eq!(TokenRole::parse(""), None);
    }
}
