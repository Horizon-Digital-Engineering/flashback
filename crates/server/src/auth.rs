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
use rand::RngCore;
use serde_json::json;
use sha2::{Digest, Sha256};
use sqlx::PgPool;

use crate::AppState;

const TOKEN_PREFIX: &str = "fb_";
const TOKEN_RANDOM_LEN: usize = 32;

/// A successfully-authenticated principal, attached to every request.
#[derive(Debug, Clone)]
pub struct AuthUser {
    pub user_id: String,
    pub token_id: uuid::Uuid,
}

#[axum::async_trait]
impl<S> FromRequestParts<S> for AuthUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, Json<serde_json::Value>);

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        parts
            .extensions
            .get::<AuthUser>()
            .cloned()
            .ok_or_else(|| {
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

    // DEV MODE: skip token validation entirely. Inject a synthetic user.
    // The startup banner warns about this; the admin UI also shows a banner.
    if state.cfg.dev_mode {
        req.extensions_mut().insert(AuthUser {
            user_id: "dev".to_string(),
            token_id: uuid::Uuid::nil(),
        });
        return Ok(next.run(req).await);
    }

    // Accept token from either Authorization: Bearer header OR a cookie set
    // by /admin/login. Bearer wins if both are present.
    let token = extract_bearer(&req)
        .or_else(|| extract_cookie_token(&req))
        .ok_or_else(|| unauthorized(path, "missing or malformed token"))?;

    let principal = validate_token(&state.pool, &token).await.map_err(|_| {
        unauthorized(path, "invalid or revoked token")
    })?;

    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}

fn is_exempt(path: &str) -> bool {
    matches!(
        path,
        "/health" | "/admin/login" | "/admin/style.css" | "/favicon.ico"
    )
}

fn extract_cookie_token(req: &Request<Body>) -> Option<String> {
    let cookies = req.headers().get(axum::http::header::COOKIE)?.to_str().ok()?;
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
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "error": msg })),
    )
        .into_response()
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

async fn validate_token(pool: &PgPool, token: &str) -> Result<AuthUser, ()> {
    if !token.starts_with(TOKEN_PREFIX) {
        return Err(());
    }
    let hash = sha256_hex(token);

    let row: Option<(uuid::Uuid, String)> = sqlx::query_as(
        r#"
        UPDATE tokens
        SET last_used_at = NOW()
        WHERE token_hash = $1 AND revoked_at IS NULL
        RETURNING id, user_id
        "#,
    )
    .bind(&hash)
    .fetch_optional(pool)
    .await
    .map_err(|_| ())?;

    let (token_id, user_id) = row.ok_or(())?;
    Ok(AuthUser { user_id, token_id })
}

pub fn sha256_hex(s: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(s.as_bytes());
    hex::encode(hasher.finalize())
}

/// Generate a new token plaintext. Caller is responsible for ensuring it is
/// never logged, never persisted, and printed exactly once.
pub fn generate_token() -> String {
    let mut bytes = [0u8; 24]; // 24 bytes → 32 chars base32 alphabet (close enough)
    rand::thread_rng().fill_bytes(&mut bytes);
    let body: String = bytes
        .iter()
        .map(|b| {
            const ALPHABET: &[u8] =
                b"ABCDEFGHJKMNPQRSTUVWXYZ23456789abcdefghjkmnpqrstuvwxyz"; // O/0/I/1/l stripped
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
) -> anyhow::Result<MintedToken> {
    let plaintext = generate_token();
    let hash = sha256_hex(&plaintext);
    let prefix = token_prefix(&plaintext);

    let id: uuid::Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO tokens (token_hash, token_prefix, user_id, name)
        VALUES ($1, $2, $3, $4)
        RETURNING id
        "#,
    )
    .bind(&hash)
    .bind(&prefix)
    .bind(user_id)
    .bind(name)
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
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used_at: Option<chrono::DateTime<chrono::Utc>>,
    pub revoked_at: Option<chrono::DateTime<chrono::Utc>>,
}

pub async fn list_tokens(pool: &PgPool, user_id: Option<&str>) -> anyhow::Result<Vec<TokenListRow>> {
    let rows = sqlx::query_as::<_, TokenListRow>(
        r#"
        SELECT id, token_prefix, user_id, name, created_at, last_used_at, revoked_at
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
