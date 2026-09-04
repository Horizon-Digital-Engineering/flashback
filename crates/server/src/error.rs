use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Sidecar error: {0}")]
    Sidecar(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl AppError {
    pub fn sidecar(msg: impl Into<String>) -> Self {
        AppError::Sidecar(msg.into())
    }

    pub fn bad_request(msg: impl Into<String>) -> Self {
        AppError::BadRequest(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        AppError::NotFound(msg.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            // Not every database error is our fault. A missing row means the
            // caller named something that is not there, and a unique violation
            // means they sent it twice — reporting both as 500 tells them the
            // server is broken and invites a retry loop that can never succeed.
            AppError::Database(sqlx::Error::RowNotFound) => {
                (StatusCode::NOT_FOUND, "Not found".to_string())
            }
            AppError::Database(e) => {
                let code = e.as_database_error().and_then(|d| d.code());
                match code.as_deref() {
                    Some("23505") => (StatusCode::CONFLICT, "Already exists".to_string()),
                    Some("23503") => (
                        StatusCode::BAD_REQUEST,
                        "References something that does not exist".to_string(),
                    ),
                    _ => {
                        tracing::error!("Database error: {e}");
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            "Database error".to_string(),
                        )
                    }
                }
            }
            AppError::Sidecar(msg) => {
                tracing::error!("Sidecar error: {msg}");
                (StatusCode::BAD_GATEWAY, format!("Sidecar: {msg}"))
            }
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, msg.clone()),
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            AppError::Conflict(msg) => (StatusCode::CONFLICT, msg.clone()),
            AppError::Internal(e) => {
                tracing::error!("Internal error: {e}");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Internal server error".to_string(),
                )
            }
        };
        (status, Json(json!({ "error": message }))).into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    async fn body_of(e: AppError) -> (StatusCode, String) {
        let r = e.into_response();
        let status = r.status();
        let bytes = to_bytes(r.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8_lossy(&bytes).to_string())
    }

    #[tokio::test]
    async fn client_errors_keep_their_message() {
        let (s, b) = body_of(AppError::bad_request("topic_id must be a string")).await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        assert!(
            b.contains("topic_id must be a string"),
            "a 400 has to say what was wrong"
        );

        let (s, b) = body_of(AppError::not_found("record 7")).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        assert!(b.contains("record 7"));

        let (s, b) = body_of(AppError::Conflict("already exists".into())).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert!(b.contains("already exists"));
    }

    #[tokio::test]
    async fn server_errors_never_leak_what_went_wrong() {
        // The message reaches an unauthenticated caller on some paths, and an
        // internal error chain routinely carries a host, a role or a constraint
        // name. Markers stand in for those so this test is not itself a place
        // where something credential-shaped lives.
        let (s, b) = body_of(AppError::Internal(anyhow::anyhow!(
            "connect failed for HOST-MARKER as ROLE-MARKER"
        )))
        .await;
        assert_eq!(s, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(
            !b.contains("HOST-MARKER"),
            "internal detail reached the body"
        );
        assert!(
            !b.contains("ROLE-MARKER"),
            "internal detail reached the body"
        );
        assert!(
            !b.contains("connect failed"),
            "the error chain itself reached the body"
        );
    }

    #[tokio::test]
    async fn a_missing_row_is_not_a_server_error() {
        // fetch_one on an absent row yields RowNotFound. Reporting that as 500
        // tells a caller the server is broken when their id was simply wrong.
        let (s, _) = body_of(AppError::Database(sqlx::Error::RowNotFound)).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn a_sidecar_failure_is_a_bad_gateway() {
        let (s, _) = body_of(AppError::sidecar("embedder timed out")).await;
        assert_eq!(s, StatusCode::BAD_GATEWAY);
    }
}
