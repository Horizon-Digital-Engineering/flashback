//! Flashback MCP server — Streamable-HTTP transport, built on rmcp.
//!
//! Wraps the Flashback REST API as typed MCP tools. Bearer tokens flow
//! end-to-end: the MCP client sends `Authorization: Bearer <flashback-token>`,
//! we read it from the request `Parts` in `RequestContext::extensions`, and
//! pass the same value through to every downstream Flashback call.

use std::{net::SocketAddr, sync::Arc};

use anyhow::Result;
use axum::{response::IntoResponse, routing::get, Json, Router};
use reqwest::Client;
use rmcp::{
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{
        CallToolResult, ContentBlock, Implementation, ProtocolVersion, ServerCapabilities,
        ServerInfo,
    },
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::{
        streamable_http_server::{
            session::local::LocalSessionManager, tower::StreamableHttpService,
        },
        StreamableHttpServerConfig,
    },
    ErrorData as McpError, RoleServer, ServerHandler,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

// ---------------------------------------------------------------------------
// Typed argument schemas
// ---------------------------------------------------------------------------

/// schemars 1.0 renders `serde_json::Value` fields as the bare boolean `true`
/// (the JSON Schema 2020-12 "any value" sentinel). Claude Code's MCP client
/// uses zod, which rejects boolean-shaped property schemas and fails
/// `tools/list` validation — dropping ALL tools, not just the offending one.
/// This helper substitutes an empty-object schema (also "any value" in JSON
/// Schema, but valid under zod) for fields we want to keep untyped.
fn any_json_schema(_generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
    schemars::json_schema!({})
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct LineageArgs {
    /// UUID of the raw record whose supersede chain to walk.
    pub record_id: String,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct StateSetArgs {
    /// The reference kind, e.g. "todo_list" or any app-defined kind.
    pub kind: String,
    /// The unique key within that kind (each (kind, key) is one reference).
    pub state_key: String,
    /// The COMPLETE new current value for this reference (never a delta). A new
    /// append-only state_object raw record is written superseding the prior one.
    #[schemars(schema_with = "any_json_schema")]
    pub data: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct StateLookupArgs {
    pub kind: String,
    pub state_key: String,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct StateListArgs {
    pub kind: String,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Flashback {
    flashback_url: Arc<String>,
    http: Client,
    // Consumed by the #[tool_router] macro below via method dispatch the
    // compiler can't see directly; not dead code despite what -D warnings says.
    #[allow(dead_code)]
    tool_router: ToolRouter<Flashback>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct RecordArgs {
    /// conversation | document | state_object — how the record must be
    /// PROCESSED, not what it is about. Tier names (episodic, semantic,
    /// summary) are conclusions the curation pass draws, never inputs.
    pub r#type: String,
    pub content: String,
    /// Origin tag, e.g. "chatgpt", "ritsu:health", "finance-sync".
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
    /// Your own id for this record, and for the one it followed. The store
    /// resolves the second to an internal id, so callers never handle ours.
    /// An unresolvable one is kept as a visible gap, not an error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev_source_ref: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct RecallArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topic_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct ProposeArgs {
    /// A short headline for the proposal.
    pub title: String,
    /// The proposed action to take (or, for kind="insight", the insight itself).
    pub action: String,
    /// Optional "action" | "insight". Defaults to "action".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Why — the reasoning behind the proposal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
    /// Raw record ids (uuids) that justify the proposal. Each must be one of the
    /// caller's own records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_ids: Option<Vec<String>>,
}

#[tool_router]
impl Flashback {
    pub fn new(flashback_url: String) -> Self {
        Self {
            flashback_url: Arc::new(flashback_url),
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("reqwest client"),
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Store a raw record in Flashback's immutable raw layer: a conversation \
                       turn, fact, document, event, or transaction. Universal typed record — set \
                       `type` (episodic/semantic/working/document/procedural/state_object) and a \
                       `source` tag. Optional `mode` pins the cognitive register \
                       (code/general/journal/research/…) it's embedded and recalled in; omitted, \
                       it's auto-classified or falls back to the default register. Append-only; \
                       never overwrites."
    )]
    async fn flashback_record(
        &self,
        Parameters(args): Parameters<RecordArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let body = to_json(&args)?;
        let res = self.post("/records", &bearer, body).await?;
        result_ok(res)
    }

    #[tool(
        description = "Recall the most relevant raw records for a query via hybrid semantic + \
                       keyword retrieval over the immutable raw layer, scoped by \
                       topic/thread/mode. `mode` picks a cognitive register \
                       (code/general/journal/research/…) and searches its vector geometry; \
                       `all` searches across registers with keyword-degraded ranking. Call \
                       before answering to ground in memory."
    )]
    async fn flashback_recall(
        &self,
        Parameters(args): Parameters<RecallArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let body = to_json(&args)?;
        let res = self.post("/records/context", &bearer, body).await?;
        result_ok(res)
    }

    #[tool(
        description = "List the data catalog: every store the lake knows about, grouped by kind \
                       (raw / curated / operational / external), each with its schema, current \
                       record count, and lineage. The answer to 'is my data organized and can I \
                       see it?'. The raw + curated layers register automatically."
    )]
    async fn flashback_catalog(
        &self,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let res = self.get("/catalog", &bearer).await?;
        result_ok(res)
    }

    #[tool(
        description = "Propose an action or insight for the operator to decide on — Flashback \
                       proposes, it never acts. Cite the raw record ids that justify it via \
                       `evidence_ids`. The proposal lands as 'proposed'; a human approves or \
                       denies it, and the host (not Flashback) carries out any approved action."
    )]
    async fn flashback_propose(
        &self,
        Parameters(args): Parameters<ProposeArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let body = json!({
            "title": args.title,
            "action": args.action,
            "kind": args.kind,
            "rationale": args.rationale,
            "evidence": args.evidence_ids.unwrap_or_default(),
        });
        let res = self.post("/proposals", &bearer, body).await?;
        result_ok(res)
    }

    #[tool(
        description = "Walk the supersede chain (back + forward) for a raw record id — how the \
                       value evolved. Corrections are new append-only rows; this reconstructs \
                       the full lineage."
    )]
    async fn flashback_lineage(
        &self,
        Parameters(args): Parameters<LineageArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/records/{}/lineage", args.record_id);
        let res = self.get(&path, &bearer).await?;
        result_ok(res)
    }

    #[tool(
        description = "Set the current value of a reference — a named mutable cell (a todo list, \
                       a plan, a config) keyed by (kind, state_key). Pass the COMPLETE new value \
                       in `data`; a new append-only state_object raw record is written that \
                       supersedes the prior one, so history is preserved and raw stays immutable. \
                       Creates the reference on first set."
    )]
    async fn flashback_state_set(
        &self,
        Parameters(args): Parameters<StateSetArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!(
            "/records/state/{}/{}",
            enc_segment(&args.kind),
            enc_segment(&args.state_key)
        );
        let body = json!({
            "data":       args.data,
            "topic_id": args.topic_id,
            "thread_id": args.thread_id,
            "importance": args.importance,
        });
        let res = self.post(&path, &bearer, body).await?;
        result_ok(res)
    }

    #[tool(description = "Get the current (terminal) value of a reference.")]
    async fn flashback_state_get(
        &self,
        Parameters(args): Parameters<StateLookupArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!(
            "/records/state/{}/{}",
            enc_segment(&args.kind),
            enc_segment(&args.state_key)
        );
        let res = self.get(&path, &bearer).await?;
        result_ok(res)
    }

    #[tool(description = "List the terminal value of every reference of a given kind.")]
    async fn flashback_state_list(
        &self,
        Parameters(args): Parameters<StateListArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/records/state/{}", enc_segment(&args.kind));
        let res = self.get(&path, &bearer).await?;
        result_ok(res)
    }

    #[tool(description = "Return the full supersede chain for a reference — how it evolved.")]
    async fn flashback_state_history(
        &self,
        Parameters(args): Parameters<StateLookupArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!(
            "/records/state/{}/{}/history",
            enc_segment(&args.kind),
            enc_segment(&args.state_key)
        );
        let res = self.get(&path, &bearer).await?;
        result_ok(res)
    }

    // ---- helpers ----

    async fn post(&self, path: &str, bearer: &str, body: Value) -> Result<Value, McpError> {
        let url = format!("{}{}", self.flashback_url, path);
        let resp = self
            .http
            .post(&url)
            .bearer_auth(bearer)
            .json(&body)
            .send()
            .await
            .map_err(|e| internal(format!("POST {path}: {e}")))?;
        decode_resp(resp, path).await
    }

    async fn get(&self, path: &str, bearer: &str) -> Result<Value, McpError> {
        let url = format!("{}{}", self.flashback_url, path);
        let resp = self
            .http
            .get(&url)
            .bearer_auth(bearer)
            .send()
            .await
            .map_err(|e| internal(format!("GET {path}: {e}")))?;
        decode_resp(resp, path).await
    }
}

#[tool_handler]
impl ServerHandler for Flashback {
    fn get_info(&self) -> ServerInfo {
        // Implementation::from_build_env() resolves env! at rmcp's call-site,
        // so it returns "rmcp" / its version. Override with ours so MCP clients
        // display the right server name.
        let mut server_impl = Implementation::from_build_env();
        server_impl.name = env!("CARGO_PKG_NAME").to_string();
        server_impl.version = env!("CARGO_PKG_VERSION").to_string();
        server_impl.title = Some("Flashback memory MCP".to_string());
        server_impl.website_url =
            Some("https://github.com/Horizon-Digital-Engineering/flashback".to_string());

        let mut info = ServerInfo::default();
        info.protocol_version = ProtocolVersion::V_2025_06_18;
        info.capabilities = ServerCapabilities::builder().enable_tools().build();
        info.server_info = server_impl;
        info.instructions = Some(
            "Persistent typed memory for AI assistants. Every tool call must include \
             `Authorization: Bearer <flashback-token>`. Mint tokens server-side with \
             `flashback token mint --user=<user> --name=<label>`."
                .into(),
        );
        info
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn bearer_or_err(ctx: &RequestContext<RoleServer>) -> Result<String, McpError> {
    let parts = ctx
        .extensions
        .get::<axum::http::request::Parts>()
        .ok_or_else(|| {
            McpError::invalid_request("MCP server is not running over HTTP transport", None)
        })?;
    let header = parts
        .headers
        .get(axum::http::header::AUTHORIZATION)
        .ok_or_else(|| {
            McpError::invalid_request(
                "missing Authorization: Bearer <flashback-token> header",
                None,
            )
        })?;
    let s = header
        .to_str()
        .map_err(|_| McpError::invalid_request("Authorization header is not ASCII", None))?;
    let token = s
        .strip_prefix("Bearer ")
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| McpError::invalid_request("Authorization must be a Bearer token", None))?;
    Ok(token.to_string())
}

fn to_json<T: Serialize>(value: &T) -> Result<Value, McpError> {
    serde_json::to_value(value).map_err(|e| internal(format!("serialize args: {e}")))
}

/// Percent-encode one URL path segment so a `kind`/`state_key` containing `/`,
/// `?`, `#`, `%`, or other path-significant bytes can't reshape the request
/// path (it stays an opaque single segment the server matches on). Unreserved
/// characters (RFC 3986 `ALPHA / DIGIT / -._~`) pass through untouched.
fn enc_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn internal(msg: impl Into<String>) -> McpError {
    McpError::internal_error(msg.into(), None)
}

async fn decode_resp(resp: reqwest::Response, path: &str) -> Result<Value, McpError> {
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(McpError::internal_error(
            format!("upstream {path} returned {status}: {text}"),
            None,
        ));
    }
    if text.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_str(&text).map_err(|e| internal(format!("decode response from {path}: {e}")))
}

fn result_ok(value: Value) -> Result<CallToolResult, McpError> {
    // Serialize as text content for clients that only consume `content`.
    // Most up-to-date MCP clients ALSO read `structured_content` when present.
    let text = serde_json::to_string(&value).unwrap_or_else(|_| "{}".into());
    let mut result = CallToolResult::success(vec![ContentBlock::text(text)]);
    result.structured_content = Some(value);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Entrypoint
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| "flashback_mcp=info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let flashback_url = Arc::new(
        std::env::var("FLASHBACK_URL").unwrap_or_else(|_| "http://localhost:8080".to_string()),
    );
    // Loopback unless asked otherwise. A default of 0.0.0.0 means every machine
    // that never sets MCP_HOST publishes this port to whatever can route to it,
    // and the ones that most need the default to be right are the ones that
    // never touch the config.
    let host = std::env::var("MCP_HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
    let port: u16 = std::env::var("MCP_PORT")
        .unwrap_or_else(|_| "8082".to_string())
        .parse()
        .unwrap_or(8082);

    let app = build_router(flashback_url.clone());

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    if host == "0.0.0.0" {
        tracing::warn!(
            "listening on all interfaces ({addr}) — every host that can route here \
             reaches this endpoint, and it holds the same memory the API does. \
             Bind a specific address (MCP_HOST=...) unless that is intended."
        );
    }
    tracing::info!("flashback-mcp listening on http://{addr}/mcp");

    axum::serve(listener, app).await?;
    Ok(())
}

/// The wrapper's HTTP surface. No CORS layer, for the reason the API server
/// carries none either: every client of this endpoint — an assistant host, the
/// API sidecar — is server-side, where CORS does not apply, and a permissive
/// layer only ever widens what a page you merely visit can do to a port on
/// your own machine. `/health` names the upstream URL, which is an internal
/// hostname on most installs.
fn build_router(flashback_url: Arc<String>) -> Router {
    let url_for_factory = flashback_url.clone();
    let svc: StreamableHttpService<Flashback, LocalSessionManager> = StreamableHttpService::new(
        move || Ok(Flashback::new((*url_for_factory).clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    let url_for_health = flashback_url;
    Router::new()
        .route(
            "/health",
            get(move || {
                let url = url_for_health.clone();
                async move {
                    let client = reqwest::Client::new();
                    let upstream_ok = client
                        .get(format!("{}/health", *url))
                        .send()
                        .await
                        .map(|r| r.status().is_success())
                        .unwrap_or(false);
                    health_response(upstream_ok, &url)
                }
            }),
        )
        .nest_service("/mcp", svc)
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

fn health_response(upstream_ok: bool, url: &str) -> impl IntoResponse {
    Json(json!({
        "status": if upstream_ok { "ok" } else { "degraded" },
        "service": "flashback-mcp",
        "upstream": url,
        "upstream_ok": upstream_ok,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    async fn stub_upstream(
        handler: axum::routing::MethodRouter,
        path: &str,
    ) -> (String, tokio::task::JoinHandle<()>) {
        let app = Router::new().route(path, handler);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let jh = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), jh)
    }

    #[tokio::test]
    async fn the_wrapper_forwards_the_callers_token_and_nothing_else() {
        let (url, jh) = stub_upstream(
            axum::routing::post(|headers: axum::http::HeaderMap, body: String| async move {
                Json(json!({
                    "auth": headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or_default(),
                    "cookie": headers.contains_key("cookie"),
                    "echo": body,
                }))
            }),
            "/records",
        )
        .await;

        let fb = Flashback::new(url);
        let out = fb
            .post("/records", "fb_tok", json!({ "content": "hi" }))
            .await
            .unwrap();
        assert_eq!(out["auth"], "Bearer fb_tok");
        assert_eq!(out["cookie"], false);
        assert_eq!(out["echo"], r#"{"content":"hi"}"#);
        jh.abort();
    }

    #[tokio::test]
    async fn an_upstream_refusal_is_reported_without_the_token() {
        let (url, jh) = stub_upstream(
            axum::routing::get(|| async { (StatusCode::UNAUTHORIZED, "invalid or revoked token") }),
            "/records",
        )
        .await;

        let fb = Flashback::new(url);
        let err = fb.get("/records", "fb_secret_token").await.unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("401"), "{msg}");
        assert!(msg.contains("/records"), "the path has to be in the error");
        assert!(
            !msg.contains("fb_secret_token"),
            "the token must not travel into an error that gets logged"
        );
        jh.abort();
    }

    #[tokio::test]
    async fn an_empty_body_is_null_and_a_non_json_body_is_an_error() {
        let (url, jh) = stub_upstream(axum::routing::get(|| async { "" }), "/records").await;
        let fb = Flashback::new(url);
        assert_eq!(fb.get("/records", "t").await.unwrap(), Value::Null);
        jh.abort();

        let (url, jh) =
            stub_upstream(axum::routing::get(|| async { "not json" }), "/records").await;
        let fb = Flashback::new(url);
        assert!(fb.get("/records", "t").await.is_err());
        jh.abort();
    }

    #[tokio::test]
    async fn an_unreachable_upstream_names_the_path_it_could_not_reach() {
        let fb = Flashback::new("http://127.0.0.1:1".to_string());
        let err = fb.get("/records", "t").await.unwrap_err();
        assert!(format!("{err:?}").contains("/records"));
    }

    #[tokio::test]
    async fn health_is_open_but_carries_no_cross_origin_grant() {
        // A page you merely visit must not be able to read a port on your own
        // machine; /health names the upstream host.
        let app = build_router(Arc::new("http://127.0.0.1:1".to_string()));
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .header("origin", "https://evil.example")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "the wrapper handed out a cross-origin grant"
        );
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["status"], "degraded");
        assert_eq!(body["upstream_ok"], false);
    }

    #[tokio::test]
    async fn the_mcp_endpoint_is_mounted() {
        let app = build_router(Arc::new("http://127.0.0.1:1".to_string()));
        let resp = app
            .oneshot(Request::builder().uri("/mcp").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_ne!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn a_tool_result_carries_the_same_value_twice() {
        // Older clients read `content`, current ones read `structured_content`;
        // a client that reads only one must not see less.
        let v = json!({ "id": "abc", "seq": 4 });
        let r = result_ok(v.clone()).unwrap();
        assert_eq!(r.structured_content, Some(v.clone()));
        let text = serde_json::to_string(&r.content).unwrap();
        assert!(text.contains("abc"), "{text}");
    }

    #[test]
    fn enc_segment_passes_unreserved_through() {
        assert_eq!(enc_segment("todo_list"), "todo_list");
        assert_eq!(enc_segment("a-b.c_d~e"), "a-b.c_d~e");
        assert_eq!(enc_segment("Key123"), "Key123");
    }

    #[test]
    fn enc_segment_escapes_path_significant_bytes() {
        // A key that tries to escape its segment can't reshape the path.
        assert_eq!(enc_segment("../../catalog"), "..%2F..%2Fcatalog");
        assert_eq!(enc_segment("foo/history"), "foo%2Fhistory");
        assert_eq!(enc_segment("a b"), "a%20b");
        assert_eq!(enc_segment("q?x=1#f"), "q%3Fx%3D1%23f");
        assert_eq!(enc_segment("100%"), "100%25");
    }

    #[test]
    fn enc_segment_escapes_non_ascii() {
        // Multi-byte UTF-8 is percent-encoded byte-by-byte (no panic, no raw bytes).
        assert_eq!(enc_segment("café"), "caf%C3%A9");
    }
}
