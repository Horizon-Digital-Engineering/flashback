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
        CallToolResult, Content, Implementation, ProtocolVersion, ServerCapabilities, ServerInfo,
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
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
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
    /// episodic | semantic | working | document | procedural | state_object
    pub r#type: String,
    pub content: String,
    /// Origin tag, e.g. "chatgpt", "ritsu:health", "finance-sync".
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_hours: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct RecallArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
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
                       project/session/mode. `mode` picks a cognitive register \
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
        let path = format!("/records/state/{}/{}", args.kind, args.state_key);
        let body = json!({
            "data":       args.data,
            "project_id": args.project_id,
            "session_id": args.session_id,
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
        let path = format!("/records/state/{}/{}", args.kind, args.state_key);
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
        let path = format!("/records/state/{}", args.kind);
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
        let path = format!("/records/state/{}/{}/history", args.kind, args.state_key);
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
    let mut result = CallToolResult::success(vec![Content::text(text)]);
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
    let host = std::env::var("MCP_HOST").unwrap_or_else(|_| "0.0.0.0".to_string());
    let port: u16 = std::env::var("MCP_PORT")
        .unwrap_or_else(|_| "8082".to_string())
        .parse()
        .unwrap_or(8082);

    // Per-session handler instance.
    let url_for_factory = flashback_url.clone();
    let svc: StreamableHttpService<Flashback, LocalSessionManager> = StreamableHttpService::new(
        move || Ok(Flashback::new((*url_for_factory).clone())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );

    let url_for_health = flashback_url.clone();
    let app: Router = Router::new()
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
        .layer(tower_http::cors::CorsLayer::permissive())
        .layer(tower_http::trace::TraceLayer::new_for_http());

    let addr: SocketAddr = format!("{host}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("flashback-mcp listening on http://{addr}/mcp");

    axum::serve(listener, app).await?;
    Ok(())
}

fn health_response(upstream_ok: bool, url: &str) -> impl IntoResponse {
    Json(json!({
        "status": if upstream_ok { "ok" } else { "degraded" },
        "service": "flashback-mcp",
        "upstream": url,
        "upstream_ok": upstream_ok,
    }))
}
