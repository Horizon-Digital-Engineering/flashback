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
pub struct RememberArgs {
    /// Raw text to remember. Either this OR user_turn/assistant_turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_turn: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assistant_turn: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// One of: working, episodic, semantic, procedural. Defaults to "working".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_hours: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct SearchArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// "answer" (relevance-weighted) or "manager" (situational-awareness-weighted)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_types: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_superseded: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct AssembleArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recent_turns_floor: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct SupersedeArgs {
    /// UUID of the memory being superseded.
    pub memory_id: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct LineageArgs {
    pub memory_id: String,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct CoreAddArgs {
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub importance: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, schemars::JsonSchema)]
pub struct StateCreateArgs {
    /// Currently supported: "todo_list".
    pub kind: String,
    pub state_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "any_json_schema")]
    pub initial: Option<Value>,
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
pub struct StatePatchArgs {
    pub kind: String,
    pub state_key: String,
    /// For kind="todo_list": "add" | "mark_done" | "unmark" | "toggle" |
    /// "remove" | "update" | "reorder" | "replace" | "clear".
    pub op: String,
    /// Required by: add, update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Required by: mark_done, unmark, toggle, remove, update.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
    /// Required by: reorder (full list of existing item ids in new order).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ids: Option<Vec<String>>,
    /// Required by: replace (new full item list).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(schema_with = "any_json_schema")]
    pub items: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct Flashback {
    flashback_url: Arc<String>,
    http: Client,
    tool_router: ToolRouter<Flashback>,
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
        description = "Append a memory (a conversation turn, a note, an extracted fact). \
                       Default type is 'working' (session-scoped, 48h TTL). Use 'episodic' \
                       for cross-session history, 'semantic' for distilled facts, \
                       'procedural' for learned workflows."
    )]
    async fn flashback_remember(
        &self,
        Parameters(args): Parameters<RememberArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let body = to_json(&args)?;
        let res = self.post("/memory/ingest", &bearer, body).await?;
        result_ok(res)
    }

    #[tool(
        description = "Hybrid retrieval over the user's memories: semantic similarity + BM25 \
                       keyword + recency + project match + entity overlap. 'answer' mode \
                       optimizes for relevance to a query; 'manager' mode for situational \
                       awareness when no specific query exists."
    )]
    async fn flashback_search(
        &self,
        Parameters(args): Parameters<SearchArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let body = to_json(&args)?;
        let res = self.post("/memory/search", &bearer, body).await?;
        result_ok(res)
    }

    #[tool(
        description = "Return a structured 5-layer prompt context for the current session: \
                       procedural / active project (core memory + state objects) / retrieved \
                       memories / document chunks / recent conversation. Call this BEFORE \
                       sending a turn to the LLM."
    )]
    async fn flashback_assemble_context(
        &self,
        Parameters(args): Parameters<AssembleArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let body = to_json(&args)?;
        let res = self.post("/context/assemble", &bearer, body).await?;
        result_ok(res)
    }

    #[tool(
        description = "Mark a memory as superseded by a new version. The old row stays in the \
                       supersede chain for /lineage queries; default retrieval returns only \
                       the new terminal node."
    )]
    async fn flashback_supersede(
        &self,
        Parameters(args): Parameters<SupersedeArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/memory/{}/supersede", args.memory_id);
        let body = json!({
            "content": args.content,
            "importance": args.importance,
        });
        let res = self.put(&path, &bearer, body).await?;
        result_ok(res)
    }

    #[tool(description = "Walk the supersede chain (back + forward) for a memory id.")]
    async fn flashback_lineage(
        &self,
        Parameters(args): Parameters<LineageArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/lineage/{}", args.memory_id);
        let res = self.get(&path, &bearer).await?;
        result_ok(res)
    }

    #[tool(
        description = "Pin a piece of core memory — always-on context injected into every \
                       /assemble_context call. Use for behavioral rules and persistent \
                       preferences that should apply on every turn."
    )]
    async fn flashback_core_add(
        &self,
        Parameters(args): Parameters<CoreAddArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let body = to_json(&args)?;
        let res = self.post("/core", &bearer, body).await?;
        result_ok(res)
    }

    #[tool(description = "List all pinned core memory entries for the authenticated user.")]
    async fn flashback_core_list(
        &self,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let res = self.get("/core", &bearer).await?;
        result_ok(res)
    }

    #[tool(
        description = "Create a state object — a named mutable structure (a todo list, a plan). \
                       Each (kind, state_key) is unique per user. The terminal node is always \
                       the current value; supersede chain preserves history."
    )]
    async fn flashback_state_create(
        &self,
        Parameters(args): Parameters<StateCreateArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/state/{}", args.kind);
        let body = json!({
            "state_key":  args.state_key,
            "initial":    args.initial,
            "project_id": args.project_id,
            "session_id": args.session_id,
            "importance": args.importance,
        });
        let res = self.post(&path, &bearer, body).await?;
        result_ok(res)
    }

    #[tool(description = "Get the current (terminal) value of a state object.")]
    async fn flashback_state_get(
        &self,
        Parameters(args): Parameters<StateLookupArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/state/{}/{}", args.kind, args.state_key);
        let res = self.get(&path, &bearer).await?;
        result_ok(res)
    }

    #[tool(
        description = "Apply an op to a state object. For kind=\"todo_list\" the supported \
                       ops are: add(text), mark_done(item_id), unmark(item_id), \
                       toggle(item_id), remove(item_id), update(item_id, text), \
                       reorder(ids: full list of existing ids in new order), \
                       replace(items: full new item list), clear."
    )]
    async fn flashback_state_patch(
        &self,
        Parameters(args): Parameters<StatePatchArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/state/{}/{}", args.kind, args.state_key);
        let mut body = serde_json::Map::new();
        body.insert("op".into(), Value::String(args.op));
        if let Some(t) = args.text {
            body.insert("text".into(), Value::String(t));
        }
        if let Some(id) = args.item_id {
            body.insert("item_id".into(), Value::String(id));
        }
        if let Some(ids) = args.ids {
            body.insert("ids".into(), serde_json::to_value(ids).unwrap());
        }
        if let Some(items) = args.items {
            body.insert("items".into(), Value::Array(items));
        }
        if let Some(p) = args.project_id {
            body.insert("project_id".into(), Value::String(p));
        }
        if let Some(s) = args.session_id {
            body.insert("session_id".into(), Value::String(s));
        }
        let res = self.patch(&path, &bearer, Value::Object(body)).await?;
        result_ok(res)
    }

    #[tool(description = "Return the full supersede chain for a state object — how it evolved.")]
    async fn flashback_state_history(
        &self,
        Parameters(args): Parameters<StateLookupArgs>,
        ctx: RequestContext<RoleServer>,
    ) -> Result<CallToolResult, McpError> {
        let bearer = bearer_or_err(&ctx)?;
        let path = format!("/state/{}/{}/history", args.kind, args.state_key);
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

    async fn put(&self, path: &str, bearer: &str, body: Value) -> Result<Value, McpError> {
        let url = format!("{}{}", self.flashback_url, path);
        let resp = self
            .http
            .put(&url)
            .bearer_auth(bearer)
            .json(&body)
            .send()
            .await
            .map_err(|e| internal(format!("PUT {path}: {e}")))?;
        decode_resp(resp, path).await
    }

    async fn patch(&self, path: &str, bearer: &str, body: Value) -> Result<Value, McpError> {
        let url = format!("{}{}", self.flashback_url, path);
        let resp = self
            .http
            .patch(&url)
            .bearer_auth(bearer)
            .json(&body)
            .send()
            .await
            .map_err(|e| internal(format!("PATCH {path}: {e}")))?;
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
    serde_json::from_str(&text)
        .map_err(|e| internal(format!("decode response from {path}: {e}")))
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
