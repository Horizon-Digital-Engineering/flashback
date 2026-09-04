//! Remote LLM provider — HTTPS to a hosted model.
//!
//! Supports three backend shapes (and via OpenRouter, ~all hosted LLMs):
//!
//! * `RemoteBackend::OpenRouter` — recommended default. OpenAI-compatible
//!   chat completions; the `model` field picks the upstream provider
//!   (e.g. `anthropic/claude-haiku-4-5`, `openai/gpt-5-mini`).
//! * `RemoteBackend::Anthropic`  — direct Messages API. Supports the prompt
//!   cache header for ~90% token discount on the system prompt.
//! * `RemoteBackend::OpenAI`     — direct chat completions API.

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::prompt::{
    build_distill_system_prompt, build_distill_user_prompt, build_system_prompt, build_user_prompt,
};
use super::{
    AiProvider, Capabilities, DistillResponse, DistilledFact, EpisodeRef, ExtractContext,
    Extraction, ProviderError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteBackend {
    OpenRouter,
    Anthropic,
    OpenAI,
}

#[derive(Debug, Clone)]
pub struct RemoteLlmConfig {
    pub backend: RemoteBackend,
    pub api_key: String,
    pub api_base: Option<String>,
    pub prompt_cache: bool,

    // Per-role model tiering — see docs/MODEL-TIERING.md. extract() runs on
    // the write path (sub-2s budget, structured classification); distill_facts()
    // runs in the consolidation worker (multi-minute budget, real reasoning).
    // Single-model setups set both to the same string.
    pub extract_model: String,
    pub extract_max_tokens: u32,
    pub extract_timeout_ms: u32,
    pub distill_model: String,
    pub distill_max_tokens: u32,
    pub distill_timeout_ms: u32,
}

impl Default for RemoteLlmConfig {
    fn default() -> Self {
        let default_model = "anthropic/claude-haiku-4-5".to_string();
        Self {
            backend: RemoteBackend::OpenRouter,
            api_key: String::new(),
            api_base: None,
            prompt_cache: true,
            extract_model: default_model.clone(),
            extract_max_tokens: 512,
            extract_timeout_ms: 5000,
            distill_model: default_model,
            distill_max_tokens: 1024,
            distill_timeout_ms: 30000,
        }
    }
}

pub struct RemoteLlmProvider {
    config: RemoteLlmConfig,
    http: reqwest::Client,
}

impl RemoteLlmProvider {
    pub fn new(config: RemoteLlmConfig) -> Result<Self, ProviderError> {
        // A key is only mandatory when the endpoint is a hosted default. With
        // an explicit api_base the endpoint is self-chosen — a local Ollama or
        // vLLM needs no key, and refusing one here silently downgraded every
        // such deploy to the heuristic provider at startup.
        if config.api_key.is_empty() && config.api_base.is_none() {
            return Err(ProviderError::NotConfigured(
                "RemoteLlmProvider requires an API key for hosted backends — set \
                 PROVIDER_REMOTE_API_KEY (or OPENROUTER_API_KEY / ANTHROPIC_API_KEY / \
                 OPENAI_API_KEY), or point PROVIDER_REMOTE_API_BASE at a self-hosted \
                 endpoint that needs none"
                    .into(),
            ));
        }
        // Builder timeout is a CEILING. Per-call .timeout() overrides per role
        // so extraction's tight budget doesn't bound distillation's long one
        // and vice versa.
        let ceiling = config.extract_timeout_ms.max(config.distill_timeout_ms);
        let http = reqwest::Client::builder()
            .timeout(Duration::from_millis(ceiling as u64))
            .build()
            .map_err(|e| ProviderError::Internal(format!("http client build: {e}")))?;
        Ok(Self { config, http })
    }

    fn base_url(&self) -> &str {
        self.config
            .api_base
            .as_deref()
            .unwrap_or(match self.config.backend {
                RemoteBackend::OpenRouter => "https://openrouter.ai/api/v1",
                RemoteBackend::Anthropic => "https://api.anthropic.com",
                RemoteBackend::OpenAI => "https://api.openai.com/v1",
            })
    }
}

#[async_trait]
impl AiProvider for RemoteLlmProvider {
    fn name(&self) -> &'static str {
        match self.config.backend {
            RemoteBackend::OpenRouter => "remote-openrouter",
            RemoteBackend::Anthropic => "remote-anthropic",
            RemoteBackend::OpenAI => "remote-openai",
        }
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            extraction: true,
            summarization: true,
            fact_distillation: true,
            typical_latency_ms: 350,
            context_window: 200_000,
        }
    }

    async fn extract(&self, text: &str, ctx: &ExtractContext) -> Result<Extraction, ProviderError> {
        let system = build_system_prompt();
        let user = build_user_prompt(text, &ctx.recent_context);

        let model = &self.config.extract_model;
        let max_tokens = self.config.extract_max_tokens;
        let timeout_ms = self.config.extract_timeout_ms;

        let raw_json = match self.config.backend {
            RemoteBackend::OpenRouter | RemoteBackend::OpenAI => {
                self.call_openai_compatible(system, &user, model, max_tokens, timeout_ms)
                    .await?
            }
            RemoteBackend::Anthropic => {
                self.call_anthropic(system, &user, model, max_tokens, timeout_ms)
                    .await?
            }
        };

        parse_extraction(&raw_json)
    }

    async fn distill_facts(
        &self,
        episodes: &[EpisodeRef],
    ) -> Result<Vec<DistilledFact>, ProviderError> {
        if episodes.is_empty() {
            return Ok(Vec::new());
        }
        let system = build_distill_system_prompt();
        let json_payload = serde_json::to_string(&episodes)
            .map_err(|e| ProviderError::Internal(format!("serialize episodes: {e}")))?;
        let user = build_distill_user_prompt(&json_payload);

        let model = &self.config.distill_model;
        let max_tokens = self.config.distill_max_tokens;
        let timeout_ms = self.config.distill_timeout_ms;

        let raw_json = match self.config.backend {
            RemoteBackend::OpenRouter | RemoteBackend::OpenAI => {
                self.call_openai_compatible(system, &user, model, max_tokens, timeout_ms)
                    .await?
            }
            RemoteBackend::Anthropic => {
                self.call_anthropic(system, &user, model, max_tokens, timeout_ms)
                    .await?
            }
        };

        parse_distill(&raw_json)
    }
}

impl RemoteLlmProvider {
    async fn call_openai_compatible(
        &self,
        system: &str,
        user: &str,
        model: &str,
        max_tokens: u32,
        timeout_ms: u32,
    ) -> Result<String, ProviderError> {
        let body = json!({
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": system },
                { "role": "user",   "content": user },
            ],
        });

        let url = format!("{}/chat/completions", self.base_url());
        let mut req = self
            .http
            .post(&url)
            .timeout(Duration::from_millis(timeout_ms as u64))
            .json(&body);
        // Keyless is a real configuration (self-hosted endpoint); don't send
        // an empty Authorization header to servers that may reject it.
        if !self.config.api_key.is_empty() {
            req = req.bearer_auth(&self.config.api_key);
        }

        // OpenRouter accepts (and recommends) HTTP-Referer + X-Title for
        // attribution. Harmless on OpenAI direct.
        req = req
            .header(
                "HTTP-Referer",
                "https://github.com/Horizon-Digital-Engineering/flashback",
            )
            .header("X-Title", "flashback");

        let resp = req
            .send()
            .await
            .map_err(|e| map_http_error(e, timeout_ms))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Upstream(format!(
                "{} returned {status}: {text}",
                self.name()
            )));
        }
        let parsed: OpenAIResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::BadOutput(format!("OpenAI response decode: {e}")))?;
        parsed
            .choices
            .into_iter()
            .next()
            .map(|c| c.message.content)
            .ok_or_else(|| ProviderError::BadOutput("no choices in OpenAI response".into()))
    }

    async fn call_anthropic(
        &self,
        system: &str,
        user: &str,
        model: &str,
        max_tokens: u32,
        timeout_ms: u32,
    ) -> Result<String, ProviderError> {
        let system_blocks = if self.config.prompt_cache {
            // System prompt is cacheable — discount across calls.
            json!([
                {
                    "type": "text",
                    "text": system,
                    "cache_control": { "type": "ephemeral" }
                }
            ])
        } else {
            json!([{ "type": "text", "text": system }])
        };
        let body = json!({
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "system": system_blocks,
            "messages": [
                { "role": "user", "content": user }
            ],
        });

        let url = format!("{}/v1/messages", self.base_url());
        let req = self
            .http
            .post(&url)
            .timeout(Duration::from_millis(timeout_ms as u64))
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body);

        let resp = req
            .send()
            .await
            .map_err(|e| map_http_error(e, timeout_ms))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Upstream(format!(
                "anthropic returned {status}: {text}"
            )));
        }
        let parsed: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| ProviderError::BadOutput(format!("anthropic response decode: {e}")))?;
        parsed
            .content
            .into_iter()
            .filter_map(|b| {
                if b.r#type == "text" {
                    Some(b.text)
                } else {
                    None
                }
            })
            .next()
            .ok_or_else(|| ProviderError::BadOutput("no text block in anthropic response".into()))
    }
}

fn map_http_error(e: reqwest::Error, timeout_ms: u32) -> ProviderError {
    if e.is_timeout() {
        ProviderError::Timeout(timeout_ms)
    } else {
        ProviderError::Upstream(e.to_string())
    }
}

/// Parse the model's JSON output into an Extraction. Tolerates a small set
/// of common LLM quirks: code fences, leading "json", whitespace, etc.
pub(crate) fn parse_distill(raw: &str) -> Result<Vec<DistilledFact>, ProviderError> {
    let cleaned = strip_fences(raw);
    let r: DistillResponse = serde_json::from_str(cleaned).map_err(|e| {
        ProviderError::BadOutput(format!(
            "could not parse DistillResponse JSON: {e}; raw: {}",
            cleaned.chars().take(300).collect::<String>()
        ))
    })?;
    Ok(r.facts)
}

fn strip_fences(raw: &str) -> &str {
    let trimmed = raw.trim();
    let no_fence = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|s| s.trim_end_matches("```").trim())
        .unwrap_or(trimmed);
    // Slice to the first { ... last } so prose around the JSON is tolerated.
    let s = no_fence.find('{');
    let e = no_fence.rfind('}');
    match (s, e) {
        (Some(s), Some(e)) if e > s => &no_fence[s..=e],
        _ => no_fence,
    }
}

pub(crate) fn parse_extraction(raw: &str) -> Result<Extraction, ProviderError> {
    let cleaned = raw
        .trim()
        .strip_prefix("```json")
        .or_else(|| raw.trim().strip_prefix("```"))
        .map(|s| s.trim_end_matches("```").trim())
        .unwrap_or(raw.trim());

    // Locate the first { ... } JSON object span and parse just that, in case
    // the model wrote prose around the JSON.
    let start = cleaned.find('{');
    let end = cleaned.rfind('}');
    let slice = match (start, end) {
        (Some(s), Some(e)) if e > s => &cleaned[s..=e],
        _ => cleaned,
    };

    serde_json::from_str::<Extraction>(slice).map_err(|e| {
        ProviderError::BadOutput(format!(
            "could not parse Extraction JSON: {e}; raw: {}",
            slice.chars().take(300).collect::<String>()
        ))
    })
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}
#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}
#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicBlock>,
}
#[derive(Debug, Deserialize)]
struct AnthropicBlock {
    r#type: String,
    #[serde(default)]
    text: String,
}

#[allow(dead_code)]
fn _serialize_for_test() -> Value {
    json!({})
}

#[allow(dead_code)]
fn _serialize_unused_to_silence_warnings() -> Result<(), serde_json::Error> {
    let _ = serde_json::to_value::<()>(()).is_ok();
    Ok(())
}

// Tiny offline tests for the parser — no network.

#[derive(Serialize)]
struct _DummySerialize;
#[cfg(test)]
mod tests {
    use super::*;

    async fn stub(app: axum::Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let jh = tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        (format!("http://{addr}"), jh)
    }

    fn cfg_for(backend: RemoteBackend, base: &str, key: &str) -> RemoteLlmConfig {
        RemoteLlmConfig {
            backend,
            api_key: key.into(),
            api_base: Some(base.to_string()),
            prompt_cache: true,
            extract_model: "x-model".into(),
            extract_max_tokens: 256,
            extract_timeout_ms: 5000,
            distill_model: "d-model".into(),
            distill_max_tokens: 512,
            distill_timeout_ms: 5000,
        }
    }

    fn openai_reply(content: &str) -> serde_json::Value {
        json!({ "choices": [ { "message": { "content": content } } ] })
    }

    #[tokio::test]
    async fn an_openai_compatible_endpoint_is_asked_for_json_and_parsed() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Value::Null));
        let captured = seen.clone();
        let (base, jh) = stub(axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(
                move |headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
                    let captured = captured.clone();
                    async move {
                        *captured.lock().unwrap() = json!({
                            "body": body,
                            "auth": headers.get("authorization").is_some(),
                            "title": headers
                                .get("x-title")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("")
                                .to_string(),
                        });
                        axum::Json(openai_reply(
                            r#"{"topic":"backups","intent":"update","entities":["backup drive"]}"#,
                        ))
                    }
                },
            ),
        ))
        .await;

        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::OpenAI, &base, "a-key")).unwrap();
        let out = p
            .extract("swapped the backup drive", &ExtractContext::default())
            .await
            .unwrap();
        assert_eq!(out.topic.as_deref(), Some("backups"));

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen["body"]["model"], "x-model");
        assert_eq!(seen["body"]["temperature"], 0.0);
        assert_eq!(seen["body"]["response_format"]["type"], "json_object");
        assert_eq!(seen["auth"], true);
        assert_eq!(seen["title"], "flashback");
        jh.abort();
    }

    #[tokio::test]
    async fn a_keyless_endpoint_is_called_without_an_authorization_header() {
        // A self-hosted server is a real configuration, and some reject an
        // empty Authorization outright.
        let (base, jh) = stub(axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(|headers: axum::http::HeaderMap| async move {
                axum::Json(openai_reply(&format!(
                    r#"{{"topic":"auth-{}"}}"#,
                    headers.contains_key("authorization")
                )))
            }),
        ))
        .await;

        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::OpenAI, &base, "")).unwrap();
        let out = p.extract("x", &ExtractContext::default()).await.unwrap();
        assert_eq!(out.topic.as_deref(), Some("auth-false"));
        jh.abort();
    }

    #[tokio::test]
    async fn the_anthropic_shape_is_its_own_and_thinking_blocks_are_skipped() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Value::Null));
        let captured = seen.clone();
        let (base, jh) = stub(axum::Router::new().route(
            "/v1/messages",
            axum::routing::post(
                move |headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
                    let captured = captured.clone();
                    async move {
                        *captured.lock().unwrap() = json!({
                            "body": body,
                            "key_header": headers.contains_key("x-api-key"),
                            "bearer": headers.contains_key("authorization"),
                            "version": headers
                                .get("anthropic-version")
                                .and_then(|v| v.to_str().ok())
                                .unwrap_or("")
                                .to_string(),
                        });
                        axum::Json(json!({
                            "content": [
                                { "type": "thinking", "text": "not the answer" },
                                { "type": "text", "text": r#"{"topic":"the answer"}"# },
                            ]
                        }))
                    }
                },
            ),
        ))
        .await;

        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::Anthropic, &base, "k")).unwrap();
        let out = p.extract("x", &ExtractContext::default()).await.unwrap();
        assert_eq!(
            out.topic.as_deref(),
            Some("the answer"),
            "a leading non-text block was read as the reply"
        );

        let seen = seen.lock().unwrap().clone();
        assert_eq!(seen["key_header"], true);
        assert_eq!(seen["bearer"], false, "the bearer scheme is the other API");
        assert_eq!(seen["version"], "2023-06-01");
        assert_eq!(
            seen["body"]["system"][0]["cache_control"]["type"],
            "ephemeral"
        );
        jh.abort();
    }

    #[tokio::test]
    async fn cache_control_is_left_off_when_the_config_says_so() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Value::Null));
        let captured = seen.clone();
        let (base, jh) = stub(axum::Router::new().route(
            "/v1/messages",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let captured = captured.clone();
                async move {
                    *captured.lock().unwrap() = body;
                    axum::Json(json!({ "content": [ { "type": "text", "text": "{}" } ] }))
                }
            }),
        ))
        .await;

        let mut cfg = cfg_for(RemoteBackend::Anthropic, &base, "k");
        cfg.prompt_cache = false;
        let p = RemoteLlmProvider::new(cfg).unwrap();
        p.extract("x", &ExtractContext::default()).await.unwrap();
        assert!(seen.lock().unwrap()["system"][0]["cache_control"].is_null());
        jh.abort();
    }

    #[tokio::test]
    async fn an_upstream_refusal_keeps_its_status_and_body() {
        let (base, jh) = stub(axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(|| async {
                (
                    axum::http::StatusCode::PAYMENT_REQUIRED,
                    "insufficient credits",
                )
            }),
        ))
        .await;

        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::OpenRouter, &base, "k")).unwrap();
        let err = p
            .extract("x", &ExtractContext::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Upstream(_)));
        let msg = err.to_string();
        assert!(msg.contains("402"), "{msg}");
        assert!(msg.contains("insufficient credits"), "{msg}");
        jh.abort();
    }

    #[tokio::test]
    async fn a_reply_with_no_choices_is_bad_output_not_an_empty_extraction() {
        let (base, jh) = stub(axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(|| async { axum::Json(json!({ "choices": [] })) }),
        ))
        .await;

        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::OpenAI, &base, "k")).unwrap();
        let err = p
            .extract("x", &ExtractContext::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::BadOutput(_)), "{err}");
        jh.abort();
    }

    #[tokio::test]
    async fn a_model_that_answers_in_prose_is_bad_output() {
        let (base, jh) = stub(axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(|| async {
                axum::Json(openai_reply("I'm afraid I can't help with that."))
            }),
        ))
        .await;

        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::OpenAI, &base, "k")).unwrap();
        assert!(matches!(
            p.extract("x", &ExtractContext::default()).await,
            Err(ProviderError::BadOutput(_))
        ));
        jh.abort();
    }

    #[tokio::test]
    async fn distillation_of_nothing_never_leaves_the_process() {
        // No endpoint is running; reaching one would hang until the timeout.
        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::OpenAI, "http://127.0.0.1:1", "k"))
            .unwrap();
        assert!(p.distill_facts(&[]).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn distilled_facts_come_back_through_the_fences_a_model_adds() {
        let (base, jh) = stub(axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(|| async {
                axum::Json(openai_reply(
                    "```json\n{\"facts\":[{\"content\":\"prefers coffee at 93C\",\"confidence\":0.9}]}\n```",
                ))
            }),
        ))
        .await;

        let p = RemoteLlmProvider::new(cfg_for(RemoteBackend::OpenAI, &base, "k")).unwrap();
        let facts = p
            .distill_facts(&[EpisodeRef {
                id: uuid::Uuid::nil(),
                content: "coffee".into(),
                topic: None,
                entities: Vec::new(),
                when: None,
            }])
            .await
            .unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].content, "prefers coffee at 93C");
        jh.abort();
    }

    #[tokio::test]
    async fn a_timeout_is_reported_as_a_timeout_not_an_upstream_error() {
        let (base, jh) = stub(axum::Router::new().route(
            "/chat/completions",
            axum::routing::post(|| async {
                tokio::time::sleep(Duration::from_secs(5)).await;
                axum::Json(openai_reply("{}"))
            }),
        ))
        .await;

        let mut cfg = cfg_for(RemoteBackend::OpenAI, &base, "k");
        cfg.extract_timeout_ms = 150;
        let p = RemoteLlmProvider::new(cfg).unwrap();
        let err = p
            .extract("x", &ExtractContext::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::Timeout(150)), "{err}");
        jh.abort();
    }

    #[test]
    fn the_base_url_falls_back_to_the_backend_default() {
        for (backend, expected) in [
            (RemoteBackend::OpenRouter, "https://openrouter.ai/api/v1"),
            (RemoteBackend::Anthropic, "https://api.anthropic.com"),
            (RemoteBackend::OpenAI, "https://api.openai.com/v1"),
        ] {
            let mut cfg = cfg_for(backend, "unused", "k");
            cfg.api_base = None;
            let p = RemoteLlmProvider::new(cfg).unwrap();
            assert_eq!(p.base_url(), expected);
        }
    }

    #[test]
    fn parses_clean_json() {
        let raw = r#"{"topic":"deploy target","intent":"update","entities":["deploy target"]}"#;
        let e = parse_extraction(raw).unwrap();
        assert_eq!(e.topic.as_deref(), Some("deploy target"));
    }

    #[test]
    fn parses_with_code_fences() {
        let raw = "```json\n{\"topic\":\"x\",\"intent\":\"unknown\"}\n```";
        let e = parse_extraction(raw).unwrap();
        assert_eq!(e.topic.as_deref(), Some("x"));
    }

    #[test]
    fn parses_with_surrounding_prose() {
        let raw = "Here is the extraction:\n{\"topic\":\"x\",\"intent\":\"unknown\"}\nDone.";
        let e = parse_extraction(raw).unwrap();
        assert_eq!(e.topic.as_deref(), Some("x"));
    }

    #[test]
    fn keyless_with_explicit_base_constructs() {
        let cfg = RemoteLlmConfig {
            api_base: Some("http://127.0.0.1:11434/v1".into()),
            ..Default::default()
        };
        assert!(cfg.api_key.is_empty());
        assert!(RemoteLlmProvider::new(cfg).is_ok());
    }

    #[test]
    fn keyless_against_a_hosted_default_is_refused() {
        let err = RemoteLlmProvider::new(RemoteLlmConfig::default()).err();
        assert!(matches!(err, Some(ProviderError::NotConfigured(_))));
    }
}
