use std::path::PathBuf;

use anyhow::{Context, Result};

#[derive(Debug, Clone)]
pub struct Config {
    pub database_url: String,
    pub host: String,
    pub port: u16,
    pub auto_migrate: bool,
    /// Optional cache directory for fastembed's downloaded ONNX models.
    /// In the Docker image we pin this to `/opt/flashback/fastembed-cache`
    /// so the build-time prefetch baked into the image is reused at runtime.
    pub fastembed_cache_dir: Option<PathBuf>,
    pub provider: ProviderConfig,
    /// DEV MODE: when true, auth middleware is BYPASSED and every request
    /// gets a synthetic `user_id="dev"`. Never enable in production. Loud
    /// warnings are logged at startup and a banner is rendered on every
    /// admin page so you can't forget it's on.
    pub dev_mode: bool,
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub kind: ProviderKind,
    /// On provider failure: `fail` (default) or `heuristic` fallback.
    pub fallback: FallbackPolicy,
    pub remote: RemoteProviderConfig,
    pub embedded: EmbeddedProviderConfig,
}

#[derive(Debug, Clone)]
pub struct EmbeddedProviderConfig {
    /// HF repo (e.g. `Qwen/Qwen3-0.6B`) OR a local GGUF path.
    pub model: String,
    pub context_size: usize,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    /// In-process rule-based extraction. No models, no network.
    Heuristic,
    /// HTTP to any OpenAI-compatible / Anthropic / OpenRouter endpoint —
    /// whether cloud, sidecar Ollama, LAN AI box, anywhere reachable by URL.
    Remote,
    /// LLM running inside the Flashback process via mistral.rs. No HTTP, no
    /// sidecar. Feature-gated behind `embedded-llm`. Phase 2b — skeleton only.
    Embedded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackPolicy {
    Fail,
    Heuristic,
}

#[derive(Debug, Clone)]
pub struct RemoteProviderConfig {
    pub backend: String, // "openrouter" | "anthropic" | "openai"
    pub api_key: String,
    pub api_base: Option<String>,
    pub prompt_cache: bool,
    // See docs/MODEL-TIERING.md. extract/distill can use different models,
    // max_tokens, and timeouts. Each falls back to the corresponding non-
    // role-specific value when unset, so a single-model deploy still works.
    pub extract_model: String,
    pub extract_max_tokens: u32,
    pub extract_timeout_ms: u32,
    pub distill_model: String,
    pub distill_max_tokens: u32,
    pub distill_timeout_ms: u32,
}

/// `--dev` anywhere in argv OR `FLASHBACK_DEV_MODE=1` env turns dev mode on.
fn dev_mode_from_env_or_args() -> bool {
    let env_on = matches!(
        std::env::var("FLASHBACK_DEV_MODE").as_deref(),
        Ok("1" | "true" | "TRUE" | "yes")
    );
    let arg_on = std::env::args().any(|a| a == "--dev");
    env_on || arg_on
}

impl ProviderConfig {
    pub fn from_env() -> Self {
        let kind = match std::env::var("PROVIDER").as_deref() {
            Ok("embedded") => ProviderKind::Embedded,
            Ok("remote") => ProviderKind::Remote,
            _ => ProviderKind::Heuristic,
        };
        let fallback = match std::env::var("PROVIDER_FALLBACK").as_deref() {
            Ok("heuristic") => FallbackPolicy::Heuristic,
            _ => FallbackPolicy::Fail,
        };

        let backend =
            std::env::var("PROVIDER_REMOTE_PROVIDER").unwrap_or_else(|_| "openrouter".to_string());
        let model = std::env::var("PROVIDER_REMOTE_MODEL").unwrap_or_else(|_| {
            match backend.as_str() {
                "anthropic" => "claude-haiku-4-5".to_string(),
                "openai" => "gpt-5-mini".to_string(),
                _ => "anthropic/claude-haiku-4-5".to_string(), // openrouter default
            }
        });
        // Accept either the generic key OR provider-specific aliases.
        let api_key = std::env::var("PROVIDER_REMOTE_API_KEY")
            .or_else(|_| match backend.as_str() {
                "anthropic" => std::env::var("ANTHROPIC_API_KEY"),
                "openai" => std::env::var("OPENAI_API_KEY"),
                _ => std::env::var("OPENROUTER_API_KEY"),
            })
            .unwrap_or_default();
        let api_base = std::env::var("PROVIDER_REMOTE_API_BASE").ok();
        let timeout_ms = std::env::var("PROVIDER_REMOTE_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5000);
        let prompt_cache = !matches!(
            std::env::var("PROVIDER_REMOTE_PROMPT_CACHE").as_deref(),
            Ok("0" | "false" | "no")
        );

        // Per-role overrides (extract vs distill). When unset, fall back to
        // the non-role-specific values above so existing single-model
        // configs keep working unchanged.
        let extract_model =
            std::env::var("PROVIDER_REMOTE_EXTRACT_MODEL").unwrap_or_else(|_| model.clone());
        let distill_model =
            std::env::var("PROVIDER_REMOTE_DISTILL_MODEL").unwrap_or_else(|_| model.clone());
        let extract_max_tokens = std::env::var("PROVIDER_REMOTE_EXTRACT_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let distill_max_tokens = std::env::var("PROVIDER_REMOTE_DISTILL_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1024);
        let extract_timeout_ms = std::env::var("PROVIDER_REMOTE_EXTRACT_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(timeout_ms);
        let distill_timeout_ms = std::env::var("PROVIDER_REMOTE_DISTILL_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(timeout_ms.max(30000));

        let embedded_model = std::env::var("PROVIDER_EMBEDDED_MODEL")
            .unwrap_or_else(|_| "Qwen/Qwen3-0.6B".to_string());
        let embedded_ctx = std::env::var("PROVIDER_EMBEDDED_CONTEXT_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4096);
        let embedded_max = std::env::var("PROVIDER_EMBEDDED_MAX_TOKENS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);

        Self {
            kind,
            fallback,
            remote: RemoteProviderConfig {
                backend,
                api_key,
                api_base,
                prompt_cache,
                extract_model,
                extract_max_tokens,
                extract_timeout_ms,
                distill_model,
                distill_max_tokens,
                distill_timeout_ms,
            },
            embedded: EmbeddedProviderConfig {
                model: embedded_model,
                context_size: embedded_ctx,
                max_tokens: embedded_max,
            },
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Config {
            database_url: std::env::var("DATABASE_URL").context("DATABASE_URL must be set")?,
            host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .context("PORT must be a valid number")?,
            auto_migrate: matches!(
                std::env::var("AUTO_MIGRATE").as_deref(),
                Ok("1" | "true" | "TRUE" | "yes")
            ),
            fastembed_cache_dir: std::env::var_os("FLASHBACK_FASTEMBED_CACHE").map(PathBuf::from),
            provider: ProviderConfig::from_env(),
            dev_mode: dev_mode_from_env_or_args(),
        })
    }

    pub fn listen_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Redact the password component for logging.
    pub fn database_url_safe(&self) -> String {
        let url = &self.database_url;
        if let Some(start) = url.find("://") {
            let rest = &url[start + 3..];
            if let Some(at) = rest.find('@') {
                if let Some(colon) = rest[..at].find(':') {
                    return format!("{}://{}:***{}", &url[..start], &rest[..colon], &rest[at..]);
                }
            }
        }
        url.clone()
    }
}
