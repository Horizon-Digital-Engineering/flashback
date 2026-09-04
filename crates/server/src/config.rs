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
    /// The settings page rejects a provider or backend name it does not know;
    /// the environment used to accept anything and quietly resolve it to the
    /// default. A one-character typo in `PROVIDER_REMOTE_PROVIDER` therefore
    /// sent `PROVIDER_REMOTE_API_KEY` to openrouter.ai, and one in `PROVIDER`
    /// downgraded the whole pipeline to the heuristic, both silently. Both
    /// surfaces now refuse the same set of names.
    pub fn from_env() -> Result<Self> {
        let kind = match std::env::var("PROVIDER").as_deref() {
            Ok("embedded") => ProviderKind::Embedded,
            Ok("remote") => ProviderKind::Remote,
            Ok("heuristic") | Ok("") | Err(_) => ProviderKind::Heuristic,
            Ok(other) => {
                anyhow::bail!("PROVIDER must be 'heuristic', 'remote' or 'embedded', got '{other}'")
            }
        };
        let fallback = match std::env::var("PROVIDER_FALLBACK").as_deref() {
            Ok("heuristic") => FallbackPolicy::Heuristic,
            Ok("fail") | Ok("") | Err(_) => FallbackPolicy::Fail,
            Ok(other) => {
                anyhow::bail!("PROVIDER_FALLBACK must be 'fail' or 'heuristic', got '{other}'")
            }
        };

        let backend =
            std::env::var("PROVIDER_REMOTE_PROVIDER").unwrap_or_else(|_| "openrouter".to_string());
        if !matches!(backend.as_str(), "openrouter" | "anthropic" | "openai") {
            anyhow::bail!(
                "PROVIDER_REMOTE_PROVIDER must be 'openrouter', 'anthropic' or 'openai', \
                 got '{backend}' — an unrecognised name resolves to openrouter.ai, which \
                 is where the API key would then be sent"
            );
        }
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

        Ok(Self {
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
        })
    }
}

impl Config {
    pub fn from_env() -> Result<Self> {
        Ok(Config {
            database_url: std::env::var("DATABASE_URL").context("DATABASE_URL must be set")?,
            // Loopback unless asked otherwise. Containers must set HOST=0.0.0.0
            // explicitly — inside one, binding loopback is unreachable from the
            // host — and docker-compose does.
            host: std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .context("PORT must be a valid number")?,
            auto_migrate: matches!(
                std::env::var("AUTO_MIGRATE").as_deref(),
                Ok("1" | "true" | "TRUE" | "yes")
            ),
            fastembed_cache_dir: std::env::var_os("FLASHBACK_FASTEMBED_CACHE").map(PathBuf::from),
            provider: ProviderConfig::from_env()?,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env-var reads are process-global. Tests that mutate env need to be
    // serialized against each other; pure-method tests below don't need it.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn cfg(database_url: &str, host: &str, port: u16) -> Config {
        Config {
            database_url: database_url.to_string(),
            host: host.to_string(),
            port,
            auto_migrate: false,
            fastembed_cache_dir: None,
            provider: ProviderConfig {
                kind: ProviderKind::Heuristic,
                fallback: FallbackPolicy::Fail,
                remote: RemoteProviderConfig {
                    backend: "openrouter".into(),
                    api_key: String::new(),
                    api_base: None,
                    prompt_cache: true,
                    extract_model: "m".into(),
                    extract_max_tokens: 0,
                    extract_timeout_ms: 0,
                    distill_model: "m".into(),
                    distill_max_tokens: 0,
                    distill_timeout_ms: 0,
                },
                embedded: EmbeddedProviderConfig {
                    model: "m".into(),
                    context_size: 0,
                    max_tokens: 0,
                },
            },
            dev_mode: false,
        }
    }

    // ---- listen_addr ------------------------------------------------------

    #[test]
    fn the_bind_address_defaults_to_loopback() {
        // Every deployment that never sets HOST inherits this. Defaulting to
        // 0.0.0.0 meant the ones least likely to notice were the ones exposed.
        let prev = std::env::var("HOST").ok();
        std::env::remove_var("HOST");
        let host = std::env::var("HOST").unwrap_or_else(|_| "127.0.0.1".to_string());
        assert_eq!(host, "127.0.0.1");
        if let Some(v) = prev {
            std::env::set_var("HOST", v);
        }
    }

    #[test]
    fn listen_addr_joins_host_and_port() {
        let c = cfg("postgres://x@y/z", "127.0.0.1", 8080);
        assert_eq!(c.listen_addr(), "127.0.0.1:8080");
    }

    #[test]
    fn listen_addr_handles_ipv4_anyhost_and_named_host() {
        assert_eq!(cfg("p", "0.0.0.0", 80).listen_addr(), "0.0.0.0:80");
        assert_eq!(cfg("p", "::1", 9090).listen_addr(), "::1:9090");
    }

    // ---- database_url_safe ------------------------------------------------

    #[test]
    fn database_url_safe_redacts_password() {
        // The literal between `:` and `@` is the fixture being redacted;
        // intentionally NOT a real-looking secret so Sonar's secrets:S6698
        // scanner doesn't flag this test file.
        let c = cfg(
            "postgres://flashback:PLACEHOLDER_REDACT_ME@localhost:5432/db",
            "h",
            1,
        );
        assert_eq!(
            c.database_url_safe(),
            "postgres://flashback:***@localhost:5432/db"
        );
    }

    #[test]
    fn database_url_safe_keeps_url_without_password() {
        // No `user:pass@` segment → URL passes through unchanged.
        let c = cfg("postgres://localhost:5432/db", "h", 1);
        assert_eq!(c.database_url_safe(), "postgres://localhost:5432/db");
    }

    #[test]
    fn database_url_safe_keeps_url_with_only_username() {
        // `user@host` (no password) is left alone — the redaction only
        // triggers on `user:pass@`.
        let c = cfg("postgres://alice@localhost/db", "h", 1);
        assert_eq!(c.database_url_safe(), "postgres://alice@localhost/db");
    }

    #[test]
    fn database_url_safe_passes_through_garbage() {
        let c = cfg("not even a url", "h", 1);
        assert_eq!(c.database_url_safe(), "not even a url");
    }

    // ---- dev_mode_from_env_or_args ----------------------------------------

    fn clear_dev_env() {
        // SAFETY: tests hold ENV_LOCK while mutating these.
        unsafe { std::env::remove_var("FLASHBACK_DEV_MODE") };
    }
    fn set_dev_env(value: &str) {
        unsafe { std::env::set_var("FLASHBACK_DEV_MODE", value) };
    }

    #[test]
    fn dev_mode_unset_returns_false() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_dev_env();
        assert!(!dev_mode_from_env_or_args());
    }

    #[test]
    fn dev_mode_truthy_env_returns_true() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for v in ["1", "true", "TRUE", "yes"] {
            set_dev_env(v);
            assert!(
                dev_mode_from_env_or_args(),
                "value {v} should turn dev mode on"
            );
        }
        clear_dev_env();
    }

    #[test]
    fn dev_mode_falsy_env_returns_false() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for v in ["0", "false", "no", "anything-else"] {
            set_dev_env(v);
            assert!(
                !dev_mode_from_env_or_args(),
                "value {v} should leave dev mode off"
            );
        }
        clear_dev_env();
    }

    // ---- ProviderConfig::from_env -----------------------------------------

    /// Snapshot of every env var ProviderConfig reads, so a test can swap
    /// the environment and restore it.
    fn snapshot_provider_env() -> Vec<(&'static str, Option<String>)> {
        let keys = [
            "PROVIDER",
            "PROVIDER_FALLBACK",
            "PROVIDER_REMOTE_PROVIDER",
            "PROVIDER_REMOTE_MODEL",
            "PROVIDER_REMOTE_API_KEY",
            "PROVIDER_REMOTE_API_BASE",
            "PROVIDER_REMOTE_TIMEOUT_MS",
            "PROVIDER_REMOTE_PROMPT_CACHE",
            "PROVIDER_REMOTE_EXTRACT_MODEL",
            "PROVIDER_REMOTE_DISTILL_MODEL",
            "PROVIDER_REMOTE_EXTRACT_MAX_TOKENS",
            "PROVIDER_REMOTE_DISTILL_MAX_TOKENS",
            "PROVIDER_REMOTE_EXTRACT_TIMEOUT_MS",
            "PROVIDER_REMOTE_DISTILL_TIMEOUT_MS",
            "PROVIDER_EMBEDDED_MODEL",
            "PROVIDER_EMBEDDED_CONTEXT_SIZE",
            "PROVIDER_EMBEDDED_MAX_TOKENS",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ];
        let snap: Vec<_> = keys.iter().map(|k| (*k, std::env::var(k).ok())).collect();
        for k in &keys {
            unsafe { std::env::remove_var(k) };
        }
        snap
    }

    fn restore_env(snap: Vec<(&'static str, Option<String>)>) {
        for (k, v) in snap {
            match v {
                Some(val) => unsafe { std::env::set_var(k, val) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
    }

    #[test]
    fn provider_from_env_defaults_to_heuristic_fail() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();
        let p = ProviderConfig::from_env().unwrap();
        assert_eq!(p.kind, ProviderKind::Heuristic);
        assert_eq!(p.fallback, FallbackPolicy::Fail);
        assert_eq!(p.remote.backend, "openrouter");
        assert_eq!(p.remote.api_base, None);
        assert!(p.remote.prompt_cache); // on by default
        restore_env(snap);
    }

    #[test]
    fn provider_from_env_recognizes_remote_and_embedded_kinds() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();

        unsafe { std::env::set_var("PROVIDER", "remote") };
        assert_eq!(
            ProviderConfig::from_env().unwrap().kind,
            ProviderKind::Remote
        );

        unsafe { std::env::set_var("PROVIDER", "embedded") };
        assert_eq!(
            ProviderConfig::from_env().unwrap().kind,
            ProviderKind::Embedded
        );

        unsafe { std::env::set_var("PROVIDER", "heuristic") };
        assert_eq!(
            ProviderConfig::from_env().unwrap().kind,
            ProviderKind::Heuristic
        );

        restore_env(snap);
    }

    #[test]
    fn an_unknown_provider_name_is_refused_rather_than_downgraded() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();

        unsafe { std::env::set_var("PROVIDER", "remot") };
        let err = ProviderConfig::from_env().unwrap_err().to_string();
        assert!(err.contains("PROVIDER"), "{err}");
        assert!(
            err.contains("remot"),
            "the message has to name the typo: {err}"
        );

        unsafe { std::env::set_var("PROVIDER", "heuristic") };
        unsafe { std::env::set_var("PROVIDER_FALLBACK", "heuristik") };
        assert!(ProviderConfig::from_env().is_err());

        restore_env(snap);
    }

    #[test]
    fn an_unknown_backend_never_silently_becomes_openrouter() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();

        unsafe {
            std::env::set_var("PROVIDER", "remote");
            std::env::set_var("PROVIDER_REMOTE_PROVIDER", "anthropi");
            std::env::set_var("PROVIDER_REMOTE_API_KEY", "the-operators-key");
        }
        let err = ProviderConfig::from_env().unwrap_err().to_string();
        assert!(err.contains("anthropi"), "{err}");
        assert!(
            !err.contains("the-operators-key"),
            "the key must not appear in an error that gets logged"
        );

        for good in ["openrouter", "anthropic", "openai"] {
            unsafe { std::env::set_var("PROVIDER_REMOTE_PROVIDER", good) };
            assert_eq!(ProviderConfig::from_env().unwrap().remote.backend, good);
        }

        restore_env(snap);
    }

    #[test]
    fn provider_from_env_picks_up_fallback_heuristic() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();
        unsafe { std::env::set_var("PROVIDER_FALLBACK", "heuristic") };
        assert_eq!(
            ProviderConfig::from_env().unwrap().fallback,
            FallbackPolicy::Heuristic
        );
        unsafe { std::env::set_var("PROVIDER_FALLBACK", "fail") };
        assert_eq!(
            ProviderConfig::from_env().unwrap().fallback,
            FallbackPolicy::Fail
        );
        restore_env(snap);
    }

    #[test]
    fn provider_from_env_resolves_backend_specific_api_keys() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();

        unsafe { std::env::set_var("PROVIDER_REMOTE_PROVIDER", "anthropic") };
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "ak-anthropic") };
        assert_eq!(
            ProviderConfig::from_env().unwrap().remote.api_key,
            "ak-anthropic"
        );

        // Generic key wins over backend-specific.
        unsafe { std::env::set_var("PROVIDER_REMOTE_API_KEY", "generic-wins") };
        assert_eq!(
            ProviderConfig::from_env().unwrap().remote.api_key,
            "generic-wins"
        );

        restore_env(snap);
    }

    #[test]
    fn provider_from_env_per_role_model_falls_back_to_default_model() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();

        unsafe { std::env::set_var("PROVIDER_REMOTE_MODEL", "default-model") };
        let p = ProviderConfig::from_env().unwrap();
        assert_eq!(p.remote.extract_model, "default-model");
        assert_eq!(p.remote.distill_model, "default-model");

        // Override only extract; distill still falls back to default.
        unsafe { std::env::set_var("PROVIDER_REMOTE_EXTRACT_MODEL", "fast-model") };
        let p = ProviderConfig::from_env().unwrap();
        assert_eq!(p.remote.extract_model, "fast-model");
        assert_eq!(p.remote.distill_model, "default-model");

        restore_env(snap);
    }

    #[test]
    fn provider_from_env_prompt_cache_off_when_explicitly_falsy() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let snap = snapshot_provider_env();

        for v in ["0", "false", "no"] {
            unsafe { std::env::set_var("PROVIDER_REMOTE_PROMPT_CACHE", v) };
            assert!(
                !ProviderConfig::from_env().unwrap().remote.prompt_cache,
                "value {v} should disable prompt cache"
            );
        }
        // Any other value (or unset) → cache on.
        unsafe { std::env::set_var("PROVIDER_REMOTE_PROMPT_CACHE", "on") };
        assert!(ProviderConfig::from_env().unwrap().remote.prompt_cache);

        restore_env(snap);
    }
}
