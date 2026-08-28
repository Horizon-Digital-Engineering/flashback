//! System provider settings — the database-backed layer over the environment.
//!
//! The environment is the bootstrap seed: it is what a fresh install runs on
//! before anyone has opened the admin UI. The `system_settings` row is the
//! runtime override: every non-NULL column wins over the corresponding
//! environment value, and saving from the settings page applies immediately
//! (the provider is rebuilt in place — no restart).
//!
//! The API key never lives in the database — a plaintext credential would land
//! in every `pg_dump`. It stays in the environment; the resolve step carries
//! the env key through unchanged.

use serde::{Deserialize, Serialize};
use sqlx::PgPool;

use crate::config::{ProviderConfig, ProviderKind};
use crate::error::AppResult;

/// The stored override row. Every field is optional; `None` inherits the
/// environment value. Also the wire shape for the settings API.
#[derive(Debug, Clone, Default, Serialize, Deserialize, sqlx::FromRow)]
pub struct SystemSettings {
    pub provider: Option<String>,
    pub remote_backend: Option<String>,
    pub api_base: Option<String>,
    pub extract_model: Option<String>,
    pub distill_model: Option<String>,
    pub extract_timeout_ms: Option<i32>,
    pub distill_timeout_ms: Option<i32>,
}

impl SystemSettings {
    pub fn is_empty(&self) -> bool {
        self.provider.is_none()
            && self.remote_backend.is_none()
            && self.api_base.is_none()
            && self.extract_model.is_none()
            && self.distill_model.is_none()
            && self.extract_timeout_ms.is_none()
            && self.distill_timeout_ms.is_none()
    }

    /// Normalise and validate a submitted settings payload. Blank strings
    /// become `None` so "cleared" and "never set" are the same state — the
    /// same rule the playground settings learned the hard way.
    pub fn normalised(mut self) -> Result<Self, String> {
        let norm = |v: Option<String>| v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        self.provider = norm(self.provider);
        self.remote_backend = norm(self.remote_backend);
        self.api_base = norm(self.api_base);
        self.extract_model = norm(self.extract_model);
        self.distill_model = norm(self.distill_model);

        if let Some(p) = self.provider.as_deref() {
            if !matches!(p, "heuristic" | "remote") {
                return Err(format!(
                    "provider must be 'heuristic' or 'remote', got '{p}'"
                ));
            }
        }
        if let Some(b) = self.remote_backend.as_deref() {
            if !matches!(b, "openai" | "anthropic" | "openrouter") {
                return Err(format!(
                    "remote_backend must be 'openai', 'anthropic' or 'openrouter', got '{b}'"
                ));
            }
        }
        if let Some(u) = self.api_base.as_deref() {
            validate_base_url(u)?;
        }
        for (label, t) in [
            ("extract_timeout_ms", self.extract_timeout_ms),
            ("distill_timeout_ms", self.distill_timeout_ms),
        ] {
            if let Some(ms) = t {
                if !(1000..=600_000).contains(&ms) {
                    return Err(format!("{label} must be between 1000 and 600000, got {ms}"));
                }
            }
        }
        Ok(self)
    }
}

pub async fn load(pool: &PgPool) -> AppResult<SystemSettings> {
    let row = sqlx::query_as::<_, SystemSettings>(
        "SELECT provider, remote_backend, api_base, extract_model, distill_model, \
                extract_timeout_ms, distill_timeout_ms \
         FROM system_settings WHERE id",
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.unwrap_or_default())
}

pub async fn save(pool: &PgPool, s: &SystemSettings) -> AppResult<()> {
    sqlx::query(
        "INSERT INTO system_settings \
           (id, provider, remote_backend, api_base, extract_model, distill_model, \
            extract_timeout_ms, distill_timeout_ms) \
         VALUES (TRUE, $1, $2, $3, $4, $5, $6, $7) \
         ON CONFLICT (id) DO UPDATE SET \
           provider = EXCLUDED.provider, \
           remote_backend = EXCLUDED.remote_backend, \
           api_base = EXCLUDED.api_base, \
           extract_model = EXCLUDED.extract_model, \
           distill_model = EXCLUDED.distill_model, \
           extract_timeout_ms = EXCLUDED.extract_timeout_ms, \
           distill_timeout_ms = EXCLUDED.distill_timeout_ms, \
           updated_at = NOW()",
    )
    .bind(&s.provider)
    .bind(&s.remote_backend)
    .bind(&s.api_base)
    .bind(&s.extract_model)
    .bind(&s.distill_model)
    .bind(s.extract_timeout_ms)
    .bind(s.distill_timeout_ms)
    .execute(pool)
    .await?;
    Ok(())
}

/// Overlay the stored settings on the environment config. Non-NULL columns
/// win; everything else — including the API key, which is never stored —
/// passes through from the environment.
pub fn resolve(env: &ProviderConfig, db: &SystemSettings) -> ProviderConfig {
    let mut cfg = env.clone();
    if let Some(p) = db.provider.as_deref() {
        cfg.kind = match p {
            "remote" => ProviderKind::Remote,
            _ => ProviderKind::Heuristic,
        };
    }
    if let Some(b) = &db.remote_backend {
        cfg.remote.backend = b.clone();
    }
    if let Some(u) = &db.api_base {
        cfg.remote.api_base = Some(u.clone());
    }
    if let Some(m) = &db.extract_model {
        cfg.remote.extract_model = m.clone();
    }
    if let Some(m) = &db.distill_model {
        cfg.remote.distill_model = m.clone();
    }
    if let Some(t) = db.extract_timeout_ms {
        cfg.remote.extract_timeout_ms = t as u32;
    }
    if let Some(t) = db.distill_timeout_ms {
        cfg.remote.distill_timeout_ms = t as u32;
    }
    cfg
}

/// Load the stored settings and resolve them over the environment config.
/// Used at startup and by the doctor so both see the same effective config
/// the settings page produces.
pub async fn resolve_from_db(pool: &PgPool, env: &ProviderConfig) -> ProviderConfig {
    match load(pool).await {
        Ok(db) => {
            if !db.is_empty() {
                tracing::info!("provider settings loaded from database (overriding environment)");
            }
            resolve(env, &db)
        }
        Err(e) => {
            tracing::warn!("could not load system settings ({e}); using environment config");
            env.clone()
        }
    }
}

/// Reject an endpoint the server should never be talked into calling.
///
/// Deliberately narrow: the *intended* endpoint is a local model server, so
/// blanket-blocking private ranges would ban the only configuration anyone
/// actually uses. What it stops is the classic SSRF payload — cloud
/// instance-metadata services, which hand out credentials to anything that
/// asks from inside the host. The real containment is layered on top: only an
/// authenticated operator reaches the routes that take a URL, and the CSRF
/// guard keeps a page you merely visit from submitting one on your behalf.
pub fn validate_base_url(raw: &str) -> Result<(), String> {
    let url = raw.trim();
    let rest = match url.split_once("://") {
        Some(("http", rest)) | Some(("https", rest)) => rest,
        Some((scheme, _)) => {
            return Err(format!(
                "scheme '{scheme}' is not allowed (http/https only)"
            ))
        }
        None => return Err("must start with http:// or https://".into()),
    };
    let host = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or("")
        .rsplit('@') // strip any user:pass@
        .next()
        .unwrap_or("");
    let hostname = host
        .rsplit_once(':')
        .map_or(host, |(h, _)| h)
        .trim_matches(['[', ']']);
    if hostname.is_empty() {
        return Err("no host in URL".into());
    }
    // Link-local: 169.254.0.0/16 and fd00:ec2::254 — the metadata endpoints on
    // every major cloud. Nothing legitimate here ever lives there.
    if hostname.starts_with("169.254.") || hostname.eq_ignore_ascii_case("metadata.google.internal")
    {
        return Err(format!("{hostname} is a cloud metadata address"));
    }
    Ok(())
}

/// Ask an OpenAI-compatible endpoint what models it serves. This is what makes
/// the model field a dropdown instead of a text box: a model that is not
/// pulled can never be named. Works against Ollama, vLLM, LM Studio and the
/// hosted providers alike — `GET {base}/models` is common to all of them.
pub async fn list_models(base_url: &str, api_key: Option<&str>) -> Result<Vec<String>, String> {
    validate_base_url(base_url)?;
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let mut req = client.get(&url);
    if let Some(key) = api_key.filter(|k| !k.trim().is_empty()) {
        req = req.bearer_auth(key);
    }
    let res = req.send().await.map_err(|e| format!("unreachable: {e}"))?;
    if !res.status().is_success() {
        return Err(format!("HTTP {} from {url}", res.status()));
    }
    let body = res.text().await.map_err(|e| e.to_string())?;
    parse_models_json(&body)
}

/// Parse the `{"data":[{"id":...}]}` list shape. Split out from the HTTP call
/// so the parsing is unit-testable without a live endpoint.
pub fn parse_models_json(body: &str) -> Result<Vec<String>, String> {
    let v: serde_json::Value =
        serde_json::from_str(body).map_err(|e| format!("model list is not JSON: {e}"))?;
    let data = v["data"]
        .as_array()
        .ok_or_else(|| "model list has no 'data' array".to_string())?;
    let mut models: Vec<String> = data
        .iter()
        .filter_map(|m| m["id"].as_str())
        .map(str::to_string)
        .collect();
    models.sort();
    models.dedup();
    if models.is_empty() {
        return Err("the endpoint reports no models".into());
    }
    Ok(models)
}

/// Startup verification for a remote provider with an explicit base URL:
/// confirm the configured models actually exist at the endpoint. Logs, never
/// fails the boot — a model server that starts slower than this service
/// (systemd gives no ordering against Ollama) must not wedge the memory store.
pub async fn verify_models_at_startup(cfg: &ProviderConfig) {
    if cfg.kind != ProviderKind::Remote {
        return;
    }
    let Some(base) = cfg.remote.api_base.as_deref() else {
        return;
    };
    let key = (!cfg.remote.api_key.is_empty()).then_some(cfg.remote.api_key.as_str());
    match list_models(base, key).await {
        Ok(models) => {
            for (role, model) in [
                ("extract", &cfg.remote.extract_model),
                ("distill", &cfg.remote.distill_model),
            ] {
                if models.iter().any(|m| m == model) {
                    tracing::info!("{role} model '{model}' present at {base}");
                } else {
                    tracing::error!(
                        "{role} model '{model}' is NOT served by {base} — every {role} call \
                         will fail. Available: {}. Fix it on the admin settings page.",
                        models.join(", ")
                    );
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                "could not verify models at {base} ({e}) — if the model server is still \
                 starting this resolves itself; otherwise extraction will fail"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;

    fn env_cfg() -> ProviderConfig {
        let mut cfg = ProviderConfig::from_env();
        cfg.kind = ProviderKind::Heuristic;
        cfg.remote.backend = "openai".into();
        cfg.remote.api_base = Some("http://127.0.0.1:11434/v1".into());
        cfg.remote.api_key = "env-key".into();
        cfg.remote.extract_model = "env-model".into();
        cfg.remote.distill_model = "env-model".into();
        cfg.remote.extract_timeout_ms = 60000;
        cfg.remote.distill_timeout_ms = 180000;
        cfg
    }

    #[test]
    fn resolve_with_empty_settings_is_the_environment() {
        let env = env_cfg();
        let out = resolve(&env, &SystemSettings::default());
        assert_eq!(out.kind, env.kind);
        assert_eq!(out.remote.extract_model, "env-model");
        assert_eq!(out.remote.extract_timeout_ms, 60000);
    }

    #[test]
    fn resolve_stored_fields_win_and_key_passes_through() {
        let db = SystemSettings {
            provider: Some("remote".into()),
            extract_model: Some("db-model".into()),
            extract_timeout_ms: Some(30000),
            ..Default::default()
        };
        let out = resolve(&env_cfg(), &db);
        assert_eq!(out.kind, ProviderKind::Remote);
        assert_eq!(out.remote.extract_model, "db-model");
        assert_eq!(out.remote.extract_timeout_ms, 30000);
        // Unset fields inherit; the key is never stored so always inherits.
        assert_eq!(out.remote.distill_model, "env-model");
        assert_eq!(out.remote.api_key, "env-key");
    }

    #[test]
    fn normalised_blanks_become_none() {
        let s = SystemSettings {
            provider: Some("  ".into()),
            extract_model: Some(" gemma4:12b ".into()),
            ..Default::default()
        }
        .normalised()
        .unwrap();
        assert!(s.provider.is_none());
        assert_eq!(s.extract_model.as_deref(), Some("gemma4:12b"));
    }

    #[test]
    fn normalised_rejects_unknown_provider_backend_and_bad_timeouts() {
        assert!(SystemSettings {
            provider: Some("embedded".into()),
            ..Default::default()
        }
        .normalised()
        .is_err());
        assert!(SystemSettings {
            remote_backend: Some("azure".into()),
            ..Default::default()
        }
        .normalised()
        .is_err());
        assert!(SystemSettings {
            extract_timeout_ms: Some(10),
            ..Default::default()
        }
        .normalised()
        .is_err());
    }

    #[test]
    fn parse_models_json_reads_sorts_and_dedupes() {
        let body = r#"{"object":"list","data":[
            {"id":"qwen3.5:9b"},{"id":"gemma4:12b"},{"id":"gemma4:12b"}]}"#;
        assert_eq!(
            parse_models_json(body).unwrap(),
            vec!["gemma4:12b".to_string(), "qwen3.5:9b".to_string()]
        );
    }

    #[test]
    fn parse_models_json_rejects_junk_and_empty() {
        assert!(parse_models_json("not json").is_err());
        assert!(parse_models_json(r#"{"data":[]}"#).is_err());
        assert!(parse_models_json(r#"{"models":["x"]}"#).is_err());
    }

    #[test]
    fn allows_the_local_model_servers_people_actually_run() {
        for ok in [
            "http://127.0.0.1:1234/v1",
            "http://localhost:11434/v1",
            "https://openrouter.ai/api/v1",
        ] {
            assert!(validate_base_url(ok).is_ok(), "{ok} should be allowed");
        }
    }

    #[test]
    fn blocks_cloud_metadata_and_odd_schemes() {
        for bad in [
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/computeMetadata/v1/",
            "file:///etc/passwd",
            "gopher://127.0.0.1:6379/_FLUSHALL",
            "not-a-url",
        ] {
            assert!(validate_base_url(bad).is_err(), "{bad} should be refused");
        }
    }

    #[test]
    fn credentials_in_the_url_cannot_disguise_the_host() {
        assert!(validate_base_url("http://evil@169.254.169.254/").is_err());
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn load_save_roundtrip_and_single_row(pool: sqlx::PgPool) {
        // Fresh database: no row, load returns the empty default.
        assert!(load(&pool).await.unwrap().is_empty());

        let s = SystemSettings {
            provider: Some("remote".into()),
            remote_backend: Some("openai".into()),
            api_base: Some("http://127.0.0.1:11434/v1".into()),
            extract_model: Some("gemma4:12b".into()),
            distill_model: Some("gemma4:12b".into()),
            extract_timeout_ms: Some(60000),
            distill_timeout_ms: Some(180000),
        };
        save(&pool, &s).await.unwrap();
        let loaded = load(&pool).await.unwrap();
        assert_eq!(loaded.extract_model.as_deref(), Some("gemma4:12b"));

        // Saving again updates in place — still exactly one row.
        let s2 = SystemSettings {
            extract_model: Some("other:7b".into()),
            ..s
        };
        save(&pool, &s2).await.unwrap();
        let n: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM system_settings")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(n, 1);
        assert_eq!(
            load(&pool).await.unwrap().extract_model.as_deref(),
            Some("other:7b")
        );
    }
}
