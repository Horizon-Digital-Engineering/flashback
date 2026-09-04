//! Settings page endpoints — the runtime control surface for the real
//! extraction/distillation provider.
//!
//! Distinct from the playground's per-operator sandbox settings: what is
//! saved here reconfigures the pipeline every ingest and curation pass runs
//! through, server-wide, live. The flow the page drives:
//!
//!   GET  /admin/settings              — the page
//!   POST /admin/api/settings/models   — what the endpoint actually serves
//!   POST /admin/api/settings/test     — one real extraction with the draft config
//!   POST /admin/api/settings          — persist + hot-swap the provider
//!
//! The models endpoint is what turns the model field into a dropdown: only
//! names the endpoint reports can be picked, so a model that is not pulled
//! can never be configured. The API key is never accepted or returned here —
//! it lives in the environment only (see the system_settings migration).

use axum::extract::State;
use axum::response::Html;
use axum::Json;
use serde::Deserialize;
use serde_json::json;

use crate::auth::AuthUser;
use crate::config::ProviderKind;
use crate::error::{AppError, AppResult};
use crate::settings::{self, SystemSettings};
use crate::AppState;

use super::views;

pub async fn view(
    State(state): State<AppState>,
    user: AuthUser,
) -> Result<Html<String>, super::Error> {
    let stored = settings::load(&state.pool).await.unwrap_or_default();
    let effective = settings::resolve(&state.cfg.provider, &stored);
    let info = views::SettingsInfo {
        stored,
        effective_backend: effective.remote.backend.clone(),
        effective_api_base: effective.remote.api_base.clone(),
        effective_extract_model: effective.remote.extract_model.clone(),
        effective_distill_model: effective.remote.distill_model.clone(),
        effective_extract_timeout_ms: effective.remote.extract_timeout_ms,
        effective_distill_timeout_ms: effective.remote.distill_timeout_ms,
        effective_provider: match effective.kind {
            ProviderKind::Remote => "remote",
            ProviderKind::Embedded => "embedded",
            ProviderKind::Heuristic => "heuristic",
        },
        live_provider: state.nlp.provider_name(),
        live_models: state.nlp.provider_models(),
        can_distill: state.nlp.provider_can_distill(),
        env_has_key: !state.cfg.provider.remote.api_key.is_empty(),
    };
    Ok(Html(views::settings_view(user.scope(), &info)))
}

#[derive(Debug, Deserialize)]
pub struct ModelsQuery {
    pub base: String,
}

/// True when the environment's API key may accompany a probe of `requested`.
/// Only the EFFECTIVE base — what the pipeline itself would call — qualifies.
/// The key must never travel to a caller-supplied URL: a GET that forwarded it
/// anywhere was a one-click credential exfiltration for anyone who could get
/// the operator to follow a link.
fn probe_key_allowed(effective: Option<&str>, requested: &str) -> bool {
    match effective {
        Some(e) => e.trim_end_matches('/') == requested.trim_end_matches('/'),
        None => false,
    }
}

/// List the models an OpenAI-compatible endpoint serves. POST, not GET, so the
/// cross-site guard covers it; the environment key is attached only when the
/// requested base IS the effective base (see `probe_key_allowed`). Probing any
/// other endpoint works keyless — which is all a local model server needs.
pub async fn models(
    State(state): State<AppState>,
    _user: AuthUser,
    Json(q): Json<ModelsQuery>,
) -> AppResult<Json<serde_json::Value>> {
    let stored = settings::load(&state.pool).await.unwrap_or_default();
    let effective = settings::resolve(&state.cfg.provider, &stored);
    let key = state.cfg.provider.remote.api_key.clone();
    let key = (!key.is_empty() && probe_key_allowed(effective.remote.api_base.as_deref(), &q.base))
        .then_some(key);
    match settings::list_models(&q.base, key.as_deref()).await {
        Ok(models) => Ok(Json(json!({ "models": models }))),
        Err(e) => Err(AppError::bad_request(e)),
    }
}

/// Run one real extraction through a provider built from the submitted (not
/// yet saved) settings. Proves the endpoint + model combination works before
/// the operator commits the pipeline to it.
pub async fn test_extraction(
    State(state): State<AppState>,
    _user: AuthUser,
    Json(draft): Json<SystemSettings>,
) -> AppResult<Json<serde_json::Value>> {
    let draft = draft.normalised().map_err(AppError::bad_request)?;
    let cfg = settings::resolve(&state.cfg.provider, &draft);
    if cfg.kind != ProviderKind::Remote {
        return Err(AppError::bad_request(
            "nothing to test: the draft resolves to the heuristic provider, \
             which runs in-process and cannot fail to be reached",
        ));
    }

    use flashback_nlp::provider::{RemoteBackend, RemoteLlmConfig, RemoteLlmProvider};
    use flashback_nlp::{AiProvider, ExtractContext};

    let backend = match cfg.remote.backend.as_str() {
        "anthropic" => RemoteBackend::Anthropic,
        "openai" => RemoteBackend::OpenAI,
        _ => RemoteBackend::OpenRouter,
    };
    let provider = RemoteLlmProvider::new(RemoteLlmConfig {
        backend,
        api_key: cfg.remote.api_key.clone(),
        api_base: cfg.remote.api_base.clone(),
        prompt_cache: cfg.remote.prompt_cache,
        extract_model: cfg.remote.extract_model.clone(),
        extract_max_tokens: cfg.remote.extract_max_tokens,
        extract_timeout_ms: cfg.remote.extract_timeout_ms,
        distill_model: cfg.remote.distill_model.clone(),
        distill_max_tokens: cfg.remote.distill_max_tokens,
        distill_timeout_ms: cfg.remote.distill_timeout_ms,
    })
    .map_err(|e| AppError::bad_request(format!("provider construction failed: {e}")))?;

    // Entity-dense sample so a working model has something to find.
    const SAMPLE: &str = "Met with Dana on Thursday about moving the billing \
                          service to the new database cluster before the June \
                          renewal; she owns the migration runbook.";
    let started = std::time::Instant::now();
    match provider.extract(SAMPLE, &ExtractContext::default()).await {
        Ok(extraction) => Ok(Json(json!({
            "ok": true,
            "model": cfg.remote.extract_model,
            "latency_ms": started.elapsed().as_millis() as u64,
            "sample": SAMPLE,
            "extraction": extraction,
        }))),
        Err(e) => Ok(Json(json!({
            "ok": false,
            "model": cfg.remote.extract_model,
            "latency_ms": started.elapsed().as_millis() as u64,
            "error": e.to_string(),
        }))),
    }
}

/// Persist the settings and swap the live provider to match. Returns what the
/// pipeline is now actually running, which is how a fallback (bad config →
/// heuristic) surfaces instead of masquerading as success.
pub async fn save(
    State(state): State<AppState>,
    _user: AuthUser,
    Json(req): Json<SystemSettings>,
) -> AppResult<Json<serde_json::Value>> {
    let s = req.normalised().map_err(AppError::bad_request)?;
    settings::save(&state.pool, &s).await?;

    let cfg = settings::resolve(&state.cfg.provider, &s);
    let applied = state.nlp.reconfigure_provider(&cfg).await;

    let warnings = provider_warnings(&cfg, &applied).await;

    Ok(Json(json!({
        "saved": s,
        "applied_provider": applied,
        "applied_models": state.nlp.provider_models(),
        "can_distill": state.nlp.provider_can_distill(),
        "warning": (!warnings.is_empty()).then(|| warnings.join(" | ")),
    })))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{spawn_stub, state_from};
    use axum::extract::State;
    use sqlx::PgPool;

    fn models_stub() -> axum::Router {
        axum::Router::new().route(
            "/v1/models",
            axum::routing::get(|headers: axum::http::HeaderMap| async move {
                axum::Json(json!({
                    "data": [
                        { "id": "qwen2.5:3b" },
                        { "id": "gemma3:12b" },
                        { "id": "qwen2.5:3b" },
                    ],
                    "seen_auth": headers
                        .get("authorization")
                        .and_then(|v| v.to_str().ok())
                        .unwrap_or("")
                        .to_string(),
                }))
            }),
        )
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn the_model_list_is_sorted_and_deduped(pool: PgPool) {
        let (base, jh) = spawn_stub(models_stub()).await;
        let state = state_from(pool);
        let out = models(
            State(state),
            operator(),
            axum::Json(ModelsQuery {
                base: format!("{base}/v1"),
            }),
        )
        .await
        .unwrap();
        assert_eq!(out.0["models"], json!(["gemma3:12b", "qwen2.5:3b"]));
        jh.abort();
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn the_environment_key_never_follows_a_caller_supplied_base(pool: PgPool) {
        // The operator can be walked onto a link that names someone else's
        // endpoint; the key must not go with them.
        let (base, jh) = spawn_stub(axum::Router::new().route(
            "/v1/models",
            axum::routing::get(|headers: axum::http::HeaderMap| async move {
                axum::Json(json!({
                    "data": [{ "id": format!(
                        "auth={}",
                        headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("none")
                    ) }]
                }))
            }),
        ))
        .await;

        let mut state = state_from(pool);
        let env_key = "env-key-must-not-travel";
        let mut cfg = crate::testsupport::test_config();
        cfg.provider.remote.api_key = env_key.into();
        cfg.provider.remote.api_base = Some("https://elsewhere.example/v1".into());
        state.cfg = std::sync::Arc::new(cfg);

        let out = models(
            State(state.clone()),
            operator(),
            axum::Json(ModelsQuery {
                base: format!("{base}/v1"),
            }),
        )
        .await
        .unwrap();
        assert_eq!(
            out.0["models"],
            json!(["auth=none"]),
            "the key travelled to a base that is not the effective one"
        );
        jh.abort();
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_base_the_pipeline_would_not_call_is_refused_outright(pool: PgPool) {
        let state = state_from(pool);
        for bad in [
            "http://169.254.169.254/v1",
            "http://[fd00:ec2::254]/v1",
            "file:///etc/passwd",
        ] {
            let err = models(
                State(state.clone()),
                operator(),
                axum::Json(ModelsQuery { base: bad.into() }),
            )
            .await
            .expect_err("{bad} was probed");
            assert!(matches!(err, AppError::BadRequest(_)));
        }
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn there_is_nothing_to_test_when_the_draft_is_the_heuristic(pool: PgPool) {
        let state = state_from(pool);
        let err = test_extraction(
            State(state),
            operator(),
            axum::Json(SystemSettings {
                provider: Some("heuristic".into()),
                ..Default::default()
            }),
        )
        .await
        .expect_err("the heuristic has no endpoint to prove");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn a_draft_that_will_not_normalise_is_refused_before_anything_is_dialled(pool: PgPool) {
        let state = state_from(pool);
        let err = test_extraction(
            State(state),
            operator(),
            axum::Json(SystemSettings {
                provider: Some("remote".into()),
                remote_backend: Some("not-a-backend".into()),
                ..Default::default()
            }),
        )
        .await
        .expect_err("an unknown backend must not reach the provider builder");
        assert!(matches!(err, AppError::BadRequest(_)));
    }

    #[sqlx::test(migrations = "../../migrations")]
    async fn saving_reports_the_provider_that_is_actually_live(pool: PgPool) {
        let state = state_from(pool);
        let out = save(
            State(state.clone()),
            operator(),
            axum::Json(SystemSettings {
                provider: Some("heuristic".into()),
                extract_model: Some("  gemma3:12b  ".into()),
                ..Default::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(out.0["saved"]["extract_model"], "gemma3:12b");
        assert_eq!(out.0["applied_provider"], "test");

        let stored = settings::load(&state.pool).await.unwrap();
        assert_eq!(stored.extract_model.as_deref(), Some("gemma3:12b"));
    }

    #[tokio::test]
    async fn a_remote_config_that_fell_back_to_heuristic_says_so() {
        let mut cfg = crate::testsupport::test_config().provider;
        cfg.kind = ProviderKind::Remote;
        let warnings = provider_warnings(&cfg, "heuristic").await;
        assert_eq!(warnings.len(), 1);
        assert!(
            warnings[0].contains("fell back to heuristic"),
            "{warnings:?}"
        );

        assert!(provider_warnings(&cfg, "remote-openai").await.is_empty());
    }

    #[tokio::test]
    async fn a_model_the_endpoint_does_not_serve_is_called_out_by_role() {
        let (base, jh) = spawn_stub(models_stub()).await;
        let mut cfg = crate::testsupport::test_config().provider;
        cfg.kind = ProviderKind::Remote;
        cfg.remote.api_base = Some(format!("{base}/v1"));
        cfg.remote.extract_model = "qwen2.5:3b".into();
        cfg.remote.distill_model = "a-model-nobody-pulled".into();

        let warnings = provider_warnings(&cfg, "remote-openai").await;
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].starts_with("distill model"), "{warnings:?}");
        jh.abort();
    }

    #[tokio::test]
    async fn an_endpoint_that_cannot_be_asked_leaves_the_models_unverified() {
        let mut cfg = crate::testsupport::test_config().provider;
        cfg.kind = ProviderKind::Remote;
        cfg.remote.api_base = Some("http://127.0.0.1:1/v1".into());
        let warnings = provider_warnings(&cfg, "remote-openai").await;
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("unverified"), "{warnings:?}");
    }

    fn operator() -> AuthUser {
        AuthUser {
            user_id: "op".into(),
            role: crate::auth::TokenRole::Operator,
        }
    }

    #[test]
    fn key_travels_only_to_the_effective_base() {
        assert!(probe_key_allowed(
            Some("http://127.0.0.1:11434/v1"),
            "http://127.0.0.1:11434/v1/"
        ));
        assert!(!probe_key_allowed(
            Some("http://127.0.0.1:11434/v1"),
            "https://evil.example/v1"
        ));
        assert!(!probe_key_allowed(None, "http://127.0.0.1:11434/v1"));
    }
}

/// Everything that saved fine but will not work. Separated from the handler
/// because it is the only part with real branching, and a caller reading `save`
/// should see "persist, apply, then report what is wrong" rather than the
/// model-list walk.
async fn provider_warnings(cfg: &crate::config::ProviderConfig, applied: &str) -> Vec<String> {
    let wanted_remote = cfg.kind == ProviderKind::Remote;
    let mut warnings: Vec<String> = Vec::new();
    if wanted_remote && applied == "heuristic" {
        warnings.push(
            "the saved config resolves to a remote provider but construction \
             failed, so the pipeline fell back to heuristic — check the base \
             URL, and the API key in the server environment if the backend \
             needs one"
                .to_string(),
        );
    }
    // A name the endpoint doesn't serve constructs fine and then fails on
    // every call — the silent-failure shape this page exists to kill. Verify
    // the saved models against the endpoint and say so in the same response.
    if wanted_remote && applied != "heuristic" {
        if let Some(base) = cfg.remote.api_base.as_deref() {
            let key = (!cfg.remote.api_key.is_empty()).then_some(cfg.remote.api_key.as_str());
            match settings::list_models(base, key).await {
                Ok(models) => {
                    for (role, m) in [
                        ("extract", &cfg.remote.extract_model),
                        ("distill", &cfg.remote.distill_model),
                    ] {
                        if !models.iter().any(|x| x == m) {
                            warnings.push(format!(
                                "{role} model '{m}' is not served by the endpoint — every \
                                 {role} call will fail until it is pulled or changed"
                            ));
                        }
                    }
                }
                Err(e) => warnings.push(format!(
                    "saved and applied, but the endpoint's model list could not be \
                     checked ({e}) — model names are unverified"
                )),
            }
        }
    }
    warnings
}
