//! Server-side wrapper around `flashback-nlp`.
//!
//! Replaces the old `SidecarClient` (HTTP roundtrip to a Python service).
//! Embeddings now run in-process via fastembed-rs; entity extraction is a
//! rule-based Rust pass that produces strictly better fingerprints than the
//! spaCy NER call it replaces.

use std::sync::Arc;

use async_trait::async_trait;
use flashback_nlp::provider::{
    EmbeddedLlmConfig, EmbeddedLlmProvider, RemoteLlmConfig, RemoteLlmProvider,
};
use flashback_nlp::{
    embed::EmbedderConfig, provider::RemoteBackend, AiProvider, Embedder, ExtractContext,
    Extraction, HeuristicProvider, ProviderError,
};

use crate::config::{FallbackPolicy, ProviderConfig as SrvProviderConfig, ProviderKind};
use crate::error::AppError;

#[derive(Clone)]
pub struct Nlp {
    embedder: Embedder,
    provider: Arc<dyn AiProvider>,
    fallback: FallbackPolicy,
    provider_kind: ProviderKind,
    provider_name: &'static str,
}

impl Nlp {
    pub async fn new(cfg: Config, provider_cfg: &SrvProviderConfig) -> Result<Self, AppError> {
        let mut embed_cfg = EmbedderConfig::default();
        embed_cfg.cache_dir = cfg.cache_dir.clone();
        embed_cfg.show_progress = false;

        let embedder = Embedder::new(embed_cfg)
            .map_err(|e| AppError::Internal(anyhow::anyhow!("embedder init: {e}")))?;
        tracing::info!(
            model = embedder.model_name(),
            dim = embedder.dimension(),
            "Embedding model loaded"
        );

        let (provider, name): (Arc<dyn AiProvider>, &'static str) = match provider_cfg.kind {
            ProviderKind::Remote => {
                let backend = match provider_cfg.remote.backend.as_str() {
                    "anthropic" => RemoteBackend::Anthropic,
                    "openai" => RemoteBackend::OpenAI,
                    _ => RemoteBackend::OpenRouter,
                };
                let rc = RemoteLlmConfig {
                    backend,
                    api_key: provider_cfg.remote.api_key.clone(),
                    api_base: provider_cfg.remote.api_base.clone(),
                    prompt_cache: provider_cfg.remote.prompt_cache,
                    extract_model: provider_cfg.remote.extract_model.clone(),
                    extract_max_tokens: provider_cfg.remote.extract_max_tokens,
                    extract_timeout_ms: provider_cfg.remote.extract_timeout_ms,
                    distill_model: provider_cfg.remote.distill_model.clone(),
                    distill_max_tokens: provider_cfg.remote.distill_max_tokens,
                    distill_timeout_ms: provider_cfg.remote.distill_timeout_ms,
                };
                match RemoteLlmProvider::new(rc) {
                    Ok(p) => {
                        let name: &'static str = match backend {
                            RemoteBackend::OpenRouter => "remote-openrouter",
                            RemoteBackend::Anthropic => "remote-anthropic",
                            RemoteBackend::OpenAI => "remote-openai",
                        };
                        tracing::info!(
                            backend = name,
                            extract_model = %provider_cfg.remote.extract_model,
                            distill_model = %provider_cfg.remote.distill_model,
                            "Remote AI provider configured"
                        );
                        (Arc::new(p), name)
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Remote AI provider misconfigured ({e}); falling back to heuristic. \
                             Set PROVIDER=heuristic explicitly to silence this warning."
                        );
                        (Arc::new(HeuristicProvider), "heuristic")
                    }
                }
            }
            ProviderKind::Embedded => {
                let ec = EmbeddedLlmConfig {
                    model: provider_cfg.embedded.model.clone(),
                    context_size: provider_cfg.embedded.context_size,
                    max_tokens: provider_cfg.embedded.max_tokens,
                    ..Default::default()
                };
                match EmbeddedLlmProvider::new(ec).await {
                    Ok(p) => {
                        tracing::info!(
                            model = %provider_cfg.embedded.model,
                            "Embedded LLM provider configured"
                        );
                        (Arc::new(p), "embedded-llm")
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Embedded LLM init failed ({e}); falling back to heuristic. \
                             Build with --features embedded-llm to enable in-process LLM."
                        );
                        (Arc::new(HeuristicProvider), "heuristic")
                    }
                }
            }
            ProviderKind::Heuristic => {
                tracing::info!("Heuristic AI provider configured (no LLM calls)");
                (Arc::new(HeuristicProvider), "heuristic")
            }
        };

        Ok(Self {
            embedder,
            provider,
            fallback: provider_cfg.fallback,
            provider_kind: provider_cfg.kind,
            provider_name: name,
        })
    }

    pub fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    pub fn provider_kind(&self) -> ProviderKind {
        self.provider_kind
    }

    pub fn provider_name(&self) -> &'static str {
        self.provider_name
    }

    pub fn provider(&self) -> &Arc<dyn AiProvider> {
        &self.provider
    }

    /// True if the configured provider supports distill_facts (i.e. is an
    /// LLM provider, not the heuristic). The consolidation worker checks
    /// this before running the weekly job.
    pub fn provider_can_distill(&self) -> bool {
        self.provider.capabilities().fact_distillation
    }

    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>, AppError> {
        self.embedder
            .embed_one(text)
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("embed: {e}")))
    }

    /// Sync rule-based extraction — used by query-time entity overlap where
    /// 100-500ms LLM round-trips are unacceptable.
    pub fn extract_entities(&self, text: &str) -> Vec<String> {
        flashback_nlp::extract_entities(text)
    }

    /// Full structured extraction via the configured provider. Falls back to
    /// the heuristic provider if remote fails and `PROVIDER_FALLBACK=heuristic`.
    pub async fn extract_full(&self, text: &str) -> Result<Extraction, AppError> {
        let ctx = ExtractContext::default();
        match self.provider.extract(text, &ctx).await {
            Ok(e) => Ok(e),
            Err(ProviderError::NotConfigured(_))
            | Err(ProviderError::Upstream(_))
            | Err(ProviderError::Timeout(_))
            | Err(ProviderError::BadOutput(_)) => match self.fallback {
                FallbackPolicy::Heuristic => {
                    tracing::warn!(
                        "AI provider failed; falling back to heuristic (PROVIDER_FALLBACK=heuristic)"
                    );
                    HeuristicProvider
                        .extract(text, &ctx)
                        .await
                        .map_err(|e| AppError::Internal(anyhow::anyhow!("heuristic fallback: {e}")))
                }
                FallbackPolicy::Fail => Err(AppError::bad_request(
                    "AI extraction provider failed and PROVIDER_FALLBACK is `fail`. \
                     Set PROVIDER_FALLBACK=heuristic to allow degraded ingestion.",
                )),
            },
            Err(e) => Err(AppError::Internal(anyhow::anyhow!("provider: {e}"))),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Config {
    /// Optional fastembed cache directory. If `None`, fastembed picks a
    /// platform default (~/.cache/fastembed). For docker we set it to a
    /// pinned path so the Dockerfile's prefetch step bakes the model into
    /// the image.
    pub cache_dir: Option<std::path::PathBuf>,
}

/// The trait handlers depend on. `Nlp` (the production type) implements it
/// directly; tests construct stubs that implement it without spinning up
/// fastembed or a real provider.
#[async_trait]
pub trait NlpService: Send + Sync {
    fn provider_name(&self) -> &'static str;
    fn provider_can_distill(&self) -> bool;
    fn embedder_model_name(&self) -> &str;
    fn embedder_dimension(&self) -> usize;
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>, AppError>;
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError>;
    fn extract_entities(&self, text: &str) -> Vec<String>;
    async fn extract_full(&self, text: &str) -> Result<Extraction, AppError>;
    /// Forwarded to the underlying AiProvider. Heuristic-only providers
    /// return `Err(NotConfigured)`; consolidation worker checks
    /// `provider_can_distill()` before calling.
    async fn distill_facts(
        &self,
        episodes: &[flashback_nlp::EpisodeRef],
    ) -> Result<Vec<flashback_nlp::DistilledFact>, ProviderError>;
}

#[async_trait]
impl NlpService for Nlp {
    fn provider_name(&self) -> &'static str {
        Nlp::provider_name(self)
    }
    fn provider_can_distill(&self) -> bool {
        Nlp::provider_can_distill(self)
    }
    fn embedder_model_name(&self) -> &str {
        self.embedder.model_name()
    }
    fn embedder_dimension(&self) -> usize {
        self.embedder.dimension()
    }
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>, AppError> {
        Nlp::embed_one(self, text).await
    }
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
        self.embedder
            .embed(texts)
            .await
            .map_err(|e| AppError::Internal(anyhow::anyhow!("embed batch: {e}")))
    }
    fn extract_entities(&self, text: &str) -> Vec<String> {
        Nlp::extract_entities(self, text)
    }
    async fn extract_full(&self, text: &str) -> Result<Extraction, AppError> {
        Nlp::extract_full(self, text).await
    }
    async fn distill_facts(
        &self,
        episodes: &[flashback_nlp::EpisodeRef],
    ) -> Result<Vec<flashback_nlp::DistilledFact>, ProviderError> {
        self.provider.distill_facts(episodes).await
    }
}

/// Arc-shareable handle for `AppState`. Cheap to clone. Stored as a trait
/// object so tests can inject stubs without booting fastembed.
pub type SharedNlp = Arc<dyn NlpService>;
