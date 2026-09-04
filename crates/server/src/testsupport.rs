//! Shared fixtures for tests that need a real `AppState`.
//!
//! Route handlers take `State<AppState>`, so testing any of them meant building
//! a config, an NLP service and two pools by hand — which is why several route
//! modules had no tests at all. This is that assembly, once.
#![cfg(test)]

use std::sync::Arc;

use async_trait::async_trait;
use sqlx::PgPool;

use crate::config::{
    Config, EmbeddedProviderConfig, FallbackPolicy, ProviderConfig, ProviderKind,
    RemoteProviderConfig,
};
use crate::error::AppError;
use crate::nlp::NlpService;
use crate::AppState;
use flashback_nlp::provider::{DistilledFact, EpisodeRef, Extraction, ProviderError};

/// Deterministic, dependency-free NLP. Vectors are uniform and non-zero so
/// cosine stays defined without asserting anything about similarity.
pub struct TestNlp;

#[async_trait]
impl NlpService for TestNlp {
    fn provider_name(&self) -> &'static str {
        "test"
    }
    fn provider_can_distill(&self) -> bool {
        false
    }
    fn embedder_model_name(&self) -> &str {
        "test-embedder"
    }
    fn embedder_dimension(&self) -> usize {
        384
    }
    async fn embed_one(&self, _text: &str) -> Result<Vec<f32>, AppError> {
        Ok(vec![0.1_f32; 384])
    }
    async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, AppError> {
        Ok((0..texts.len()).map(|_| vec![0.1_f32; 384]).collect())
    }
    fn extract_entities(&self, _text: &str) -> Vec<String> {
        Vec::new()
    }
    async fn extract_full(&self, _text: &str) -> Result<Extraction, AppError> {
        Ok(Extraction::empty())
    }
    async fn distill_facts(&self, _e: &[EpisodeRef]) -> Result<Vec<DistilledFact>, ProviderError> {
        Err(ProviderError::NotConfigured("test".into()))
    }
}

pub fn test_config() -> Config {
    Config {
        database_url: "postgres://test/test".to_string(),
        host: "127.0.0.1".to_string(),
        port: 8080,
        auto_migrate: false,
        fastembed_cache_dir: None,
        dev_mode: false,
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
    }
}

/// An `AppState` over one pool. The playground pool is the same handle: tests
/// that care about schema isolation qualify their tables explicitly, and the
/// ones that do not should not need a second container.
pub fn state_from(pool: PgPool) -> AppState {
    AppState {
        pool: pool.clone(),
        playground: pool,
        nlp: Arc::new(TestNlp),
        cfg: Arc::new(test_config()),
    }
}
