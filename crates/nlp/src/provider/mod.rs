//! Pluggable AI provider — extraction (Phase 2c) plus room for
//! summarization / fact distillation in Phase 3.
//!
//! Three implementations:
//!
//! * `heuristic.rs` — rule-based, in-process, no models. Always available.
//!                    Default when no provider is configured.
//! * `remote.rs`    — HTTPS to any OpenAI-compatible / Anthropic / OpenRouter
//!                    endpoint. This is the **90% case**: whether the model is
//!                    cloud-hosted (Anthropic, OpenAI), in a sidecar container
//!                    next to Flashback (Ollama, vLLM), or on a separate box
//!                    on your LAN (Mac mini, DGX Spark), it's all the same
//!                    code path — just a different URL.
//! * `embedded.rs`  — LLM running **in-process inside the Flashback binary**
//!                    via mistral.rs. No HTTP, no sidecar. Feature-gated
//!                    behind `embedded-llm`. Use when Flashback IS the AI
//!                    box's only service or in air-gapped deployments.
//!
//! Callers pick at startup via env config; the trait shape stays stable.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

pub mod embedded;
pub mod heuristic;
pub mod prompt;
pub mod remote;
pub mod schema;

pub use embedded::{EmbeddedEngine, EmbeddedLlmConfig, EmbeddedLlmProvider};
pub use heuristic::HeuristicProvider;
pub use remote::{RemoteBackend, RemoteLlmConfig, RemoteLlmProvider};
pub use schema::*;

// Re-export from schema for ergonomics in `consolidation`.
pub use schema::{DistillResponse, DistilledFact, EpisodeRef};

/// The provider interface.
///
/// `extract()` is the hot-path call run on every ingest. `distill_facts()`
/// is the cold-path call run by the consolidation worker on a schedule;
/// providers without LLM-grade capability return `Err(NotConfigured)` by
/// default (which the consolidator catches + skips).
#[async_trait]
pub trait AiProvider: Send + Sync {
    fn name(&self) -> &'static str;
    fn capabilities(&self) -> Capabilities;

    async fn extract(
        &self,
        text: &str,
        ctx: &ExtractContext,
    ) -> Result<Extraction, ProviderError>;

    /// Given a cluster of related episodic memories, return one or more
    /// distilled semantic facts. Default impl returns `NotConfigured` —
    /// implement it on providers whose `capabilities().fact_distillation`
    /// is true.
    async fn distill_facts(
        &self,
        _episodes: &[EpisodeRef],
    ) -> Result<Vec<DistilledFact>, ProviderError> {
        Err(ProviderError::NotConfigured(
            "this provider does not implement fact distillation".into(),
        ))
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct Capabilities {
    pub extraction: bool,
    pub summarization: bool,
    pub fact_distillation: bool,
    pub typical_latency_ms: u32,
    pub context_window: u32,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            extraction: false,
            summarization: false,
            fact_distillation: false,
            typical_latency_ms: 0,
            context_window: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ExtractContext {
    /// Optional caller-supplied hint about what kind of memory this is
    /// (e.g. "conversation turn", "document chunk"). Currently unused by
    /// heuristic; remote providers may include it in the prompt.
    pub hint: Option<String>,
    /// Recent prior memories — passed in for the remote/local LLM paths so
    /// they can resolve coreference ("the DB" → "Postgres") without us
    /// rebuilding session state inside the provider.
    pub recent_context: Vec<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("provider is not configured: {0}")]
    NotConfigured(String),
    #[error("upstream provider error: {0}")]
    Upstream(String),
    #[error("provider response did not match the expected schema: {0}")]
    BadOutput(String),
    #[error("provider timed out after {0}ms")]
    Timeout(u32),
    #[error("internal: {0}")]
    Internal(String),
}
