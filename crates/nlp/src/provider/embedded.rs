//! Embedded LLM provider — gated behind the `embedded-llm` feature.
//!
//! Runs an LLM **in-process inside the Flashback binary** via mistral.rs.
//! No separate Ollama process, no HTTP roundtrip, no API key. The 90% case
//! is `RemoteLlmProvider` pointing at an Ollama / vLLM / cloud endpoint —
//! this provider is the narrow path for:
//!
//!   - dedicated AI-box deployments where Flashback IS the only service
//!     (DGX Spark, Mac Studio, M-series workstation) and you want a single
//!     binary that owns the GPU/Metal directly
//!   - air-gapped / high-security environments where no network egress to
//!     a model endpoint is acceptable
//!   - tight single-container builds where adding a sidecar isn't an option
//!
//! Default model is Qwen3-0.6B-Instruct (small, fast on CPU, decent at
//! structured extraction). Swap via `PROVIDER_EMBEDDED_MODEL=<hf-repo-or-path>`.
//! mistral.rs abstracts CPU / CUDA / Metal so the same code runs on a $5
//! droplet (CPU, slow) or a DGX Spark (GPU, fast) with no source changes.

use async_trait::async_trait;

#[cfg(feature = "embedded-llm")]
use super::prompt::{build_system_prompt, build_user_prompt};
use super::{AiProvider, Capabilities, ExtractContext, Extraction, ProviderError};

#[cfg(feature = "embedded-llm")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct EmbeddedLlmConfig {
    /// Hugging Face repo id (e.g. `Qwen/Qwen3-0.6B`) OR a local directory
    /// containing a GGUF file. Auto-detected: anything starting with `/` or
    /// `./` or ending in `.gguf` is treated as a local path, otherwise as HF.
    pub model: String,
    /// Optional system override of which inference engine the embedded
    /// provider uses. Default: mistralrs.
    pub engine: EmbeddedEngine,
    /// Device hint. "auto" lets the engine pick the fastest available.
    pub device: String,
    pub context_size: usize,
    /// Max generated tokens for a single extraction call.
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddedEngine {
    /// GGUF / HF via mistral.rs. Default. Best general-purpose choice.
    MistralRs,
    /// ONNX via ort. Best for purpose-built extraction models like NuExtract.
    /// Not wired in Phase 2b — Phase 3.
    Onnx,
    /// Pure-Rust ML via candle. Niche; useful when ort is hard to ship.
    Candle,
    /// Direct llama.cpp bindings. Very niche.
    LlamaCpp,
}

impl Default for EmbeddedLlmConfig {
    fn default() -> Self {
        Self {
            model: "Qwen/Qwen3-0.6B".to_string(),
            engine: EmbeddedEngine::MistralRs,
            device: "auto".to_string(),
            context_size: 4096,
            max_tokens: 512,
        }
    }
}

pub struct EmbeddedLlmProvider {
    #[allow(dead_code)]
    config: EmbeddedLlmConfig,
    #[cfg(feature = "embedded-llm")]
    model: Arc<mistralrs::Model>,
}

impl EmbeddedLlmProvider {
    /// Async constructor — loads the model. Server calls this at startup.
    ///
    /// On a CPU-only host this can take 30-90s for a 600M-param Q4 model.
    /// On Metal/CUDA it's near-instant.
    pub async fn new(_config: EmbeddedLlmConfig) -> Result<Self, ProviderError> {
        #[cfg(not(feature = "embedded-llm"))]
        {
            Err(ProviderError::NotConfigured(
                "this build was compiled without the `embedded-llm` feature; \
                 rebuild with `--features embedded-llm` to run an in-process LLM. \
                 For the 90% case use PROVIDER=remote pointing at Ollama / a hosted API."
                    .into(),
            ))
        }
        #[cfg(feature = "embedded-llm")]
        {
            use mistralrs::TextModelBuilder;
            tracing::info!(
                model = %_config.model,
                "Loading embedded LLM via mistral.rs — this can take a while on first run"
            );
            let model = TextModelBuilder::new(&_config.model)
                .with_logging()
                .build()
                .await
                .map_err(|e| ProviderError::Internal(format!("mistralrs build: {e}")))?;
            tracing::info!("Embedded LLM ready");
            Ok(Self {
                config: _config,
                model: Arc::new(model),
            })
        }
    }
}

#[async_trait]
impl AiProvider for EmbeddedLlmProvider {
    fn name(&self) -> &'static str {
        "embedded-llm"
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            extraction: true,
            summarization: false,     // Phase 3
            fact_distillation: false, // Phase 3
            typical_latency_ms: 200,
            context_window: 4096,
        }
    }

    async fn extract(
        &self,
        _text: &str,
        _ctx: &ExtractContext,
    ) -> Result<Extraction, ProviderError> {
        #[cfg(not(feature = "embedded-llm"))]
        {
            Err(ProviderError::NotConfigured(
                "EmbeddedLlmProvider not built — recompile with --features embedded-llm".into(),
            ))
        }
        #[cfg(feature = "embedded-llm")]
        {
            use mistralrs::{TextMessageRole, TextMessages};

            let messages = TextMessages::new()
                .add_message(TextMessageRole::System, build_system_prompt())
                .add_message(
                    TextMessageRole::User,
                    &build_user_prompt(_text, &_ctx.recent_context),
                );

            let response = self
                .model
                .send_chat_request(messages)
                .await
                .map_err(|e| ProviderError::Upstream(format!("embedded inference: {e}")))?;

            let raw = response
                .choices
                .first()
                .and_then(|c| c.message.content.as_ref())
                .ok_or_else(|| ProviderError::BadOutput("empty response".into()))?
                .clone();

            super::remote::parse_extraction(&raw)
        }
    }
}
