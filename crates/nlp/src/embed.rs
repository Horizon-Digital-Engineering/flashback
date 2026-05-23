//! Text embeddings via fastembed-rs.
//!
//! Default model is `sentence-transformers/all-MiniLM-L6-v2` (384-dim), the
//! same model the Python sidecar used. Embedding output matches the previous
//! shape so the migration is invisible to anything querying the vector index.
//!
//! fastembed's `TextEmbedding::embed` is `&mut self` and synchronous. We:
//! * wrap it in a `parking_lot::Mutex` so it can be shared across async
//!   tasks safely (one in-flight inference at a time);
//! * call it from `tokio::task::spawn_blocking` so the tokio runtime stays
//!   responsive while the model runs.

use std::path::PathBuf;
use std::sync::Arc;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use parking_lot::Mutex;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("embedding model failed to load: {0}")]
    Init(String),
    #[error("embedding inference failed: {0}")]
    Inference(String),
    #[error("embedder pool was poisoned")]
    Poisoned,
}

/// Configuration for the default Embedder. `cache_dir` defaults to fastembed's
/// own resolution (env `FASTEMBED_CACHE` or `~/.cache/fastembed`). Pin it in
/// production to a persistent volume so cold-start doesn't re-download.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    pub model: EmbeddingModel,
    pub cache_dir: Option<PathBuf>,
    pub show_progress: bool,
}

impl Default for EmbedderConfig {
    fn default() -> Self {
        Self {
            model: EmbeddingModel::AllMiniLML6V2,
            cache_dir: None,
            show_progress: false,
        }
    }
}

#[derive(Clone)]
pub struct Embedder {
    inner: Arc<Mutex<TextEmbedding>>,
    dimension: usize,
    model_name: &'static str,
}

impl Embedder {
    pub fn new(cfg: EmbedderConfig) -> Result<Self, EmbedError> {
        let model_name = model_name_for(&cfg.model);
        let mut opts =
            InitOptions::new(cfg.model).with_show_download_progress(cfg.show_progress);
        if let Some(dir) = cfg.cache_dir {
            opts = opts.with_cache_dir(dir);
        }
        let mut model =
            TextEmbedding::try_new(opts).map_err(|e| EmbedError::Init(e.to_string()))?;

        // Probe to record the dimension. Cheap — one inference on a 1-word input.
        let probe = model
            .embed(vec!["probe"], None)
            .map_err(|e| EmbedError::Init(format!("probe failed: {e}")))?;
        let dimension = probe.first().map(|v| v.len()).ok_or_else(|| {
            EmbedError::Init("probe returned no embeddings".to_string())
        })?;

        Ok(Self {
            inner: Arc::new(Mutex::new(model)),
            dimension,
            model_name,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn model_name(&self) -> &'static str {
        self.model_name
    }

    /// Embed a single string. Runs on a blocking thread so the tokio runtime
    /// keeps making progress while ONNX inference is on a worker thread.
    pub async fn embed_one(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        let mut v = self.embed(vec![text.to_string()]).await?;
        v.pop()
            .ok_or_else(|| EmbedError::Inference("empty result".into()))
    }

    /// Batch-embed a list of strings. fastembed is happy to batch internally;
    /// pass the whole list at once for best throughput.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.lock();
            guard
                .embed(texts, None)
                .map_err(|e| EmbedError::Inference(e.to_string()))
        })
        .await
        .map_err(|e| EmbedError::Inference(format!("join: {e}")))?
    }
}

fn model_name_for(m: &EmbeddingModel) -> &'static str {
    // Tiny mapping for /health output. The fastembed enum's Display isn't
    // particularly nice. Extend as we add other models.
    match m {
        EmbeddingModel::AllMiniLML6V2 => "sentence-transformers/all-MiniLM-L6-v2",
        EmbeddingModel::AllMiniLML6V2Q => "sentence-transformers/all-MiniLM-L6-v2 (quantized)",
        EmbeddingModel::AllMiniLML12V2 => "sentence-transformers/all-MiniLM-L12-v2",
        EmbeddingModel::BGESmallENV15 => "BAAI/bge-small-en-v1.5",
        EmbeddingModel::BGEBaseENV15 => "BAAI/bge-base-en-v1.5",
        _ => "fastembed model",
    }
}
