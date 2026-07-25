//! Flashback NLP — embeddings + entity extraction, all-Rust.
//!
//! Replaces the Python sidecar. Two primitives:
//!
//! * [`Embedder`]   — sentence-transformer text embeddings via fastembed-rs
//!   (default model: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim).
//! * [`extract_entities`] — pure-Rust noun-phrase + capitalized-word
//!   extraction. Replaces the spaCy NER call. Smaller, faster, catches the
//!   multi-word domain phrases ("deploy target", "auth middleware") that
//!   spaCy's NER misses by design.
//!
//! The `AiProvider` trait wraps this module plus local- and remote-LLM
//! backends behind a single interface.

pub mod embed;
pub mod heuristic;
pub mod provider;

/// True when this binary was compiled with the `embedded-llm` feature.
/// Lets callers (e.g. `flashback doctor`) report capability without paying
/// for a provider construction attempt that is known to fail.
pub const EMBEDDED_LLM_COMPILED: bool = cfg!(feature = "embedded-llm");

pub use embed::{model_for_key, model_name_for_key, EmbedError, Embedder, EmbedderConfig};
pub use heuristic::extract_entities;
pub use provider::{
    AiProvider, Capabilities, DistillResponse, DistilledFact, EpisodeRef, ExtractContext,
    Extraction, HeuristicProvider, Intent, Operation, ProviderError,
};
