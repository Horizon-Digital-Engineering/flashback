//! Flashback NLP — embeddings + entity extraction, all-Rust.
//!
//! Replaces the Python sidecar. Phase 2a exposes two primitives:
//!
//! * [`Embedder`]   — sentence-transformer text embeddings via fastembed-rs
//!   (default model: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim).
//! * [`extract_entities`] — pure-Rust noun-phrase + capitalized-word
//!   extraction. Replaces the spaCy NER call. Smaller, faster, catches the
//!   multi-word domain phrases ("deploy target", "auth middleware") that
//!   spaCy's NER misses by design.
//!
//! Phase 2b will add an `AiProvider` trait that wraps this module plus
//! local- and remote-LLM backends behind a single interface.

pub mod embed;
pub mod heuristic;
pub mod provider;

pub use embed::{EmbedError, Embedder, EmbedderConfig};
pub use heuristic::extract_entities;
pub use provider::{
    AiProvider, Capabilities, DistillResponse, DistilledFact, EpisodeRef, ExtractContext,
    Extraction, HeuristicProvider, Intent, Operation, ProviderError,
};
