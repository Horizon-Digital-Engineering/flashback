//! Pre-download the default embedding model at Docker build time.
//!
//! Run during the builder stage of the Dockerfile so the runtime image starts
//! with the model already cached and the first user request doesn't pay for
//! the download.
//!
//! Pin the cache via the env var `FLASHBACK_FASTEMBED_CACHE`. The runtime
//! image must use the same path for the cache to be picked up.

use flashback_nlp::{Embedder, embed::EmbedderConfig};

fn main() {
    let cache_dir = std::env::var_os("FLASHBACK_FASTEMBED_CACHE")
        .map(std::path::PathBuf::from);
    let cfg = EmbedderConfig {
        show_progress: true,
        cache_dir,
        ..Default::default()
    };

    eprintln!("[prefetch] caching default embedding model ...");
    match Embedder::new(cfg) {
        Ok(e) => eprintln!(
            "[prefetch] ok — model={}, dim={}",
            e.model_name(),
            e.dimension()
        ),
        Err(err) => {
            eprintln!("[prefetch] failed: {err}");
            std::process::exit(1);
        }
    }
}
