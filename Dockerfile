# Build stage — builds the whole workspace, prefetches the embedding model,
# and copies out the binaries we need.
FROM rust:1.97-slim@sha256:14c4fe50ea427dc42381a1a09a9a839c1d2346a2e508cd491bf02c659dbc0ed7 AS builder

# g++ provides libstdc++ which `onig-sys` (a transitive dep of fastembed →
# tokenizers) wants at link time. Cheaper than dropping the onig backend.
RUN apt-get update && apt-get install -y \
    pkg-config \
    ca-certificates \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pinned cache location — the prefetch step writes here at build time and the
# runtime image reads from here. Keeps cold-start fast.
ENV FLASHBACK_FASTEMBED_CACHE=/opt/flashback/fastembed-cache

# Workspace + per-crate manifests first for dep caching.
COPY Cargo.toml Cargo.lock* ./
COPY crates/server/Cargo.toml ./crates/server/Cargo.toml
COPY crates/mcp/Cargo.toml ./crates/mcp/Cargo.toml
COPY crates/nlp/Cargo.toml ./crates/nlp/Cargo.toml

# Dummy sources so cargo can fetch + compile dependencies into a layer.
RUN mkdir -p crates/server/src crates/mcp/src crates/nlp/src crates/nlp/src/bin \
    && echo 'fn main() {}' > crates/server/src/main.rs \
    && echo 'fn main() {}' > crates/mcp/src/main.rs \
    && echo 'pub fn _stub() {}' > crates/nlp/src/lib.rs \
    && echo 'fn main() {}' > crates/nlp/src/bin/prefetch.rs \
    && mkdir migrations \
    && cargo build --release \
    && rm -rf crates/server/src crates/mcp/src crates/nlp/src

# Real sources. `sqlx::migrate!("../../migrations")` is a compile-time macro,
# so migrations must be present at build time.
COPY crates/server/src ./crates/server/src
COPY crates/mcp/src    ./crates/mcp/src
COPY crates/nlp/src    ./crates/nlp/src
COPY migrations        ./migrations
RUN touch crates/server/src/main.rs crates/mcp/src/main.rs crates/nlp/src/lib.rs \
    && cargo build --release --workspace

# Pre-download the default embedding model so the runtime image starts with
# the ONNX weights baked in. The cache path matches FLASHBACK_FASTEMBED_CACHE.
RUN mkdir -p $FLASHBACK_FASTEMBED_CACHE \
    && ./target/release/flashback-nlp-prefetch

# Runtime stage — small image with just the binaries + cached model.
FROM debian:trixie-slim@sha256:b6e2a152f22a40ff69d92cb397223c906017e1391a73c952b588e51af8883bf8

# Match the builder's Debian version (rust:1.95-slim is Debian trixie, glibc 2.41).
# Runtime needs libstdc++6 for `onig`'s C extension that gets dynamically linked.
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV FLASHBACK_FASTEMBED_CACHE=/opt/flashback/fastembed-cache

COPY --from=builder /app/target/release/flashback       ./flashback
COPY --from=builder /app/target/release/flashback-mcp   ./flashback-mcp
COPY --from=builder /opt/flashback/fastembed-cache      /opt/flashback/fastembed-cache
COPY migrations ./migrations

# Run as a non-root user. Limits blast radius if the binary is exploited
# (semgrep dockerfile.security.missing-user). Port 8080 is non-privileged so
# the unprivileged user can bind it.
RUN groupadd --system flashback \
    && useradd --system --gid flashback --home-dir /app --shell /usr/sbin/nologin flashback \
    && chown -R flashback:flashback /app /opt/flashback
USER flashback

EXPOSE 8080
CMD ["./flashback"]
