# Deploying Flashback

Flashback is deliberately easy to deploy: **two static Rust binaries** (`flashback`, the REST server + admin UI, and `flashback-mcp`, the MCP server) with **one stateful dependency** — PostgreSQL 16+ with the pgvector extension. Embeddings run in-process (fastembed-rs / ONNX); there is no Python, no sidecar, no GPU requirement for the core engine.

One codebase, several thin packaging paths. Pick by hardware and taste — capability is identical everywhere; only *acceleration for the optional LLM extraction step* varies, and that is a runtime URL, not a build.

## Choose your path

| You have | Recommended path | Guide |
|---|---|---|
| Any box with Docker (Linux VPS, homelab) | Docker Compose | [deploy/README.md](deploy/README.md) |
| A DigitalOcean account, want one-click | App Platform button | [deploy/README.md](deploy/README.md#digitalocean-app-platform-one-click) |
| A Mac (mini / Studio / M-series) as a dedicated server | Native binaries + launchd | [deploy/macos/README.md](deploy/macos/README.md) |
| Native Linux (DGX Spark, Strix Halo box, bare metal) | Native binaries + systemd | [deploy/systemd/README.md](deploy/systemd/README.md) |
| Windows | Planned (native service via winget/scoop). Today: Docker Desktop or WSL2 + the systemd path | — |

All paths produce the same product: REST on `:8080`, MCP (Streamable HTTP) on `:8082`, admin UI at `/admin`, bearer-token auth.

## The three packaging tiers

**Tier A — Docker Compose.** `docker compose up --build` anywhere Docker runs. The default for Linux servers and CI. Zero host setup, easy teardown. The tradeoff: on macOS (and for GPU work generally) Docker runs inside a VM — no Metal, awkward GPU passthrough — so for dedicated hardware prefer Tier B.

**Tier B — native appliance.** Per-platform release binaries (or `cargo build --release` on the box) supervised by the OS's init system: launchd on macOS, systemd on Linux. Full access to the host's GPU for a *native* inference runtime (Ollama, vLLM) via `PROVIDER=remote`. This is the right shape for a Mac mini, a DGX, or a Strix Halo box acting as your household/team memory server.

**Tier C — embedded-LLM builds.** Special-purpose binaries compiled with `--features flashback-nlp/embedded-llm` (plus `mistralrs/metal` or `mistralrs/cuda`) that run the extraction model **in-process** — no HTTP boundary, no second service. Only worth it when Flashback *is* the only service on a dedicated AI box, or in air-gapped deployments. See the [embedded-LLM runbook](deploy/README.md#embedded-llm-runbook).

## Acceleration without lock-in

The LLM-powered extraction step (optional — the default heuristic provider needs nothing) is behind `PROVIDER=remote`, which speaks the OpenAI-compatible API. GPU heterogeneity is therefore **outsourced to the inference runtime**, which already supports every platform we care about:

| Hardware | Inference runtime | Flashback config |
|---|---|---|
| Mac mini / Studio (Metal) | Ollama (native macOS) | `PROVIDER_REMOTE_API_BASE=http://localhost:11434/v1` |
| DGX Spark / NVIDIA (CUDA) | Ollama or vLLM | same, pointing at the box |
| Strix Halo / AMD (ROCm/Vulkan) | Ollama | same, pointing at the box |
| No GPU anywhere | heuristic provider, or a hosted API (Anthropic / OpenRouter) | `PROVIDER=heuristic` or `PROVIDER_REMOTE_PROVIDER=anthropic` |

The Flashback binary is *identical* in every row. Buying a DGX or a Strix box later doesn't change your deployment — it changes one URL in `.env`.

## Release artifacts

Tagged releases (`v*`) build binaries for:

- `aarch64-apple-darwin` (Apple Silicon)
- `x86_64-unknown-linux-gnu`
- `aarch64-unknown-linux-gnu`
- `x86_64-pc-windows-msvc` (experimental)

Each tarball contains `flashback`, `flashback-mcp`, `flashback-nlp-prefetch` (pre-downloads the embedding model so first boot is fast), the `migrations/` directory, and the launchd/systemd templates, with a `.sha256` alongside. Verify with `openssl dgst -sha256 -r <file>` against the published checksum. Building from source on the target box is equally supported (`cargo build --release --workspace`, ~5 minutes on Apple Silicon).

## Cross-cutting notes

**Networking.** Bearer tokens over plain HTTP are fine on a trusted LAN. For remote access prefer Tailscale (no open ports, works behind CGNAT; `tailscale serve` can add HTTPS). For public exposure put Caddy in front — see [deploy/README.md](deploy/README.md#add-tls-recommended-before-pointing-real-clients-at-it).

**Backups.** Postgres is the only state. `pg_dump -U flashback flashback | gzip > backup-$(date +%F).sql.gz` on a schedule, shipped off-box.

**Upgrades.** Tier A: re-run the installer. Tier B: pull + rebuild + restart the services (`deploy/macos/update.sh` does this on macOS; `systemctl restart flashback flashback-mcp` after a rebuild on Linux).

## Roadmap

- `flashback setup` / `flashback doctor` / `flashback upgrade` subcommands — the binary becomes its own cross-platform installer (detect init system, provision Postgres, write service files, mint the first token).
- Managed/embedded Postgres option (zero-dependency single-binary install).
- Homebrew tap (macOS + Linux), then winget/scoop manifests for Windows.
- Kubernetes manifests if demand shows up — open an issue.
