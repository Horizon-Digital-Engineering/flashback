# Deploying Flashback

Two paths:

- **[DigitalOcean App Platform (one-click)](#digitalocean-app-platform-one-click)** — paste a button click, app comes up with managed Postgres, automatic HTTPS, and a `.ondigitalocean.app` URL. ~$45/mo.
- **[Droplet / generic VPS (one SSH command)](#digitalocean-droplet--any-ubuntu-vps)** — single VM running docker compose. ~$24/mo. Cheaper, more control, you handle TLS.

Both produce the same product. Pick by cost vs. clickiness.

## DigitalOcean App Platform (one-click)

[![Deploy to DO](https://www.deploytodo.com/do-btn-blue.svg)](https://cloud.digitalocean.com/apps/new?repo=https://github.com/Horizon-Digital-Engineering/flashback/tree/main)

App Platform reads [`.do/app.yaml`](../.do/app.yaml) at the repo root and provisions three pieces (down from four since the Python sidecar was killed in Phase 2a):

| Component   | Plan                  | Why                          |
|-------------|-----------------------|------------------------------|
| `server`    | `apps-s-1vcpu-1gb`    | REST API at `/`. Embeddings + extraction run in-process via fastembed-rs. ~$5/mo. |
| `mcp`       | `apps-s-1vcpu-1gb`    | MCP server at `/mcp`. ~$5/mo. |
| `db`        | `db-s-dev-database`   | Managed Postgres + pgvector. ~$15/mo. |
| **Total**   |                       | **~$25/mo**                  |

### After the app comes up

App Platform handles HTTPS, the public URL (`https://<app>.ondigitalocean.app`), and service-to-service networking. The first thing you need to do is mint a bearer token:

```bash
doctl auth init                              # if you haven't already
doctl apps list                              # find your app id
doctl apps exec <app-id> server -- \
    ./flashback token mint --user=admin --name=initial
```

The token plaintext is printed once. Paste it + the MCP URL into your Claude Desktop / Cursor / Claude Code config (see [the main README](../README.md#wire-up-an-mcp-client)).

### Pinning to a tag

The included spec deploys from `main` with `deploy_on_push: true` semantics off (you have to redeploy explicitly). To pin to a release tag, edit `.do/app.yaml` and change `branch: main` to a specific ref.

### When App Platform is the wrong choice

- **Cost-sensitive deploys.** $25/mo App Platform > $12-24/mo droplet.
- **You want SSH access to the host.** App Platform doesn't give you that. `doctl apps exec` is your only shell.
- **You want to run the embedded LLM on the box itself.** App Platform's instance sizes max out around 4 vCPU / 8 GB RAM and don't expose a GPU — embedded LLM is much happier on a droplet you can scale (or a real AI box). See the runbook below.

For those cases, use the droplet path below.

---

## DigitalOcean Droplet / any Ubuntu VPS

1. **Create a Droplet** in the DigitalOcean dashboard:
   - **Image:** Ubuntu 22.04 LTS (or 24.04)
   - **Plan:** Basic — Regular, **2 vCPU / 4 GB RAM** (`s-2vcpu-4gb`, $24/mo) recommended for the Python sidecar's model load. `s-2vcpu-2gb` works but boots slower.
   - **Authentication:** SSH key (don't use password)
   - **Hostname:** whatever you want (`flashback-prod`)

2. **Run the installer** as root, either via SSH:

   ```bash
   ssh root@<your-droplet-ip>
   curl -sSL https://raw.githubusercontent.com/Horizon-Digital-Engineering/flashback/main/deploy/install.sh | bash
   ```

   …or paste the script into the **User Data** field of the droplet creation form (Advanced options → Add Initialization scripts → User data) — it runs as cloud-init on first boot.

3. **Grab your token** (created on first run only):

   ```bash
   ssh root@<your-droplet-ip> 'cat /root/FLASHBACK_TOKEN.txt'
   ```

   The output looks like:

   ```
   REST endpoint:  http://203.0.113.42:8080
   MCP endpoint:   http://203.0.113.42:8082/mcp
   Bearer token (user=admin): fb_YOUR_TOKEN_HERE
   ```

4. **Wire up Claude Desktop / Cursor / Claude Code** — see [the main README](../README.md#wire-up-an-mcp-client) for the JSON snippet. Paste the URL + bearer into your client config.

### Add TLS (recommended before pointing real clients at it)

Bearer-token auth over plain HTTP is fine on a LAN but you should not run it on the public internet without TLS. The simplest path:

```bash
ssh root@<your-droplet-ip>
apt-get install -y caddy
cat > /etc/caddy/Caddyfile <<'EOF'
flashback.yourdomain.com {
    reverse_proxy /mcp* localhost:8082
    reverse_proxy /* localhost:8080
}
EOF
systemctl restart caddy
```

Point a DNS A record for `flashback.yourdomain.com` at the droplet IP — Caddy fetches a Let's Encrypt cert automatically on first request. Your MCP URL becomes `https://flashback.yourdomain.com/mcp`.

### Costs

| Component                              | $/mo  |
|----------------------------------------|-------|
| Droplet `s-2vcpu-4gb`                  | ~$24  |
| Domain (existing)                      | $0    |
| TLS (Let's Encrypt via Caddy)          | $0    |
| **Total**                              | **~$24** |

Smaller droplets work (`s-2vcpu-2gb` $18/mo, `s-1vcpu-2gb` $12/mo) but the Python sidecar's models eat ~600 MB RAM at idle, so go to 4 GB if you can.

## Generic Ubuntu / Debian VPS

The installer is generic. It works on Hetzner, Vultr, Linode, AWS Lightsail, your home server, anywhere — provided:

- Ubuntu 22.04+ or Debian 12+ (other distros need manual docker install)
- Root access
- Outbound internet (to pull docker images + clone the repo)

```bash
ssh root@<host>
curl -sSL https://raw.githubusercontent.com/Horizon-Digital-Engineering/flashback/main/deploy/install.sh | bash
```

## What the installer does

[`install.sh`](install.sh) is idempotent:

1. Installs docker engine + the compose v2 plugin (skips if already present)
2. Clones / fast-forwards `/opt/flashback` to the current `main`
3. Generates a strong `POSTGRES_PASSWORD` into `/opt/flashback/.env` (chmod 600). On re-runs the existing password is preserved — the Postgres data volume is tied to it.
4. `docker compose up -d --build`
5. Waits for `/health` on the REST server (up to 5 min — first run downloads ~1 GB of Python sidecar models)
6. Mints an initial admin token, writes it to `/root/FLASHBACK_TOKEN.txt` (chmod 600, only on first run)

Re-running the installer pulls new commits and rebuilds without re-minting tokens or touching the database.

## Upgrading

```bash
ssh root@<host>
curl -sSL https://raw.githubusercontent.com/Horizon-Digital-Engineering/flashback/main/deploy/install.sh | bash
```

Same script. It detects an existing install and fast-forwards `main`, leaving `.env` and existing tokens intact.

## Backing up

The only stateful service is Postgres (volume `pgdata`). Snapshot it with `pg_dump`:

```bash
docker compose exec db pg_dump -U flashback flashback | gzip > backup-$(date +%F).sql.gz
```

For DigitalOcean specifically: enable Droplet snapshots ($1–2/mo) for whole-VM backups.

## Security notes

- `POSTGRES_PASSWORD` is randomly generated on first install and stored in `/opt/flashback/.env` (chmod 600). The compose file binds Postgres to `127.0.0.1:5432`, so it's only reachable via the docker network or from localhost on the host.
- The sidecar (`:8081`) is also localhost-only.
- The REST server (`:8080`) and MCP server (`:8082`) bind to `0.0.0.0` and are protected by bearer-token auth. Front them with TLS in production.
- Tokens are sha256-hashed at rest. The plaintext is only shown once at mint time. Rotate any token with `flashback token revoke <id>`.

## Embedded-LLM runbook

The 90% case is `PROVIDER=remote` pointing at Anthropic / OpenAI / a sidecar Ollama / a LAN AI box — no rebuild needed, just env vars. **The embedded-LLM path is for the narrow case where Flashback IS the only service on a dedicated AI box** (DGX Spark, Mac Studio, M-series workstation) and you want a single binary that owns the GPU directly, with no HTTP boundary.

### Decision matrix

| Setup | Right answer |
|---|---|
| Cloud Claude / GPT / OpenRouter | `PROVIDER=remote`, no rebuild |
| Ollama in another docker container | `PROVIDER=remote` + `PROVIDER_REMOTE_API_BASE=http://ollama:11434/v1` |
| Ollama or vLLM on a Mac mini / DGX over LAN | `PROVIDER=remote` + `PROVIDER_REMOTE_API_BASE=http://<box-ip>:11434/v1` |
| Flashback IS the AI box, no other services | `PROVIDER=embedded` + rebuild with `--features embedded-llm` |
| Air-gapped, no network egress allowed | `PROVIDER=embedded` (only path) |

If any row above is true, use that row. The embedded path is bottom-of-list because the rebuild cost (and binary size) is real.

### Building with the feature

```bash
# Cargo workspace knows the feature is on the flashback-nlp crate.
cargo build --release --bin flashback --features flashback-nlp/embedded-llm

# Inside docker compose, edit the server's Dockerfile or pass a build arg:
docker compose build --build-arg FEATURES=flashback-nlp/embedded-llm server
```

Cold compile takes ~10-15 minutes the first time (mistralrs pulls in candle + tokenizers + ndarray + a small army of math crates). Subsequent rebuilds are <2 min.

Binary grows ~150 MB. Model weights are downloaded separately on first run.

### Choosing a model

mistralrs accepts either a Hugging Face repo id (it downloads + caches) or a local GGUF path.

| Model | Size (Q4) | CPU speed | Quality for extraction | Notes |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` (default) | ~400 MB | 5-15 tok/s | Decent | Smallest competent model. CPU-viable. |
| `Qwen/Qwen3-1.7B` | ~1.0 GB | 2-8 tok/s | Good | Solid sweet spot on a Mac mini M-series. |
| `Qwen/Qwen3-4B` | ~2.5 GB | <2 tok/s CPU, fast on GPU | Better | Needs Metal/CUDA to be usable. |
| `microsoft/Phi-4-mini-instruct` | ~2.4 GB | similar | Better at reasoning | Good for the consolidation worker (Phase 4) |
| any other HF chat model that speaks ChatML | varies | varies | varies | Same code path |

Pick the smallest model that gives acceptable extraction quality on YOUR data. Smaller is dramatically faster.

### Runtime config

```bash
# .env
PROVIDER=embedded
PROVIDER_EMBEDDED_MODEL=Qwen/Qwen3-0.6B    # HF repo or local GGUF path
PROVIDER_EMBEDDED_CONTEXT_SIZE=4096
PROVIDER_EMBEDDED_MAX_TOKENS=512
```

Or via CLI:

```bash
PROVIDER=embedded PROVIDER_EMBEDDED_MODEL=Qwen/Qwen3-0.6B ./flashback
```

### Hardware expectations

| Hardware | Expected ingest latency w/ Qwen3-0.6B |
|---|---|
| Small VPS, 2 vCPU CPU only | 30-90s (too slow for production, OK for testing) |
| Modern laptop CPU (Apple Silicon, AMD Ryzen 7) | 3-8s |
| Mac mini M2/M4 with Metal feature flag | 0.5-1.5s |
| DGX Spark / RTX 5090 with CUDA feature flag | 0.1-0.3s |

For GPU acceleration, also pass the relevant mistralrs feature:

```bash
cargo build --release --bin flashback \
    --features flashback-nlp/embedded-llm,mistralrs/metal      # macOS GPU
cargo build --release --bin flashback \
    --features flashback-nlp/embedded-llm,mistralrs/cuda       # NVIDIA
```

### First-boot expectations

1. Server starts → reads `PROVIDER=embedded`.
2. mistralrs downloads the model from HF (one-time, model-size + a tokenizer cache).
3. Cold-loads the model into memory. CPU: 30-90s. Metal/CUDA: ~5s.
4. `/health` flips from `extractor.provider=heuristic` to `extractor.provider=embedded-llm`.
5. First `/memory/ingest` runs the model. Latency in the table above.

### Common pitfalls

- **`ModelBuilder` not found / API mismatch** → mistralrs version churn. The repo pins 0.7. If you bump it manually, expect to patch `embedded.rs` for renamed types.
- **HF download fails behind corporate proxy** → either set `HF_ENDPOINT` or pre-download the model and point `PROVIDER_EMBEDDED_MODEL` at the local directory.
- **OOM on small RAM** → drop to `Qwen3-0.6B` or smaller. Q4 models still need their full weight in RAM.
- **GPU feature builds fail** → CUDA needs the CUDA toolkit installed (`cuda-toolkit` apt package on Linux, Xcode CLT for Metal on macOS).
- **JSON output malformed** → the parser tolerates code fences and surrounding prose. If you get `BadOutput` errors, set `PROVIDER_FALLBACK=heuristic` and check logs to see what the model produced.

### When to swap back to remote

If the embedded model's extraction quality lags noticeably behind Claude Haiku on your data, just flip:

```bash
PROVIDER=remote
PROVIDER_REMOTE_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
```

No rebuild needed (remote provider is always compiled in). You can A/B by ingesting the same content through both providers and comparing the `extraction` JSONB columns.

---

## Going to managed infrastructure

If the small-VPS pattern outgrows you (multiple replicas, autoscaling, managed DB), the next step is Kubernetes manifests. We don't ship them yet — open an issue if you want them.

## Files in this directory

- [`install.sh`](install.sh) — the one-shot bootstrap for droplets / VPSes
- [`../.do/app.yaml`](../.do/app.yaml) — App Platform spec used by the "Deploy to DO" button
- [`../.do/deploy.template.yaml`](../.do/deploy.template.yaml) — the spec-wrapped variant the deploy URL consumes
