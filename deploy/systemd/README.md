# Flashback on native Linux (systemd)

For bare-metal Linux where Docker is unwanted or in the way of the hardware —
a DGX, a Strix Halo box, any dedicated server. Two binaries under
`/opt/flashback`, supervised by systemd, talking to the distro's Postgres.
If you just want the stack up on a VPS, the Docker path
([deploy/README.md](../README.md)) is less work; see
[DEPLOY.md](../../DEPLOY.md) for the full platform matrix.

Assumes Ubuntu 24.04+ or Debian 12+; adjust package names elsewhere.

## 1. Postgres + pgvector

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-16-pgvector
```

(Match the pgvector package to your Postgres major — `postgresql-17-pgvector`
on newer releases. On distros without a pgvector package, use the
[PGDG repository](https://wiki.postgresql.org/wiki/Apt).)

```bash
sudo -u postgres psql -c "CREATE ROLE flashback LOGIN PASSWORD '<generate one>'"
sudo -u postgres createdb -O flashback flashback
sudo -u postgres psql -d flashback -c "CREATE EXTENSION IF NOT EXISTS vector"
```

## 2. Binaries

Either download a release tarball for your arch (`x86_64-unknown-linux-gnu`
or `aarch64-unknown-linux-gnu`) from the GitHub releases page, or build on
the box:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
git clone https://github.com/Horizon-Digital-Engineering/flashback
cd flashback
cargo build --release --workspace
```

## 3. Install tree

```bash
sudo useradd --system --home-dir /opt/flashback --shell /usr/sbin/nologin flashback
sudo mkdir -p /opt/flashback/{bin,logs,fastembed-cache}
sudo install -m 755 target/release/{flashback,flashback-mcp,flashback-nlp-prefetch} /opt/flashback/bin/
sudo cp -R migrations /opt/flashback/migrations
```

`/opt/flashback/.env` (chmod 600, owned by `flashback`):

```bash
DATABASE_URL=postgres://flashback:<password>@127.0.0.1:5432/flashback
HOST=0.0.0.0
PORT=8080
AUTO_MIGRATE=1
FLASHBACK_FASTEMBED_CACHE=/opt/flashback/fastembed-cache
RUST_LOG=flashback=info
PROVIDER=heuristic
```

```bash
sudo chown -R flashback:flashback /opt/flashback
sudo chmod 600 /opt/flashback/.env
# Pre-download the embedding model so first boot is fast:
sudo -u flashback FLASHBACK_FASTEMBED_CACHE=/opt/flashback/fastembed-cache \
    /opt/flashback/bin/flashback-nlp-prefetch
```

## 4. Services

```bash
sudo cp deploy/systemd/flashback.service deploy/systemd/flashback-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now flashback flashback-mcp
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8082/health
```

If anything misbehaves, run the built-in diagnostic:

```bash
cd /opt/flashback && sudo -u flashback ./bin/flashback doctor
```

## 5. First token

```bash
cd /opt/flashback && sudo -u flashback ./bin/flashback token mint --user=admin --name=initial
```

Shown once — save it. Point MCP clients at `http://<box>:8082/mcp` with the
Bearer.

## GPU-accelerated extraction (optional)

The heuristic provider needs nothing. For LLM extraction, run an inference
server natively so it owns the GPU — Ollama covers CUDA (DGX/NVIDIA) and
ROCm (Strix Halo/AMD) with the same install:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

Then in `/opt/flashback/.env`:

```bash
PROVIDER=remote
PROVIDER_REMOTE_PROVIDER=openai
PROVIDER_REMOTE_MODEL=qwen2.5:3b
PROVIDER_REMOTE_API_BASE=http://127.0.0.1:11434/v1
PROVIDER_REMOTE_API_KEY=ollama
```

`sudo systemctl restart flashback` to apply. A hosted provider (Anthropic /
OpenRouter) or another box on the LAN is the same flip with a different URL —
see the decision matrix in [deploy/README.md](../README.md#embedded-llm-runbook).

## Update

```bash
cd flashback && git pull --ff-only && cargo build --release --workspace
sudo install -m 755 target/release/{flashback,flashback-mcp,flashback-nlp-prefetch} /opt/flashback/bin/
sudo rm -rf /opt/flashback/migrations && sudo cp -R migrations /opt/flashback/migrations
sudo chown -R flashback:flashback /opt/flashback
sudo systemctl restart flashback flashback-mcp
```

## Backups

```bash
sudo -u postgres pg_dump flashback | gzip > flashback-backup-$(date +%F).sql.gz
```

## Uninstall

```bash
sudo systemctl disable --now flashback flashback-mcp
sudo rm /etc/systemd/system/flashback.service /etc/systemd/system/flashback-mcp.service
sudo systemctl daemon-reload
sudo rm -rf /opt/flashback
sudo userdel flashback
# Only if you also want the data gone — take a final backup first (above):
sudo -u postgres dropdb --if-exists flashback
sudo -u postgres psql -qc "DROP ROLE IF EXISTS flashback"
```

The database survives everything above except the last two lines — a later
reinstall reconnects to it.

## Networking

Same guidance as every deploy: bearer tokens over plain HTTP are fine on a
trusted LAN or tailnet; front with Caddy + a domain for public HTTPS
([deploy/README.md](../README.md#add-tls-recommended-before-pointing-real-clients-at-it)).
