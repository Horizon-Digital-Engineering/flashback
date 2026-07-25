# Flashback on macOS (native + launchd)

Turns a Mac — a mini or Studio is ideal — into a dedicated Flashback server.
Native binaries supervised by launchd, Postgres + pgvector from Homebrew, no
Docker. Native matters on Apple Silicon: Docker runs inside a Linux VM with no
Metal access, so a containerized deploy forfeits GPU inference for a local
Ollama (and for a future `embedded-llm` build). See [DEPLOY.md](../../DEPLOY.md)
for the full platform matrix.

## Install

Prerequisites: [Homebrew](https://brew.sh) and the Xcode Command Line Tools
(`xcode-select --install`).

```bash
git clone https://github.com/Horizon-Digital-Engineering/flashback
cd flashback
./deploy/macos/install.sh
```

What it does, in order: installs Rust (rustup) and `postgresql@17` + `pgvector`
if missing, creates the `flashback` role + database with a generated password,
builds the workspace natively (`cargo build --release`), installs binaries and
migrations into `/opt/flashback`, writes `/opt/flashback/.env` (chmod 600),
pre-downloads the embedding model, installs two LaunchDaemons
(`com.flashback.server`, `com.flashback.mcp`) that start at boot with nobody
logged in, waits for `/health`, and on first run mints an admin bearer token
into `/opt/flashback/FLASHBACK_TOKEN.txt` (chmod 600).

Re-running is safe: it rebuilds and restarts without touching the database,
`.env`, or tokens.

| What | Where |
|---|---|
| REST API + admin UI | `http://<mac>:8080` (`/admin`) |
| MCP (Streamable HTTP) | `http://<mac>:8082/mcp` |
| Config | `/opt/flashback/.env` |
| Logs | `/opt/flashback/logs/{server,mcp}.log` |
| First token | `/opt/flashback/FLASHBACK_TOKEN.txt` |

## Update

```bash
cd flashback
./deploy/macos/update.sh
```

Pulls, rebuilds, swaps binaries, restarts both services, waits for health.

## Service management

```bash
sudo launchctl kickstart -k system/com.flashback.server   # restart
sudo launchctl kickstart -k system/com.flashback.mcp
sudo launchctl print system/com.flashback.server          # status
tail -f /opt/flashback/logs/server.log
```

## GPU-accelerated extraction (optional)

The default `heuristic` provider needs nothing. For LLM-powered extraction,
run Ollama natively — it uses Metal automatically:

```bash
brew install ollama
brew services start ollama
ollama pull qwen2.5:3b
```

Then uncomment the `PROVIDER=remote` block in `/opt/flashback/.env` and
restart the server. Any OpenAI-compatible endpoint works the same way — a
DGX or other AI box on your LAN is just a different
`PROVIDER_REMOTE_API_BASE` URL.

## Reaching it from off-LAN

Prefer [Tailscale](https://tailscale.com): install it on the Mac and your
clients, then use `http://<tailnet-name>:8082/mcp` — no open router ports,
works behind CGNAT, and `tailscale serve` can add HTTPS if a client requires
it. Only reach for a public domain + reverse proxy if you need to hand access
to machines outside your tailnet.

## Dedicated-server settings

```bash
sudo pmset -a sleep 0 displaysleep 5 autorestart 1
```

Disables system sleep and auto-restarts after power loss. Two boot caveats:

- **FileVault:** with FileVault on, the machine waits at the unlock screen
  after a reboot and no daemon starts until someone enters the password.
  Decide which wins for your threat model — disk encryption or unattended
  reboots.
- **Software updates:** leave automatic macOS restarts off; update on your
  schedule.

## Backups

Postgres is the only state:

```bash
/opt/homebrew/opt/postgresql@17/bin/pg_dump -h 127.0.0.1 -U flashback flashback \
    | gzip > flashback-backup-$(date +%F).sql.gz
```

Schedule it (launchd or cron) and ship the file off-box.

## Uninstall

```bash
./deploy/macos/uninstall.sh              # keeps the database — reinstall reconnects to it
./deploy/macos/uninstall.sh --purge-db   # drops the database too, after a final backup to ~
```

Asks for confirmation before touching anything; `--yes` skips the prompt and
`--no-backup` skips the final dump. Homebrew packages (postgresql, pgvector,
ollama, rustup) are never removed — use `brew uninstall` for those.

## Troubleshooting

First move for any issue — the built-in diagnostic:

```bash
cd /opt/flashback && ./bin/flashback doctor
```

It checks config, Postgres, pgvector, migrations, the embedding cache, and
provider reachability, and exits non-zero on failures.

- **`CREATE EXTENSION vector` fails** — Homebrew's `pgvector` formula builds
  against one Postgres major. If you run a different `postgresql@N`, either
  switch (`PG_FORMULA=postgresql@N ./deploy/macos/install.sh` after installing
  a matching pgvector) or build pgvector from source against your server.
- **Service exits immediately** — `tail /opt/flashback/logs/server.log`; the
  usual causes are a bad `DATABASE_URL` or Postgres not running
  (`brew services list`).
- **Port already in use** — change `PORT` in `.env` (REST) or `MCP_PORT` in
  the mcp plist, then reinstall the plist or edit it in place and
  `kickstart -k`.
- **First boot slow** — the embedding model downloads on first use unless the
  installer pre-fetched it; check `/opt/flashback/fastembed-cache`.
