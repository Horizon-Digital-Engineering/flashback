#!/usr/bin/env bash
# Flashback on macOS — native binaries + launchd + Homebrew Postgres.
#
# Turns a Mac (mini / Studio / any M-series) into a Flashback appliance:
# builds the workspace natively, provisions Postgres + pgvector via Homebrew,
# installs two LaunchDaemons that start at boot with no login, and mints the
# first bearer token.
#
# Run from a repo checkout:   ./deploy/macos/install.sh
# Idempotent: re-running rebuilds + restarts without touching the database,
# .env, or existing tokens. For routine upgrades use update.sh.
#
# Native (not Docker) on purpose: Docker on macOS runs inside a Linux VM with
# no Metal access, which forfeits GPU inference for a local Ollama or a future
# embedded-llm build. See DEPLOY.md.

set -euo pipefail

FLASHBACK_HOME="${FLASHBACK_HOME:-/opt/flashback}"
PG_FORMULA="${PG_FORMULA:-postgresql@17}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_USER="$(id -un)"

log() { printf '\033[1;36m[flashback]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[flashback] error:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "macOS only — on Linux use deploy/install.sh (Docker) or deploy/systemd/ (native)."
[[ "$RUN_USER" != "root" ]] || die "run as your admin user, not root — sudo is used only where needed."
command -v brew >/dev/null 2>&1 || die "Homebrew not found — install it from https://brew.sh and re-run."
xcode-select -p >/dev/null 2>&1 || die "Xcode Command Line Tools missing — run: xcode-select --install"

# --- Rust toolchain -------------------------------------------------------
if ! command -v cargo >/dev/null 2>&1; then
    log "Installing Rust via rustup..."
    brew install rustup
    rustup-init -y --no-modify-path --default-toolchain stable
fi
# rustup installs into ~/.cargo; pick it up for this shell if needed.
if ! command -v cargo >/dev/null 2>&1 && [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
command -v cargo >/dev/null 2>&1 || die "cargo still not on PATH after rustup install — open a new shell and re-run."

# --- Postgres + pgvector --------------------------------------------------
if ! brew list --formula "$PG_FORMULA" >/dev/null 2>&1; then
    log "Installing $PG_FORMULA..."
    brew install "$PG_FORMULA"
fi
if ! brew list --formula pgvector >/dev/null 2>&1; then
    log "Installing pgvector..."
    brew install pgvector
fi
brew services start "$PG_FORMULA" >/dev/null

PG_PREFIX="$(brew --prefix "$PG_FORMULA")"
export PATH="$PG_PREFIX/bin:$PATH"

log "Waiting for Postgres..."
pg_ok=""
for _ in $(seq 1 30); do
    if pg_isready -h 127.0.0.1 -q 2>/dev/null; then pg_ok=1; break; fi
    sleep 1
done
[ -n "$pg_ok" ] || die "Postgres did not come up — check: brew services list"

# --- Database role + db (idempotent) --------------------------------------
# The role password is (re)generated only when .env is being created, so
# re-runs never invalidate an existing install.
ENV_FILE="$FLASHBACK_HOME/.env"
DB_PASSWORD=""
if ! sudo test -f "$ENV_FILE"; then
    DB_PASSWORD="$(openssl rand -hex 24)"
fi

psql -h 127.0.0.1 -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='flashback'" | grep -q 1 \
    || psql -h 127.0.0.1 -d postgres -qc "CREATE ROLE flashback LOGIN"
if [ -n "$DB_PASSWORD" ]; then
    psql -h 127.0.0.1 -d postgres -qc "ALTER ROLE flashback WITH LOGIN PASSWORD '$DB_PASSWORD'"
fi
psql -h 127.0.0.1 -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='flashback'" | grep -q 1 \
    || createdb -h 127.0.0.1 -O flashback flashback
psql -h 127.0.0.1 -d flashback -qc "CREATE EXTENSION IF NOT EXISTS vector" \
    || die "could not create the vector extension — Homebrew's pgvector builds against a single Postgres major; make sure it matches $PG_FORMULA (see deploy/macos/README.md)."

# --- Build ----------------------------------------------------------------
log "Building release binaries (the first build takes a few minutes)..."
cd "$REPO_DIR"
cargo build --release --workspace

# --- Install tree ---------------------------------------------------------
log "Installing into $FLASHBACK_HOME..."
sudo mkdir -p "$FLASHBACK_HOME/bin" "$FLASHBACK_HOME/logs" "$FLASHBACK_HOME/fastembed-cache"
sudo install -m 755 \
    target/release/flashback \
    target/release/flashback-mcp \
    target/release/flashback-nlp-prefetch \
    "$FLASHBACK_HOME/bin/"
sudo install -m 755 "$SCRIPT_DIR/backup.sh" "$FLASHBACK_HOME/bin/backup.sh"
sudo rm -rf "$FLASHBACK_HOME/migrations"
sudo cp -R migrations "$FLASHBACK_HOME/migrations"

OLLAMA_BASE="${FLASHBACK_OLLAMA_BASE:-http://127.0.0.1:11434/v1}"

# Ask a local model runtime what it serves — same discovery the systemd
# installer does. Only a model the endpoint reports can be chosen; among
# several, prefer one reporting the `tools` capability, since extraction
# output is parsed as JSON. Prints the chosen model; fails when nothing answers.
discover_ollama_model() {
    command -v python3 >/dev/null 2>&1 || return 1
    python3 - "${OLLAMA_BASE%/v1}" <<'PY' 2>/dev/null
import json, sys, urllib.request

root = sys.argv[1]

def call(path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        root + path, data=data,
        headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)

models = [m["name"] for m in call("/api/tags").get("models", [])]
if not models:
    sys.exit(1)

def tools_capable(name):
    try:
        return "tools" in (call("/api/show", {"model": name}).get("capabilities") or [])
    except Exception:
        return False

print(next((m for m in models if tools_capable(m)), models[0]))
PY
}

if [ -n "$DB_PASSWORD" ]; then
    PROVIDER_BLOCK="# Extraction provider. heuristic = rule-based, in-process, no network — works
# out of the box but produces no semantic facts. Run Ollama natively
# (Metal-accelerated), pull a model, and pick it on the admin settings page —
# or re-run this installer, which discovers whatever the runtime serves:
#   brew install ollama && brew services start ollama && ollama pull <model>
PROVIDER=heuristic"
    if DISCOVERED_MODEL="$(discover_ollama_model)"; then
        log "Model runtime at $OLLAMA_BASE serves '$DISCOVERED_MODEL' — configuring the remote provider."
        PROVIDER_BLOCK="# Discovered from the local model runtime at install time. Change it on the
# admin settings page (/admin/settings) — saved settings win over these values.
PROVIDER=remote
PROVIDER_REMOTE_PROVIDER=openai
PROVIDER_REMOTE_API_BASE=${OLLAMA_BASE}
PROVIDER_REMOTE_MODEL=${DISCOVERED_MODEL}
# Sized for a local model, not a hosted API — see deploy/systemd/README.md.
PROVIDER_REMOTE_EXTRACT_TIMEOUT_MS=60000
PROVIDER_REMOTE_DISTILL_TIMEOUT_MS=180000"
    else
        log "No local model runtime answered at $OLLAMA_BASE — starting with the heuristic provider (no semantic facts)."
    fi
    TMP_ENV="$(mktemp)"
    cat > "$TMP_ENV" <<EOF
# Generated by deploy/macos/install.sh. Edit, then restart:
#   sudo launchctl kickstart -k system/com.flashback.server
DATABASE_URL=postgres://flashback:${DB_PASSWORD}@127.0.0.1:5432/flashback
HOST=0.0.0.0
PORT=8080
AUTO_MIGRATE=1
FLASHBACK_FASTEMBED_CACHE=${FLASHBACK_HOME}/fastembed-cache
RUST_LOG=flashback=info

${PROVIDER_BLOCK}
EOF
    sudo install -m 600 -o "$RUN_USER" "$TMP_ENV" "$ENV_FILE"
    rm -f "$TMP_ENV"
else
    log "Keeping existing $ENV_FILE"
fi

sudo chown -R "$RUN_USER" "$FLASHBACK_HOME"

# --- Pre-download the embedding model so first boot is fast ---------------
if [ -z "$(ls -A "$FLASHBACK_HOME/fastembed-cache" 2>/dev/null)" ]; then
    log "Pre-downloading the embedding model..."
    FLASHBACK_FASTEMBED_CACHE="$FLASHBACK_HOME/fastembed-cache" \
        "$FLASHBACK_HOME/bin/flashback-nlp-prefetch"
fi

# --- launchd daemons (server, mcp, nightly backup) ------------------------
for svc in server mcp backup; do
    plist_dst="/Library/LaunchDaemons/com.flashback.${svc}.plist"
    plist_tmp="$(mktemp)"
    sed -e "s|__FLASHBACK_HOME__|$FLASHBACK_HOME|g" \
        -e "s|__RUN_USER__|$RUN_USER|g" \
        "$SCRIPT_DIR/com.flashback.${svc}.plist.tmpl" > "$plist_tmp"
    # bootout first so a re-run picks up new binaries/plists cleanly.
    sudo launchctl bootout "system/com.flashback.${svc}" 2>/dev/null || true
    sudo install -m 644 -o root -g wheel "$plist_tmp" "$plist_dst"
    rm -f "$plist_tmp"
    sudo launchctl bootstrap system "$plist_dst"
done

# --- Health ---------------------------------------------------------------
log "Waiting for the REST server..."
rest_ok=""
for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then rest_ok=1; break; fi
    sleep 2
done
[ -n "$rest_ok" ] || die "server not healthy after 2 minutes — check $FLASHBACK_HOME/logs/server.log"

mcp_ok=""
for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:8082/health >/dev/null 2>&1; then mcp_ok=1; break; fi
    sleep 2
done
[ -n "$mcp_ok" ] || die "MCP server not healthy — check $FLASHBACK_HOME/logs/mcp.log"

# --- First bearer token (first run only) ----------------------------------
TOKEN_FILE="$FLASHBACK_HOME/FLASHBACK_TOKEN.txt"
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)"
if [ ! -f "$TOKEN_FILE" ]; then
    log "Minting the initial tokens (shown once, saved to $TOKEN_FILE)..."
    # Two surfaces, two tokens: the operator token opens the admin UI, the
    # service token is what MCP/REST clients carry. Neither works on the other.
    {
        echo "REST endpoint:  http://${LAN_IP}:8080"
        echo "MCP endpoint:   http://${LAN_IP}:8082/mcp"
        echo "Admin UI:       http://${LAN_IP}:8080/admin"
        echo
        echo "=== OPERATOR token — sign in to the admin UI with this ==="
        (cd "$FLASHBACK_HOME" && ./bin/flashback token mint --user=admin --name=admin-ui --role=operator)
        echo "=== SERVICE token — for MCP/REST clients ==="
        (cd "$FLASHBACK_HOME" && ./bin/flashback token mint --user=admin --name=initial-client)
    } | tee "$TOKEN_FILE"
    chmod 600 "$TOKEN_FILE"
fi

log "Done."
log "  REST + admin UI:  http://${LAN_IP}:8080  (/admin)"
log "  MCP (clients):    http://${LAN_IP}:8082/mcp"
log "  Config:           $ENV_FILE"
log "  Logs:             $FLASHBACK_HOME/logs/"
log ""
log "Server-grade power settings (recommended for a dedicated box):"
log "  sudo pmset -a sleep 0 displaysleep 5 autorestart 1"
log "Note: with FileVault enabled, the daemons only start after the disk is"
log "unlocked at the boot screen — see deploy/macos/README.md."
