#!/usr/bin/env bash
# Upgrade a macOS Flashback install: pull, rebuild, swap binaries, restart.
# Never touches .env, the database, or tokens. Run from a repo checkout:
#   ./deploy/macos/update.sh

set -euo pipefail

FLASHBACK_HOME="${FLASHBACK_HOME:-/opt/flashback}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

log() { printf '\033[1;36m[flashback]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[flashback] error:\033[0m %s\n' "$*" >&2; exit 1; }

[ -d "$FLASHBACK_HOME/bin" ] || die "$FLASHBACK_HOME not found — run deploy/macos/install.sh first."

cd "$REPO_DIR"
log "Pulling latest..."
git pull --ff-only

log "Building..."
cargo build --release --workspace

log "Installing binaries + migrations..."
sudo install -m 755 \
    target/release/flashback \
    target/release/flashback-mcp \
    target/release/flashback-nlp-prefetch \
    "$FLASHBACK_HOME/bin/"
sudo rm -rf "$FLASHBACK_HOME/migrations"
sudo cp -R migrations "$FLASHBACK_HOME/migrations"

log "Restarting services..."
sudo launchctl kickstart -k system/com.flashback.server
sudo launchctl kickstart -k system/com.flashback.mcp

log "Waiting for health..."
ok=""
for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 \
        && curl -fsS http://127.0.0.1:8082/health >/dev/null 2>&1; then
        ok=1
        break
    fi
    sleep 2
done
[ -n "$ok" ] || die "services not healthy after restart — check $FLASHBACK_HOME/logs/"

log "Updated to $(git -C "$REPO_DIR" rev-parse --short HEAD)."
