#!/usr/bin/env bash
#
# Flashback native (systemd) installer — the scripted form of this directory's
# README. Run it from a checkout:
#
#   sudo ./deploy/systemd/install.sh
#
# Idempotent. Re-running rebuilds, reinstalls the binaries and restarts the
# services; it never regenerates the database password or overwrites an
# existing .env, so it doubles as the upgrade path.
#
# Assumes Postgres is already installed and running with a matching
# postgresql-<major>-pgvector package. Everything else it creates.
#
# Not the Docker installer — that's deploy/install.sh, which puts a git
# checkout at /opt/flashback. The two layouts collide; pick one per host.

set -euo pipefail

INSTALL_DIR="${FLASHBACK_INSTALL_DIR:-/opt/flashback}"
SERVICE_USER="${FLASHBACK_USER:-flashback}"
DB_NAME="${FLASHBACK_DB_NAME:-flashback}"
DB_ROLE="${FLASHBACK_DB_ROLE:-flashback}"
OLLAMA_BASE="${FLASHBACK_OLLAMA_BASE:-http://127.0.0.1:11434/v1}"
OLLAMA_MODEL="${FLASHBACK_OLLAMA_MODEL:-}"
HEALTH_TIMEOUT_S="${FLASHBACK_HEALTH_TIMEOUT_S:-90}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$INSTALL_DIR/.env"

log() { printf '\n[flashback] %s\n' "$*"; }
die() { printf '\n[flashback] ERROR: %s\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "must run as root (try: sudo $0)"
[ -f "$REPO_ROOT/Cargo.toml" ] || die "can't find the repo root from $REPO_ROOT"

# Builds run as the invoking user so cargo's caches and rustup toolchain stay
# in their home, not root's. Skipped entirely if the binaries are already there
# and newer than the sources.
build_user="${SUDO_USER:-root}"

require_postgres() {
    command -v psql >/dev/null 2>&1 || die "postgres client not found — install postgresql first"
    systemctl is-active --quiet postgresql || die "postgresql is not running"
}

# The role's password is the one durable secret here. Generated once, kept in
# .env, and read back out on every subsequent run so re-installing never
# invalidates a working config.
provision_database() {
    local password
    if [ -f "$ENV_FILE" ] && grep -q '^DATABASE_URL=' "$ENV_FILE"; then
        password="$(sed -n 's|^DATABASE_URL=postgres://[^:]*:\([^@]*\)@.*|\1|p' "$ENV_FILE")"
        log "reusing the database password from $ENV_FILE"
    else
        password="$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-32)"
        log "generated a new database password"
    fi
    GENERATED_DB_PASSWORD="$password"

    if sudo -u postgres psql -tAc \
        "SELECT 1 FROM pg_roles WHERE rolname='$DB_ROLE'" | grep -q 1; then
        sudo -u postgres psql -c \
            "ALTER ROLE $DB_ROLE LOGIN PASSWORD '$password'" >/dev/null
        log "role $DB_ROLE exists — password synced to .env"
    else
        sudo -u postgres psql -c \
            "CREATE ROLE $DB_ROLE LOGIN PASSWORD '$password'" >/dev/null
        log "created role $DB_ROLE"
    fi

    if ! sudo -u postgres psql -tAc \
        "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1; then
        sudo -u postgres createdb -O "$DB_ROLE" "$DB_NAME"
        log "created database $DB_NAME"
    fi

    sudo -u postgres psql -d "$DB_NAME" -c \
        "CREATE EXTENSION IF NOT EXISTS vector" >/dev/null \
        || die "pgvector missing — install postgresql-$(psql -V | grep -oE '[0-9]+' | head -1)-pgvector"
}

build_binaries() {
    if [ -x "$REPO_ROOT/target/release/flashback" ] \
        && [ -z "${FLASHBACK_FORCE_BUILD:-}" ]; then
        log "binaries present — skipping build (set FLASHBACK_FORCE_BUILD=1 to override)"
        return
    fi
    log "building (as $build_user)"
    sudo -u "$build_user" -H bash -lc \
        "cd '$REPO_ROOT' && cargo build --release --workspace" \
        || die "cargo build failed"
}

install_tree() {
    id -u "$SERVICE_USER" >/dev/null 2>&1 || {
        useradd --system --home-dir "$INSTALL_DIR" --shell /usr/sbin/nologin "$SERVICE_USER"
        log "created service user $SERVICE_USER"
    }
    mkdir -p "$INSTALL_DIR"/{bin,logs,fastembed-cache}

    # Stop before swapping binaries — replacing a running executable in place
    # is what gives you a service that reports healthy while running old code.
    systemctl stop flashback flashback-mcp 2>/dev/null || true

    install -m 755 \
        "$REPO_ROOT"/target/release/{flashback,flashback-mcp,flashback-nlp-prefetch} \
        "$INSTALL_DIR/bin/"
    rm -rf "$INSTALL_DIR/migrations"
    cp -R "$REPO_ROOT/migrations" "$INSTALL_DIR/migrations"
    log "installed binaries and migrations to $INSTALL_DIR"
}

write_env() {
    if [ -f "$ENV_FILE" ]; then
        log "$ENV_FILE exists — leaving it alone"
        return
    fi

    # Only claim a provider when a model was named. Guessing a model that isn't
    # pulled yields a service that starts, accepts writes, and silently fails
    # every distillation — worse than an honest heuristic fallback.
    local provider_block="PROVIDER=heuristic"
    if [ -n "$OLLAMA_MODEL" ]; then
        provider_block="PROVIDER=remote
PROVIDER_REMOTE_PROVIDER=openai
PROVIDER_REMOTE_API_BASE=$OLLAMA_BASE
PROVIDER_REMOTE_MODEL=$OLLAMA_MODEL
# Defaults are sized for a hosted API. A local model needs far longer before
# the call is considered lost.
PROVIDER_REMOTE_EXTRACT_TIMEOUT_MS=60000
PROVIDER_REMOTE_DISTILL_TIMEOUT_MS=180000"
    fi

    cat > "$ENV_FILE" <<EOF
DATABASE_URL=postgres://$DB_ROLE:$GENERATED_DB_PASSWORD@127.0.0.1:5432/$DB_NAME
HOST=0.0.0.0
PORT=8080
AUTO_MIGRATE=1
FLASHBACK_FASTEMBED_CACHE=$INSTALL_DIR/fastembed-cache
RUST_LOG=flashback=info
$provider_block
EOF
    log "wrote $ENV_FILE"
}

prefetch_embedder() {
    # First start otherwise blocks on a few hundred MB of ONNX download, which
    # reads as a hung service.
    if [ -n "$(ls -A "$INSTALL_DIR/fastembed-cache" 2>/dev/null)" ]; then
        log "embedding model already cached"
        return
    fi
    log "pre-downloading the embedding model"
    sudo -u "$SERVICE_USER" \
        FLASHBACK_FASTEMBED_CACHE="$INSTALL_DIR/fastembed-cache" \
        "$INSTALL_DIR/bin/flashback-nlp-prefetch" || die "embedding prefetch failed"
}

install_services() {
    cp "$REPO_ROOT/deploy/systemd/flashback.service" \
       "$REPO_ROOT/deploy/systemd/flashback-mcp.service" /etc/systemd/system/
    systemctl daemon-reload
    systemctl enable --now flashback flashback-mcp
    log "services enabled and started"
}

wait_for_health() {
    local deadline=$((SECONDS + HEALTH_TIMEOUT_S))
    while [ $SECONDS -lt $deadline ]; do
        if curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1; then
            log "healthy: http://127.0.0.1:8080/health"
            return 0
        fi
        sleep 2
    done
    die "no /health response after ${HEALTH_TIMEOUT_S}s — check: journalctl -u flashback -n 50"
}

require_postgres
provision_database
build_binaries
install_tree
write_env
chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"
chmod 600 "$ENV_FILE"
prefetch_embedder
install_services
wait_for_health

cat <<EOF

Installed to $INSTALL_DIR
  REST + admin UI   http://127.0.0.1:8080
  MCP               http://127.0.0.1:8082
  config            $ENV_FILE
  logs              journalctl -u flashback -f
  validate          $INSTALL_DIR/bin/flashback doctor

Re-run this script after a git pull to rebuild and restart in place.
EOF
