#!/usr/bin/env bash
# Remove a macOS Flashback install: launchd daemons, plists, /opt/flashback.
#
# The database is KEPT by default — your memories survive a reinstall, and a
# later install.sh run reuses them. Pass --purge-db to drop the database and
# role too; a final backup is written to your home directory first (skip
# with --no-backup). Homebrew packages (postgresql, pgvector, ollama, rust)
# are never touched — remove those with brew if you want them gone.
#
#   ./deploy/macos/uninstall.sh [--purge-db] [--no-backup] [--yes]

set -euo pipefail

FLASHBACK_HOME="${FLASHBACK_HOME:-/opt/flashback}"
PG_FORMULA="${PG_FORMULA:-postgresql@17}"

PURGE_DB=0
BACKUP=1
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        --purge-db) PURGE_DB=1 ;;
        --no-backup) BACKUP=0 ;;
        --yes) ASSUME_YES=1 ;;
        *) echo "unknown flag: $arg (known: --purge-db --no-backup --yes)" >&2; exit 2 ;;
    esac
done

log() { printf '\033[1;36m[flashback]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[flashback] error:\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Darwin" ]] || die "macOS only."

echo "This will remove:"
echo "  - LaunchDaemons com.flashback.server + com.flashback.mcp"
echo "  - $FLASHBACK_HOME (binaries, logs, .env, FLASHBACK_TOKEN.txt, model cache)"
if [ "$PURGE_DB" -eq 1 ]; then
    echo "  - the 'flashback' database and role"
    [ "$BACKUP" -eq 1 ] && echo "    (a final backup lands in $HOME first)"
else
    echo "The database is kept — re-running install.sh reconnects to it."
fi
if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "Type 'uninstall' to continue: " answer
    [ "$answer" = "uninstall" ] || die "aborted — nothing was changed."
fi

# --- Stop and remove the daemons ------------------------------------------
for svc in server mcp; do
    sudo launchctl bootout "system/com.flashback.${svc}" 2>/dev/null || true
    sudo rm -f "/Library/LaunchDaemons/com.flashback.${svc}.plist"
done
log "Daemons stopped and plists removed."

# --- Database --------------------------------------------------------------
if [ "$PURGE_DB" -eq 1 ]; then
    PG_BIN="$(brew --prefix "$PG_FORMULA" 2>/dev/null)/bin"
    [ -x "$PG_BIN/psql" ] || die "psql not found under $PG_FORMULA — is Postgres still installed? Drop the database manually or re-run without --purge-db."
    if [ "$BACKUP" -eq 1 ]; then
        backup_file="$HOME/flashback-final-backup-$(date +%F-%H%M%S).sql.gz"
        "$PG_BIN/pg_dump" -h 127.0.0.1 -U flashback flashback | gzip > "$backup_file" \
            || die "final backup failed — database left untouched. Re-run with --no-backup to skip it."
        chmod 600 "$backup_file"
        log "Final backup: $backup_file"
    fi
    "$PG_BIN/dropdb" -h 127.0.0.1 --if-exists flashback
    "$PG_BIN/psql" -h 127.0.0.1 -d postgres -qc "DROP ROLE IF EXISTS flashback"
    log "Database and role dropped."
else
    log "Database kept (drop later with: dropdb flashback)."
fi

# --- Install tree ----------------------------------------------------------
sudo rm -rf "$FLASHBACK_HOME"
log "$FLASHBACK_HOME removed."

log "Done. Untouched (remove via brew if unwanted): $PG_FORMULA, pgvector, ollama, rustup."
