#!/usr/bin/env bash
# Nightly Flashback backup, run by the com.flashback.backup LaunchDaemon.
# Dumps the database gzipped into /opt/flashback/backups and keeps the most
# recent 14 dumps. Run it by hand any time: /opt/flashback/bin/backup.sh

set -euo pipefail

FLASHBACK_HOME="${FLASHBACK_HOME:-/opt/flashback}"
PG_FORMULA="${PG_FORMULA:-postgresql@17}"
KEEP="${FLASHBACK_BACKUP_KEEP:-14}"

BACKUP_DIR="$FLASHBACK_HOME/backups"
PG_BIN="$(/opt/homebrew/bin/brew --prefix "$PG_FORMULA" 2>/dev/null || echo "/opt/homebrew/opt/$PG_FORMULA")/bin"

mkdir -p "$BACKUP_DIR"
out="$BACKUP_DIR/flashback-$(date +%F-%H%M%S).sql.gz"

"$PG_BIN/pg_dump" -h 127.0.0.1 -U flashback flashback | gzip > "$out"
chmod 600 "$out"

# Retention: keep the newest $KEEP dumps, remove the rest. ls -t sorts
# newest-first; everything after line $KEEP goes.
ls -t "$BACKUP_DIR"/flashback-*.sql.gz 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
    rm -f "$old"
done

echo "backup written: $out ($(du -h "$out" | cut -f1))"
