#!/bin/bash
# Sync fiber_base to a remote cluster (macOS — uses rsync + watchdog).
# Usage: sync-fiber.sh <target>
#   target: SSH host name, e.g. nibi or fir

set -e

if [[ -z "$1" ]]; then
    echo "Usage: $(basename "$0") <target>  (e.g. nibi, fir)" >&2
    exit 1
fi

TARGET="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE="${TARGET}:/project/def-maxwl/azr/code/fiber_base"

# Check for an existing sync process for this target
if pgrep -f "sync-watch.py.*${TARGET}" > /dev/null 2>&1; then
    echo "[sync] Session for '${TARGET}' already running."
    exit 0
fi

# Open SSH ControlMaster socket (no-op if already open)
ssh -fN "$TARGET" 2>/dev/null || true

exec python3 "$SCRIPT_DIR/sync-watch.py" "$LOCAL" "$REMOTE" --reverse
