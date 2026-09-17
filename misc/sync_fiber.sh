#!/bin/bash
# Sync fiber_base to a remote cluster (macOS — uses rsync + watchdog).
# Usage: sync_fiber.sh <target>
#   target: SSH host name, e.g. nibi or fir

set -e

TARGET="${1:-nibi}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL="$(cd "$SCRIPT_DIR/.." && pwd)"
REMOTE="${TARGET}:/project/def-maxwl/azr/code/fiber_base"

# Check for an existing sync process for this target
if pgrep -f "sync_watch.py.*${TARGET}" > /dev/null 2>&1; then
    echo "[sync] Session for '${TARGET}' already running."
    exit 0
fi

# Open SSH ControlMaster socket (no-op if already open)
if ! ssh -fN "$TARGET"; then
    echo "[sync] ERROR: SSH connection to '$TARGET' failed." >&2
    exit 1
fi

exec python3 "$SCRIPT_DIR/sync_watch.py" "$LOCAL" "$REMOTE" --reverse
