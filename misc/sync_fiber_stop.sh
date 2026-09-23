#!/bin/bash
# Stop all fiber sync processes and close SSH connections.
# Run this before closing your laptop to prevent fail2ban bans.

TARGETS=("nibi" "fir")

# Kill any running sync_watch.py processes
if pkill -f "sync_watch.py" 2>/dev/null; then
    echo "[sync] Watcher stopped."
else
    echo "[sync] No watcher running."
fi

# Close SSH ControlMaster sockets
for t in "${TARGETS[@]}"; do
    if ssh -O exit "$t" 2>/dev/null; then
        echo "[sync] SSH socket closed: $t"
    fi
done

echo "[sync] Done."
