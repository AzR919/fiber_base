#!/usr/bin/env python3
"""
Event-driven local→remote sync using rsync + watchdog.
Usage: sync_watch.py <local_dir> <remote:path>

Pushes local saves to remote. logs/ and plots/ are excluded entirely.
"""

import sys
import threading
import subprocess
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

EXCLUDES = [
    "wandb/", "results/", "ignore/",
    "logs/", "plots/",
    "__pycache__/", "*.pyc", "*.pyo", ".git/",
]

DEBOUNCE_SECS = 0.5


def build_rsync_cmd(src, dst, extra_flags=None):
    exclude_args = []
    for pattern in EXCLUDES:
        exclude_args += ["--exclude", pattern]
    flags = ["-avz", "--delete"] + (extra_flags or [])
    return ["rsync"] + flags + exclude_args + [src, dst]


def run_rsync(src, dst, label="→"):
    cmd = build_rsync_cmd(src, dst)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        changed = [l for l in result.stdout.splitlines()
                   if l and not l.startswith(("sending", "sent", "total", "receiving", "recv", "./"))]
        if changed:
            print(f"[sync {label}] " + ", ".join(changed[:5]) +
                  (f" (+{len(changed)-5} more)" if len(changed) > 5 else ""))
    else:
        print(f"[sync {label}] ERROR: {result.stderr.strip()}", file=sys.stderr)


class DebounceHandler(FileSystemEventHandler):
    def __init__(self, local_dir, remote):
        super().__init__()
        self._local = local_dir.rstrip("/") + "/"
        self._remote = remote.rstrip("/") + "/"
        self._timer = None
        self._lock = threading.Lock()

    def on_any_event(self, event):
        if event.is_directory:
            return
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECS, self._do_sync)
            self._timer.daemon = True
            self._timer.start()

    def _do_sync(self):
        run_rsync(self._local, self._remote, label="local→remote")


def main():
    parser = argparse.ArgumentParser(description="Watch local dir and sync to remote via rsync.")
    parser.add_argument("local_dir", help="Local directory to watch")
    parser.add_argument("remote", help="Remote destination, e.g. nibi:/project/.../fiber_base")
    args = parser.parse_args()

    local_dir = str(Path(args.local_dir).expanduser().resolve())
    if not Path(local_dir).is_dir():
        print(f"ERROR: local dir not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[sync] Watching {local_dir}")
    print(f"[sync] Remote:  {args.remote}")
    print(f"[sync] Excluded: {', '.join(EXCLUDES)}")
    print("[sync] Press Ctrl+C to stop.\n")

    handler = DebounceHandler(local_dir, args.remote)
    observer = Observer()
    observer.schedule(handler, local_dir, recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        print("\n[sync] Stopping.")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
