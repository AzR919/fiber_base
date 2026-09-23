#!/usr/bin/env python3
"""
Event-driven local→remote sync using rsync + watchdog.
Usage: sync_watch.py <local_dir> <remote:path>

On each save, syncs only the changed file(s) to remote.
logs/ and plots/ are excluded entirely.
"""

import os
import sys
import fnmatch
import threading
import subprocess
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileDeletedEvent, FileMovedEvent

EXCLUDES = [
    "wandb/", "results/", "ignore/",
    "logs/", "plots/",
    "__pycache__/", "*.pyc", "*.pyo", ".git/",
]

DEBOUNCE_SECS = 0.5


def is_excluded(rel_path):
    """Return True if rel_path matches any EXCLUDES pattern."""
    for pattern in EXCLUDES:
        if pattern.endswith("/"):
            # directory prefix: rel_path starts with this dir
            if rel_path.startswith(pattern) or rel_path == pattern.rstrip("/"):
                return True
        else:
            # glob pattern: match against the filename or full rel path
            if fnmatch.fnmatch(os.path.basename(rel_path), pattern) or \
               fnmatch.fnmatch(rel_path, pattern):
                return True
    return False


def print_synced(files, label="local→remote"):
    names = [os.path.basename(f) for f in files]
    msg = ", ".join(names[:5])
    if len(names) > 5:
        msg += f" (+{len(names)-5} more)"
    print(f"[sync {label}] {msg}")


class DebounceHandler(FileSystemEventHandler):
    def __init__(self, local_dir, remote):
        super().__init__()
        self._local = local_dir.rstrip("/")
        host, remote_path = remote.split(":", 1)
        self._host = host
        self._remote_path = remote_path.rstrip("/")
        self._remote = remote
        self._timer = None
        self._lock = threading.Lock()
        self._pending = set()         # paths to rsync (modified/created)
        self._pending_deletes = set() # paths to delete on remote

    def on_any_event(self, event):
        if event.is_directory:
            return
        rel = os.path.relpath(event.src_path, self._local)
        if is_excluded(rel):
            return
        with self._lock:
            if isinstance(event, (FileDeletedEvent, FileMovedEvent)):
                self._pending_deletes.add(event.src_path)
            else:
                self._pending.add(event.src_path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECS, self._do_sync)
            self._timer.daemon = True
            self._timer.start()

    def _do_sync(self):
        with self._lock:
            changes = set(self._pending)
            deletes = set(self._pending_deletes)
            self._pending.clear()
            self._pending_deletes.clear()

        if changes:
            # rsync -avzR with /. anchor preserves relative paths
            anchored = [
                os.path.join(self._local, ".", os.path.relpath(p, self._local))
                for p in changes
            ]
            cmd = ["rsync", "-avzR"] + anchored + [self._remote.rstrip("/") + "/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                synced = [l for l in result.stdout.splitlines()
                          if l and not l.startswith(("sending", "sent", "total", "receiving", "recv", "./"))]
                if synced:
                    print_synced(synced)
            else:
                print(f"[sync] ERROR: {result.stderr.strip()}", file=sys.stderr)

        for path in deletes:
            rel = os.path.relpath(path, self._local)
            remote_path = f"{self._remote_path}/{rel}"
            result = subprocess.run(
                ["ssh", self._host, f"rm -rf '{remote_path}'"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"[sync local→remote] deleted {rel}")
            else:
                print(f"[sync] ERROR deleting {rel}: {result.stderr.strip()}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Watch local dir and sync changed files to remote.")
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
