#!/usr/bin/env python3
"""
Event-driven local→remote sync using rsync + watchdog.
Usage: sync_watch.py <local_dir> <remote:path> [--reverse] [--interval N]

On startup: pulls remote → local (remote is ground truth).
While running: pushes local saves → remote.
--reverse    Also poll remote→local every N seconds (default 5)
"""

import sys
import time
import threading
import subprocess
import argparse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

EXCLUDES = [
    "wandb/", "results/", "ignore/",
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
            pass  # no-op sync, stay quiet
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


def reverse_poll_loop(local_dir, remote, interval):
    local = local_dir.rstrip("/") + "/"
    remote_src = remote.rstrip("/") + "/"
    while True:
        time.sleep(interval)
        run_rsync(remote_src, local, label="remote→local")


def main():
    parser = argparse.ArgumentParser(description="Watch local dir and sync to remote via rsync.")
    parser.add_argument("local_dir", help="Local directory to watch")
    parser.add_argument("remote", help="Remote destination, e.g. nibi:/project/.../fiber_base")
    parser.add_argument("--reverse", action="store_true",
                        help="Also poll remote→local periodically")
    parser.add_argument("--interval", type=int, default=5,
                        help="Reverse poll interval in seconds (default: 5)")
    parser.add_argument("--no-pull", action="store_true",
                        help="Skip the initial remote→local pull on startup")
    args = parser.parse_args()

    local_dir = str(Path(args.local_dir).expanduser().resolve())
    if not Path(local_dir).is_dir():
        print(f"ERROR: local dir not found: {local_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[sync] Watching {local_dir}")
    print(f"[sync] Remote:  {args.remote}")
    print(f"[sync] Excludes: {', '.join(EXCLUDES)}")
    if args.reverse:
        print(f"[sync] Reverse poll every {args.interval}s")
    print("[sync] Press Ctrl+C to stop.\n")

    # Remote is ground truth: pull remote → local before starting the watcher
    if args.no_pull:
        print("[sync] Skipping initial pull (--no-pull).")
    else:
        print("[sync] Pulling from remote (remote is ground truth)...")
        run_rsync(args.remote.rstrip("/") + "/", local_dir.rstrip("/") + "/", label="remote→local (init)")

    handler = DebounceHandler(local_dir, args.remote)
    observer = Observer()
    observer.schedule(handler, local_dir, recursive=True)
    observer.start()

    if args.reverse:
        t = threading.Thread(
            target=reverse_poll_loop,
            args=(local_dir, args.remote, args.interval),
            daemon=True,
        )
        t.start()

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        print("\n[sync] Stopping.")
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
