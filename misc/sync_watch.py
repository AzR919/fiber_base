#!/usr/bin/env python3
"""
Event-driven local→remote sync using rsync + watchdog.
Usage: sync_watch.py <local_dir> <remote:path>

On each save, syncs only the changed file(s) to remote. Renames and deletes
(of files and folders) are mirrored too. Paths in EXCLUDES are never touched.
"""

import os
import sys
import fnmatch
import threading
import subprocess
import argparse
import shlex
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

EXCLUDES = [
    "wandb/", "results/", "ignore/",
    "logs/", "plots/",
    "__pycache__/", "*.pyc", "*.pyo", ".git/",
    # editor / OS clutter
    ".DS_Store", "*.swp", "*.swx", "*~", "4913", ".#*",
]

DEBOUNCE_SECS = 0.5

# Event types that never change file contents (Linux inotify only; harmless elsewhere)
IGNORED_EVENT_TYPES = {"opened", "closed_no_write"}


def is_excluded(rel_path):
    """Return True if rel_path (or any of its parent dirs) matches an EXCLUDES pattern."""
    parts = rel_path.split(os.sep)
    for pattern in EXCLUDES:
        if pattern.endswith("/"):
            # directory pattern: matches that directory at ANY depth
            name = pattern.rstrip("/")
            if any(fnmatch.fnmatch(p, name) for p in parts):
                return True
        else:
            if fnmatch.fnmatch(parts[-1], pattern) or fnmatch.fnmatch(rel_path, pattern):
                return True
    return False


def print_synced(files, label="local→remote"):
    msg = ", ".join(files[:5])
    if len(files) > 5:
        msg += f" (+{len(files)-5} more)"
    print(f"[sync {label}] {msg}", flush=True)


class DebounceHandler(FileSystemEventHandler):
    def __init__(self, local_dir, remote):
        super().__init__()
        self._local = local_dir.rstrip("/")
        host, remote_path = remote.split(":", 1)
        self._host = host
        self._remote_path = remote_path.rstrip("/")
        self._remote = remote.rstrip("/")
        self._timer = None
        self._lock = threading.Lock()       # guards the pending sets / timer
        self._sync_lock = threading.Lock()  # only one sync pass at a time, in order
        self._pending = set()               # rel paths to upload (files or dirs)
        self._pending_deletes = set()       # rel paths to delete on remote

    # ---- event intake -------------------------------------------------

    def _rel(self, path):
        rel = os.path.relpath(path, self._local)
        if rel == "." or rel.startswith(".." + os.sep) or rel == "..":
            return None  # the root itself, or outside the watched tree
        if is_excluded(rel):
            return None
        return rel

    def _queue_change(self, rel):
        self._pending_deletes.discard(rel)
        self._pending.add(rel)

    def _queue_delete(self, rel):
        self._pending.discard(rel)
        # anything queued underneath a deleted dir is moot
        prefix = rel + os.sep
        self._pending = {p for p in self._pending if not p.startswith(prefix)}
        self._pending_deletes.add(rel)

    def on_any_event(self, event):
        if event.event_type in IGNORED_EVENT_TYPES:
            return
        with self._lock:
            touched = False
            src = self._rel(event.src_path)
            if event.event_type == "moved":
                # Covers atomic saves (tmp file renamed onto target) and real
                # renames of files or folders: drop old name, upload new name.
                dest = self._rel(event.dest_path)
                if src:
                    self._queue_delete(src)
                    touched = True
                if dest:
                    self._queue_change(dest)
                    touched = True
            elif src:
                if event.event_type == "deleted":
                    self._queue_delete(src)
                    touched = True
                elif not event.is_directory:
                    # created / modified / closed on a file
                    self._queue_change(src)
                    touched = True
                # dir created/modified: its files raise their own events
            if not touched:
                return
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECS, self._do_sync)
            self._timer.daemon = True
            self._timer.start()

    # ---- sync ---------------------------------------------------------

    def flush(self):
        """Run any pending sync immediately (used on shutdown)."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        self._do_sync()

    def _requeue(self, changes=(), deletes=()):
        """Put failed work back so it is retried with the next event."""
        with self._lock:
            for r in deletes:
                if r not in self._pending:
                    self._pending_deletes.add(r)
            for r in changes:
                if r not in self._pending_deletes:
                    self._pending.add(r)

    def _do_sync(self):
        with self._sync_lock:
            with self._lock:
                changes = set(self._pending)
                deletes = set(self._pending_deletes)
                self._pending.clear()
                self._pending_deletes.clear()

            # Things that vanished before we got to them (e.g. editor temp
            # files) become deletes rather than rsync errors.
            gone = {r for r in changes if not os.path.lexists(os.path.join(self._local, r))}
            changes -= gone
            deletes |= gone
            # Drop deletes already covered by a deleted parent folder.
            deletes = {r for r in deletes
                       if not any(r.startswith(d + os.sep) for d in deletes)}

            # Deletes first, so "rm old/ then upload new/" orders correctly
            # when a folder is replaced. rel paths in both sets are disjoint.
            if deletes:
                targets = " ".join(shlex.quote(f"{self._remote_path}/{r}") for r in sorted(deletes))
                result = subprocess.run(
                    ["ssh", self._host, f"rm -rf -- {targets}"],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    for r in sorted(deletes):
                        print(f"[sync local→remote] deleted {r}", flush=True)
                else:
                    print(f"[sync] ERROR deleting {sorted(deletes)}: {result.stderr.strip()}",
                          file=sys.stderr, flush=True)
                    self._requeue(deletes=deletes)

            if changes:
                # Run rsync from inside the local root with *relative* paths and
                # -R, so "misc/foo.py" lands at <remote>/misc/foo.py. Avoids the
                # "/./" anchor that macOS's openrsync does not honour.
                # Dirs (from folder renames) are sent recursively by -a; the
                # excludes keep their junk out.
                rels = sorted(changes)
                exclude_args = [f"--exclude={p}" for p in EXCLUDES]
                cmd = (["rsync", "-avzR"] + exclude_args
                       + ["--"] + rels + [self._remote + "/"])
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self._local)
                if result.returncode == 0:
                    synced = [l for l in result.stdout.splitlines()
                              if l and not l.endswith("/") and not l.startswith(
                                  ("sending ", "sent ", "total size", "building file list",
                                   "receiving ", "created directory"))]
                    if synced:
                        print_synced(synced)
                else:
                    print(f"[sync] ERROR: {result.stderr.strip()}", file=sys.stderr, flush=True)
                    self._requeue(changes=changes)


def main():
    parser = argparse.ArgumentParser(description="Watch local dir and sync changed files to remote.")
    parser.add_argument("local_dir", help="Local directory to watch")
    parser.add_argument("remote", help="Remote destination, e.g. nibi:/project/.../fiber_base")
    args = parser.parse_args()

    local_dir = str(Path(args.local_dir).expanduser().resolve())
    if not Path(local_dir).is_dir():
        print(f"ERROR: local dir not found: {local_dir}", file=sys.stderr)
        sys.exit(1)
    if ":" not in args.remote:
        print(f"ERROR: remote must look like host:/path, got {args.remote}", file=sys.stderr)
        sys.exit(1)

    print(f"[sync] Watching {local_dir}")
    print(f"[sync] Remote:  {args.remote}")
    print(f"[sync] Excluded: {', '.join(EXCLUDES)}")
    print("[sync] Press Ctrl+C to stop.\n", flush=True)

    handler = DebounceHandler(local_dir, args.remote)
    observer = Observer()
    observer.schedule(handler, local_dir, recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(timeout=1)
    except KeyboardInterrupt:
        print("\n[sync] Stopping.", flush=True)
    finally:
        observer.stop()
        observer.join()
        handler.flush()  # don't lose a save made in the last half second


if __name__ == "__main__":
    main()
