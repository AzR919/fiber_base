"""
Resumable, md5-verified download of a single file.

Data is written to <dest>.part and only renamed to <dest> once size and md5 match, so <dest> existing
always means a complete, verified file. An interrupted .part is resumed with an HTTP Range request.
"""

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.request

CHUNK = 8 * 1024 * 1024


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h


def is_complete(dest, md5sum, size, recorded_md5=None, verify=False):
    """True if dest is already the expected file. Re-hashes unless the previous manifest recorded the
    same md5 for it (pass verify=True to always re-hash)."""
    if not os.path.exists(dest) or os.path.getsize(dest) != size:
        return False
    if recorded_md5 == md5sum and not verify:
        return True
    return md5_of(dest).hexdigest() == md5sum


def _finish(part, dest, h, size, md5sum):
    got = os.path.getsize(part)
    if got != size:
        raise IOError(f"size mismatch: got {got} bytes, expected {size}")
    if h.hexdigest() != md5sum:
        os.remove(part)  # corrupt: restart from scratch on the next attempt
        raise IOError(f"md5 mismatch: got {h.hexdigest()}, expected {md5sum}")
    os.replace(part, dest)


def fetch_url(url, dest, retries=3):
    """Download a small file with no published md5/size (e.g. the ENCODE blacklist) to dest via dest.part.
    Returns the md5 hex digest; callers record it so later runs can detect changes."""
    part = dest + ".part"
    for attempt in range(retries):
        try:
            h = hashlib.md5()
            with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out:
                for block in iter(lambda: resp.read(CHUNK), b""):
                    out.write(block)
                    h.update(block)
            os.replace(part, dest)
            return h.hexdigest()
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            if attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"    retry {attempt + 1}/{retries - 1} for {os.path.basename(dest)} in {wait}s: {e}", flush=True)
            time.sleep(wait)


def fetch(url, dest, md5sum, size, retries=5):
    """Download url to dest (resuming dest.part if present), verifying size and md5. Raises on failure."""
    part = dest + ".part"
    for attempt in range(retries):
        try:
            done = os.path.getsize(part) if os.path.exists(part) else 0
            if done > size:
                os.remove(part)
                done = 0
            h = md5_of(part) if done else hashlib.md5()
            if done == size:
                _finish(part, dest, h, size, md5sum)
                return dest

            headers = {"Range": f"bytes={done}-"} if done else {}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as resp:
                if done and resp.status != 206:  # server ignored the range: start over
                    done, h = 0, hashlib.md5()
                with open(part, "ab" if done else "wb") as out:
                    for block in iter(lambda: resp.read(CHUNK), b""):
                        out.write(block)
                        h.update(block)
            _finish(part, dest, h, size, md5sum)
            return dest
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            if attempt == retries - 1:
                raise
            wait = min(60, 5 * 2 ** attempt)
            print(f"    retry {attempt + 1}/{retries - 1} for {os.path.basename(dest)} in {wait}s: {e}", flush=True)
            time.sleep(wait)
