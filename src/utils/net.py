"""HTTP helpers with retries, integrity checks and honest failure reporting.

Network in this sandbox is restricted to github.com / codeload.github.com /
api.github.com / pypi.org / files.pythonhosted.org. Every other host fails with a
TLS handshake error. Nothing here hides that: `probe()` and `download()` return
structured results that get written verbatim into outputs/DATASETS.md and
outputs/DECISIONS.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

UA = "GeoFish-AI/0.1 (research; contact: repo maintainer)"
DEFAULT_TIMEOUT = int(os.environ.get("GF_TIMEOUT", 8))
RETRIES = int(os.environ.get("GF_RETRIES", 2))
BACKOFF = 1.0


@dataclass
class ProbeResult:
    url: str
    ok: bool
    status: int | None
    content_length: int | None
    error: str | None
    checked_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _open(url: str, timeout: int = DEFAULT_TIMEOUT, method: str = "GET",
          headers: dict[str, str] | None = None, data: bytes | None = None):
    hdrs = {"User-Agent": UA}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs, method=method, data=data)
    return urllib.request.urlopen(req, timeout=timeout)


def probe(url: str, timeout: int = DEFAULT_TIMEOUT) -> ProbeResult:
    """Cheap reachability check with a short timeout (blocked hosts black-hole TCP)."""
    """HEAD (falling back to a 1-byte GET) to test reachability without downloading."""
    err: str | None = None
    for attempt in range(RETRIES):
        try:
            try:
                r = _open(url, timeout=timeout, method="HEAD")
                return ProbeResult(url, True, r.status,
                                   int(r.headers.get("Content-Length") or 0) or None,
                                   None, _now())
            except urllib.error.HTTPError as he:
                # Some hosts reject HEAD; try a ranged GET before declaring failure.
                r = _open(url, timeout=timeout, headers={"Range": "bytes=0-1023"})
                total = r.headers.get("Content-Range", "").split("/")[-1]
                try:
                    clen = int(total) or None
                except (TypeError, ValueError):
                    clen = None
                return ProbeResult(url, True, he.code if he.code < 500 else r.status,
                                   clen, None, _now())
        except Exception as exc:  # noqa: BLE001 - we report, never raise
            err = f"{type(exc).__name__}: {str(exc)[:200]}"
            if attempt < RETRIES - 1:
                time.sleep(BACKOFF * (attempt + 1))
    return ProbeResult(url, False, None, None, err, _now())


def get_json(url: str, timeout: int = DEFAULT_TIMEOUT,
             headers: dict[str, str] | None = None) -> Any:
    hdrs = {"Accept": "application/vnd.github+json"}
    if headers:
        hdrs.update(headers)
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with _open(url, timeout=timeout, headers=hdrs) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(BACKOFF * (attempt + 1))
    raise RuntimeError(f"get_json failed for {url}: {last}")


def sha256_of(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: str | Path, timeout: int = 120,
             expect_bytes_min: int = 1, max_retries: int = 3) -> dict[str, Any]:
    """Stream `url` to `dest`. Returns a record dict (never raises for network errors)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    rec: dict[str, Any] = {"url": url, "dest": str(dest), "ok": False,
                           "bytes": 0, "sha256": None, "error": None, "started_at": _now()}
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            with _open(url, timeout=timeout) as r, open(tmp, "wb") as fh:
                total = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
            if tmp.stat().st_size < expect_bytes_min:
                raise RuntimeError(f"truncated download: {tmp.stat().st_size} bytes")
            os.replace(tmp, dest)
            rec.update(ok=True, bytes=dest.stat().st_size, sha256=sha256_of(dest),
                       seconds=round(time.time() - t0, 1), finished_at=_now())
            return rec
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            rec["finished_at"] = _now()
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < max_retries - 1:
                time.sleep(BACKOFF * (attempt + 1))
    return rec


def github_repo_sha(repo: str, branch: str | None = None) -> str:
    """Resolve (and thereby pin) the commit sha of a GitHub repo default branch."""
    info = get_json(f"https://api.github.com/repos/{repo}")
    ref = branch or info["default_branch"]
    commit = get_json(f"https://api.github.com/repos/{repo}/commits/{ref}")
    return commit["sha"]


def github_zip_url(repo: str, sha: str | None = None) -> str:
    ref = sha or "HEAD"
    return f"https://codeload.github.com/{repo}/zip/{ref}"


def iter_hosts(urls: Iterable[str]) -> list[str]:
    from urllib.parse import urlparse
    return sorted({urlparse(u).netloc for u in urls if u.startswith("http")})
