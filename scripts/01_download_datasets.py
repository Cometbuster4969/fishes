#!/usr/bin/env python3
"""Phase 2: acquire datasets (official first, verified mirror as fallback) + manifest.

Every official URL listed in src/data/registry.py is probed and the result recorded.
Mirrors are then downloaded from codeload.github.com at a pinned commit sha, sha256'd,
extracted to data/raw/<key>/ and logged in data/raw/MANIFEST.json.

Nothing is ever reported as "downloaded" unless the bytes are on disk.

Run:
  .venv/bin/python scripts/01_download_datasets.py            # all mirrors
  .venv/bin/python scripts/01_download_datasets.py --only fish_detection_proxy
  .venv/bin/python scripts/01_download_datasets.py --probe-only
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import registry
from src.utils import net
from src.utils.paths import (ARCHIVES, RAW, TABLES, ensure_dirs, rel, write_csv,
                             write_json)

MANIFEST = RAW / "MANIFEST.json"

# mirrors whose repo zip contains further per-class zips (one zip per class)
NESTED_ZIP_MIRRORS = {"bd_indigenous_fish"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


def safe_extract(zip_path: Path, dest: Path) -> int:
    """Extract a GitHub zip and strip its single top-level directory. Returns file count."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        tops = {n.split("/")[0] for n in names if n}
        strip = tops.pop() if len(tops) == 1 else None
        count = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if strip and name.startswith(strip + "/"):
                name = name[len(strip) + 1:]
            if not name:
                continue
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def extract_nested_class_zips(root: Path) -> list[dict]:
    """BDIndigenousFish2019 ships one zip per class at repo root."""
    report = []
    for zpath in sorted(root.glob("*.zip")):
        cls = zpath.stem
        outdir = root / cls
        n = safe_extract(zpath, outdir)
        report.append({"class_zip": zpath.name, "extracted_to": rel(outdir), "files": n})
        zpath.unlink()
        print(f"    nested: {zpath.name} -> {rel(outdir)} ({n} files)")
    return report


def probe_official() -> list[dict]:
    rows = []
    for entry in registry.OFFICIAL:
        urls = entry["official_urls"]
        if not urls:
            rows.append({"key": entry["key"], "name": entry["name"], "url": None,
                         "ok": False, "status": None, "error": "no official URL resolvable "
                         "from this environment", "access": entry["access"]})
            continue
        for url in urls:
            r = net.probe(url)
            rows.append({"key": entry["key"], "name": entry["name"], "url": url,
                         "ok": r.ok, "status": r.status, "error": r.error,
                         "access": entry["access"]})
            print(f"  probe {entry['key']:22s} {'OK  ' if r.ok else 'FAIL'} {url[:70]}"
                  f"{'' if r.ok else '  <- ' + (r.error or '')[:60]}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="mirror keys to fetch")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    ensure_dirs()
    manifest = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "official_probe": [], "mirrors": []}
    if MANIFEST.exists() and not args.force:
        import json
        manifest = json.loads(MANIFEST.read_text())

    print("== step 1: probe official sources (spec section 3) ==")
    manifest["official_probe"] = probe_official()
    write_json(MANIFEST, manifest)
    if args.probe_only:
        write_csv(TABLES / "download_status.csv", manifest["official_probe"])
        print(f"\nprobe-only: wrote {rel(MANIFEST)}")
        return 0

    targets = registry.MIRRORS
    if args.only:
        targets = [m for m in targets if m["key"] in args.only]

    print("\n== step 2: fetch reachable mirrors (pinned commits) ==")
    by_key = {m["key"]: m for m in manifest.get("mirrors", [])}
    for m in targets:
        key, repo = m["key"], m["repo"]
        dest = RAW / key
        if dest.exists() and any(dest.iterdir()) and not args.force:
            print(f"  skip {key}: {rel(dest)} already populated")
            continue
        print(f"  -> {key}: {repo}")
        try:
            sha = net.github_repo_sha(repo)
        except Exception as exc:  # noqa: BLE001
            print(f"     FAIL resolving commit: {type(exc).__name__}: {exc}")
            by_key[key] = {**m, "ok": False, "error": f"sha resolve: {exc}"}
            continue
        url = net.github_zip_url(repo, sha)
        archive = ARCHIVES / f"{key}.zip"
        rec = net.download(url, archive, timeout=180, expect_bytes_min=1024)
        rec.update({"key": key, "repo": repo, "commit": sha,
                    "licence": m["licence"], "provenance": m["provenance"],
                    "usable_for": m["usable_for"],
                    "identical_to_official": m["identical_to_official"],
                    "substitute_for": m["substitute_for"]})
        if not rec["ok"]:
            print(f"     FAIL download: {rec['error']}")
            by_key[key] = rec
            write_json(MANIFEST, {**manifest, "mirrors": list(by_key.values())})
            continue
        print(f"     downloaded {human(rec['bytes'])} sha256={rec['sha256'][:12]}...")
        if dest.exists() and args.force:
            shutil.rmtree(dest)
        n = safe_extract(archive, dest)
        rec["extracted_files"] = n
        if key in NESTED_ZIP_MIRRORS:
            rec["nested"] = extract_nested_class_zips(dest)
        print(f"     extracted {n} files -> {rel(dest)}")
        by_key[key] = rec
        manifest["mirrors"] = list(by_key.values())
        write_json(MANIFEST, manifest)

    manifest["mirrors"] = list(by_key.values())
    write_json(MANIFEST, manifest)

    rows = []
    for r in manifest["official_probe"]:
        rows.append({"item": r["key"], "source": "official", "url": r["url"],
                     "ok": r["ok"], "status": r["status"], "note": (r["error"] or "")[:120]})
    for r in manifest["mirrors"]:
        rows.append({"item": r["key"], "source": f"mirror:{r['repo']}@{str(r.get('commit'))[:8]}",
                     "url": r["url"], "ok": r["ok"], "status": 200 if r["ok"] else None,
                     "note": (r.get("error") or "")[:120]})
    write_csv(TABLES / "download_status.csv", rows)
    print(f"\nmanifest: {rel(MANIFEST)}\nstatus  : {rel(TABLES / 'download_status.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
