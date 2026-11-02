#!/usr/bin/env python3
"""Phase 2, step 0: probe every official dataset host and record reachability.

Writes outputs/reports/network_probe.json and prints a table.
Run:  .venv/bin/python scripts/00_probe_network.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import registry
from src.utils import net
from src.utils.paths import REPORTS, ensure_dirs, rel, write_json


def main() -> int:
    ensure_dirs()
    results = []
    print(f"{'host':58s} {'ok':3s} {'status':6s} info")
    print("-" * 100)
    for url in registry.PROBE_HOSTS:
        r = net.probe(url)
        results.append({**r.as_dict(), "host": net.iter_hosts([url])[0]})
        info = f"len={r.content_length}" if r.ok else (r.error or "")[:60]
        print(f"{net.iter_hosts([url])[0]:58s} {str(r.ok):3s} {str(r.status):6s} {info}")
    out = write_json(REPORTS / "network_probe.json",
                     {"checked_at": results[0]["checked_at"] if results else None,
                      "results": results})
    reachable = sum(1 for r in results if r["ok"])
    print("-" * 100)
    print(f"reachable: {reachable}/{len(results)}  -> {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
