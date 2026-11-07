#!/usr/bin/env python3
"""Kaggle environment installer + verifier.

Why this exists: `pip install -r requirements-kaggle.txt` is atomic - one unresolvable
package (or one package whose build needs a compiler) aborts the whole transaction and
installs nothing. On the Kaggle image that showed up as "open_clip_torch MISSING" with no
explanation, because the caller piped the output through `tail`.

This script instead:
  * checks what is importable right now,
  * installs *only* what is missing, one package at a time,
  * prints the real pip error (last lines) when one fails, instead of hiding it,
  * installs open-clip with --no-deps + its pure-python deps, so pip never tries to
    reinstall torch/torchvision inside the Kaggle image,
  * tries a few open-clip versions (BioCLIP is the audit's zero-shot opinion),
  * writes outputs/logs/env_report.json and exits non-zero if a REQUIRED package is absent.

Usage (inside the notebook):
    !python scripts/kaggle_env.py              # install what is missing, then verify
    !python scripts/kaggle_env.py --check-only # just report
    !python scripts/kaggle_env.py --json /kaggle/working/outputs/logs/env_report.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


@dataclass
class Pkg:
    module: str                     # import name
    spec: str | None                # pip requirement (None = must be preinstalled)
    why: str
    required: bool
    dist: str | None = None         # distribution name for version lookup
    no_deps_with: list[str] = field(default_factory=list)   # extra deps for --no-deps
    fallbacks: list[str] = field(default_factory=list)      # tried in order
    pip_module: str | None = None   # for modules that are not the dist name


PACKAGES: list[Pkg] = [
    Pkg("torch", None, "deep learning runtime (preinstalled on Kaggle)", True, dist="torch"),
    Pkg("torchvision", None, "image transforms", True, dist="torchvision"),
    Pkg("numpy", "numpy", "arrays", True, dist="numpy"),
    Pkg("pandas", "pandas", "manifests, metrics tables", True, dist="pandas"),
    Pkg("PIL", "pillow", "image IO", True, dist="pillow", pip_module="pillow"),
    Pkg("cv2", "opencv-python-headless", "frames, crops, Laplacian quality check", True,
        dist="opencv-python-headless", pip_module="opencv-python-headless"),
    Pkg("timm", "timm>=1.0.11", "ResNet/EfficientNet/ConvNeXt/ViT/DINOv2 backbones",
        True, dist="timm"),
    Pkg("ultralytics", "ultralytics>=8.4.0", "YOLO11 / YOLOv12 / RT-DETR detector", True,
        dist="ultralytics"),
    Pkg("open_clip", "open-clip-torch", "BioCLIP zero-shot audit opinion", True,
        dist="open-clip-torch", pip_module="open-clip-torch",
        fallbacks=["open-clip-torch>=4.0", "open-clip-torch==3.3.0",
                   "open-clip-torch==2.32.0"],
        no_deps_with=["tiktoken", "ftfy", "wcwidth", "regex", "huggingface-hub",
                      "safetensors"]),
    Pkg("cleanlab", "cleanlab>=2.6.6", "confident-learning label audit", True,
        dist="cleanlab"),
    Pkg("imagehash", "imagehash>=4.3.1", "pHash duplicate detection", True, dist="imagehash"),
    Pkg("lap", "lap>=0.5.12", "ByteTrack / BoT-SORT assignment", True, dist="lap"),
    Pkg("sklearn", "scikit-learn", "metrics, logistic probe", True, dist="scikit-learn",
        pip_module="scikit-learn"),
    Pkg("matplotlib", "matplotlib", "figures", True, dist="matplotlib"),
    Pkg("seaborn", "seaborn", "confusion-matrix heatmaps", False, dist="seaborn"),
    Pkg("motmetrics", "motmetrics>=1.4.0", "MOTA / ID switches (spec 7)", False,
        dist="motmetrics"),
    Pkg("geopandas", "geopandas>=1.0", "spatial outputs (spec 10)", False, dist="geopandas"),
    Pkg("folium", "folium>=0.17", "interactive density map", False, dist="folium"),
    Pkg("mapclassify", "mapclassify>=2.8", "choropleth classification", False,
        dist="mapclassify"),
    Pkg("contextily", "contextily>=1.6", "basemap tiles", False, dist="contextily"),
    Pkg("lightgbm", "lightgbm>=4.5", "species presence model (spec 9)", False,
        dist="lightgbm"),
    Pkg("torch.utils.tensorboard", "tensorboard>=2.17", "run logging", False,
        dist="tensorboard"),
]


def installed_version(pkg: Pkg) -> str | None:
    for name in filter(None, (pkg.dist, pkg.module)):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def importable(module: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:200]


def pip_install(spec: str, no_deps_with: list[str] | None = None) -> tuple[bool, str]:
    """Install one requirement. Returns (ok, tail_of_output)."""
    if no_deps_with:
        # open-clip case: never let pip touch torch/torchvision inside the Kaggle image
        cmds = [["pip", "install", "-q", "--no-deps", spec],
                ["pip", "install", "-q", *no_deps_with]]
    else:
        cmds = [["pip", "install", "-q", spec]]
    tails = []
    for cmd in cmds:
        res = subprocess.run([sys.executable, "-m", *cmd], capture_output=True, text=True)
        if res.returncode != 0:
            tail = (res.stderr or res.stdout).strip().splitlines()[-12:]
            return False, "\n".join(tail)
        if res.stderr.strip():
            tails.append(res.stderr.strip().splitlines()[-1])
    return True, "; ".join(tails)


def ensure(pkg: Pkg, check_only: bool) -> dict:
    ok, err = importable(pkg.module)
    rec = {"module": pkg.module, "dist": pkg.dist, "required": pkg.required,
           "why": pkg.why, "present": ok, "version_before": installed_version(pkg),
           "action": "none", "error": err, "version_after": None}

    if ok:
        rec["version_after"] = rec["version_before"]
        rec["action"] = "already present"
        return rec
    if check_only:
        rec["action"] = "missing (check-only)"
        return rec
    if pkg.spec is None:
        rec["action"] = "missing and not installable by pip (expected preinstalled)"
        return rec

    specs = [pkg.spec, *pkg.fallbacks]
    errors = []
    for spec in specs:
        good, tail = pip_install(spec, pkg.no_deps_with or None)
        after_ok, after_err = importable(pkg.module)
        if good and after_ok:
            rec.update(action=f"installed {spec}", version_after=installed_version(pkg),
                       error=None)
            return rec
        errors.append(f"[{spec}] {'import still failing: ' + str(after_err) if good else tail}")
    rec.update(action="FAILED", error="\n".join(errors)[-1200:])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]} | executable {sys.executable}")
    results = [ensure(p, args.check_only) for p in PACKAGES]

    width = max(len(r["module"]) for r in results) + 2
    print(f"\n{'module':{width}s} {'required':8s} {'status':10s} version / note")
    print("-" * (width + 52))
    for r in results:
        status = "ok" if r["present"] or r["version_after"] else "MISSING"
        note = r["version_after"] or (r["error"] or "")[:60].replace("\n", " ")
        print(f"{r['module']:{width}s} {str(r['required']):8s} {status:10s} {note}")
        if r["action"].startswith(("installed", "FAILED")):
            print(f"{'':{width}s} -> {r['action']}")

    missing_required = [r for r in results if r["required"] and not (
        r["present"] or r["version_after"])]
    missing_optional = [r for r in results if not r["required"] and not (
        r["present"] or r["version_after"])]

    out = {"python": sys.version.split()[0], "check_only": args.check_only,
           "results": results,
           "missing_required": [r["module"] for r in missing_required],
           "missing_optional": [r["module"] for r in missing_optional]}
    path = Path(args.json) if args.json else Path("outputs/logs/env_report.json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {path}")
    except OSError as exc:
        print(f"\ncould not write {path}: {exc}")

    if missing_optional:
        print(f"optional, not installed (features degrade, notebooks say so): "
              f"{[r['module'] for r in missing_optional]}")
    if missing_required:
        print("\n" + "=" * 78)
        print(f"REQUIRED PACKAGES MISSING: {[r['module'] for r in missing_required]}")
        for r in missing_required:
            print(f"\n--- {r['module']} ({r['dist']}) needed for: {r['why']}")
            print(r["error"] or "not installed")
        print("=" * 78)
        return 1
    print("\nENVIRONMENT OK - every required package imports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
