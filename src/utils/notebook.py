"""Notebook bootstrap: one call at the top of every notebook.

    from src.utils.notebook import bootstrap
    env = bootstrap("02_taxonomy_audit")
    cfg, tr = env.cfg, env.tracker

It does, in order:
 1. locates the repo (clone or attached source dataset) and puts it on sys.path,
 2. loads configs/kaggle.yaml and creates every writable directory,
 3. seeds everything (42) and configures verbose logging to file + stdout,
 4. opens the resume state for this notebook,
 5. prints a mount report: which datasets are attached, which are missing, GPU,
    free disk - so a broken run fails in cell 1 instead of cell 40.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/Cometbuster4969/fishes.git"
REPO_BRANCH = "arena/3dc076ac-fishes"


@dataclass
class Env:
    cfg: Any
    tracker: Any
    log: logging.Logger
    repo: Path


# --------------------------------------------------------------------------- #
def _on_kaggle() -> bool:
    return Path("/kaggle").exists()


def ensure_repo(prefer: Path | None = None) -> Path:
    """Find the repo, or clone it into /kaggle/working.

    Notebooks are opened from the repo, so this is a no-op there; it matters when
    the .ipynb is uploaded to Kaggle on its own.
    """
    here = Path.cwd()
    for cand in (prefer, here, *here.parents):
        if cand and (cand / "configs" / "kaggle.yaml").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand

    # attached repo as an input dataset?
    for cand in sorted(Path("/kaggle/input").glob("*/configs/kaggle.yaml")) if Path(
            "/kaggle/input").exists() else []:
        repo = cand.parent.parent
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        return repo

    dest = Path("/kaggle/working/fishes")
    if not (dest / "configs" / "kaggle.yaml").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "-b", REPO_BRANCH, REPO_URL,
                        str(dest)], check=True)
    if str(dest) not in sys.path:
        sys.path.insert(0, str(dest))
    return dest


def _setup_logging(log_dir: Path, notebook: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("geofish")
    log.setLevel(logging.INFO)
    if log.handlers:
        return log
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{notebook}.log", mode="a")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    log.propagate = False
    return log


def gpu_report() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not importable"}
    if not torch.cuda.is_available():
        return {"available": False, "reason": "no CUDA device"}
    props = torch.cuda.get_device_properties(0)
    return {"available": True, "name": props.name,
            "vram_gb": round(props.total_memory / 1e9, 1),
            "count": torch.cuda.device_count(),
            "capability": f"{props.major}.{props.minor}"}


# --------------------------------------------------------------------------- #
def bootstrap(notebook: str, require_gpu: bool = True, repo: Path | None = None) -> Env:
    repo = ensure_repo(repo)
    from src.config import load_config
    from src.utils.paths import set_seed
    from src.utils.state import StageTracker

    cfg = load_config()
    cfg.paths.ensure()
    set_seed(cfg.seed)
    log = _setup_logging(cfg.paths.logs, notebook)

    pt = cfg.require_pretrained()          # cache dirs + policy assertions
    log.info("=" * 78)
    log.info("notebook %s | config %s | seed %d", notebook, cfg.path.name, cfg.seed)
    log.info("pretrained policy: require=%s on_failure=%s", pt["require"], pt["on_failure"])

    gpu = gpu_report()
    if require_gpu and not gpu["available"]:
        raise RuntimeError(
            f"{notebook} needs a GPU. Kaggle -> Settings -> Accelerator -> GPU T4 x2, "
            f"then Settings -> Persistence: 'Files only'. Details: {gpu}"
        )
    log.info("gpu: %s", gpu)
    log.info("free disk: %.1f GB", cfg.free_disk_gb())

    mounted = cfg.available_datasets()
    log.info("datasets mounted (%d): %s", len(mounted), ", ".join(mounted) or "none")
    for key, d in cfg.raw["datasets"].items():
        if key not in mounted:
            log.info("  %-22s %s", key, "MISSING (required)" if d.get("required")
                     else "absent (optional)")

    resume_cfg = cfg.raw.get("resume", {})
    prev = resume_cfg.get("previous_outputs") or []
    tracker = StageTracker(cfg.paths.logs, previous_outputs=prev, notebook=notebook)
    log.info("resume: %d stages already done", sum(
        1 for s in tracker.state["stages"].values() if s.get("status") == "done"))
    return Env(cfg=cfg, tracker=tracker, log=log, repo=repo)


def progress(env: Env, message: str, current: int | None = None,
             total: int | None = None) -> None:
    """Uniform progress logging (works in a Kaggle cell and in a plain terminal)."""
    if current is None or total is None:
        env.log.info(message)
    else:
        env.log.info("[%d/%d] %s", current, total, message)
