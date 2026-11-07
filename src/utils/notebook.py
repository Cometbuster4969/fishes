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

REPO_SLUG = "Cometbuster4969/fishes"
REPO_URL = f"https://github.com/{REPO_SLUG}.git"
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


def _github_token() -> str | None:
    """Token for a PRIVATE repo: env var first, then a Kaggle secret named GITHUB_TOKEN."""
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN").strip()
    except Exception:  # noqa: BLE001 - no secret configured is a normal case
        return None


def _clone_url(token: str | None) -> str:
    if token:
        return f"https://x-access-token:{token}@github.com/{REPO_SLUG}.git"
    return REPO_URL


def _attached_repo() -> Path | None:
    """The repo uploaded as a Kaggle Dataset appears read-only under /kaggle/input."""
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for pattern in ("*/configs/kaggle.yaml", "*/*/configs/kaggle.yaml"):
        for cand in sorted(root.glob(pattern)):
            return cand.parent.parent
    return None


def ensure_repo(prefer: Path | None = None) -> Path:
    """Find the repo locally, in an attached dataset, or clone it into /kaggle/working."""
    here = Path.cwd()
    for cand in (prefer, here, *here.parents):
        if cand and (cand / "configs" / "kaggle.yaml").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return cand

    attached = _attached_repo()
    if attached is not None:
        if str(attached) not in sys.path:
            sys.path.insert(0, str(attached))
        return attached

    dest = Path("/kaggle/working/fishes")
    if not (dest / "configs" / "kaggle.yaml").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        token = _github_token()
        cmd = ["git", "clone", "--depth", "1", "-b", REPO_BRANCH, _clone_url(token), str(dest)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            hint_private = "" if token else (
                f"\n  The repository {REPO_SLUG} is PRIVATE (or the token is missing).")
            raise RuntimeError(
                "could not clone the GeoFish-AI repository into "
                f"{dest}\n  git said: {(res.stderr or res.stdout).strip()[:300]}{hint_private}"
                "\n  Pick one of these three fixes and re-run this cell:"
                "\n   1) Add-ons -> Secrets -> add a secret named GITHUB_TOKEN holding a GitHub"
                " personal access token with 'repo' scope, then enable it for this notebook."
                "\n   2) Upload the repo as a Kaggle Dataset (it appears read-only under"
                " /kaggle/input) - this works without internet at all."
                "\n   3) Make the repository public (GitHub -> Settings -> Change visibility)."
            )
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
