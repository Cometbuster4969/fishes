"""Repository paths, deterministic seeding and small IO helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA = REPO_ROOT / "data"
RAW = DATA / "raw"
ARCHIVES = RAW / "_archives"
INTERIM = DATA / "interim"
CURATED = DATA / "curated"
TAXONOMY = DATA / "taxonomy"
SPLITS = DATA / "splits"

OUT = REPO_ROOT / "outputs"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
MODELS = OUT / "models"
REVIEW = OUT / "review"
QUARANTINE = OUT / "quarantine"
LOGS = OUT / "logs"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
CONFIGS = REPO_ROOT / "configs"

ALL_DIRS = [RAW, ARCHIVES, INTERIM, CURATED, TAXONOMY, SPLITS,
            METRICS, FIGURES, MODELS, REVIEW, QUARANTINE, LOGS, TABLES, REPORTS, CONFIGS]

SEED = 42


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    return path


def write_csv(path: str | Path, rows: Iterable[dict], columns: Sequence[str] | None = None) -> Path:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text(",".join(columns or []) + "\n")
        return path
    cols = list(columns or rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def rel(path: str | Path) -> str:
    """Path relative to the repo root, for logs and reports."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
