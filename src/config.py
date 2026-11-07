"""Single entry point for configuration + path resolution (Kaggle profile).

Every notebook does:

    from src.config import load_config
    cfg = load_config()            # -> configs/kaggle.yaml
    cfg.paths.ensure()             # creates /kaggle/working/... trees
    cfg.require_dataset("large_scale_fish")   # raises if "Add Data" was skipped

Design rules:
  * No path is ever hard-coded outside configs/kaggle.yaml.
  * A missing *required* dataset raises DatasetMissingError with the exact Kaggle
    instructions - it never silently skips a phase.
  * Pretrained weights are mandatory: `cfg.require_pretrained()` is called by the
    model builders, and `pretrained=False` is refused outright.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "kaggle.yaml"


class ConfigError(RuntimeError):
    """Base class for configuration problems."""


class DatasetMissingError(ConfigError):
    """A required Kaggle dataset mount is not present."""


class FromScratchForbiddenError(ConfigError):
    """Raised when code tries to initialise a model without pretrained weights."""


def _parse_remap() -> list[tuple[str, str]]:
    raw = os.environ.get("GF_REMAP", "").strip()
    pairs = []
    for chunk in filter(None, (c.strip() for c in raw.split(";"))):
        if "=" not in chunk:
            raise ConfigError(f"GF_REMAP entry {chunk!r} is not OLD=NEW")
        old, new = chunk.split("=", 1)
        pairs.append((old.strip(), new.strip()))
    return pairs


def _remap(p: Path, pairs: list[tuple[str, str]]) -> Path:
    if p.exists():
        return p
    for old, new in pairs:
        if str(p) == old or str(p).startswith(old.rstrip("/") + "/"):
            return Path(new + str(p)[len(old):])
    return p


@dataclass
class Paths:
    input_root: Path
    working_root: Path
    repo_dir: Path
    cache_dir: Path
    data_raw: Path
    data_interim: Path
    data_curated: Path
    data_taxonomy: Path
    data_splits: Path
    outputs: Path
    checkpoints: Path
    metrics: Path
    figures: Path
    tables: Path
    review: Path
    quarantine: Path
    logs: Path
    runs: Path

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> "Paths":
        """Build paths, honouring GF_REMAP="OLD=NEW;OLD2=NEW2" for offline dry-runs.

        GF_REMAP exists so the notebooks can be smoke-tested on a laptop (or in CI)
        without a /kaggle mount. It never changes a path that already exists on
        Kaggle, so a real run is unaffected.
        """
        remap = _parse_remap()
        return cls(**{k: _remap(Path(v), remap) for k, v in d.items()})

    def all_local(self) -> list[Path]:
        """Writable directories (everything except input_root)."""
        return [p for name, p in self.__dict__.items() if name != "input_root"]

    def ensure(self) -> "Paths":
        for p in self.all_local():
            try:
                p.mkdir(parents=True, exist_ok=True)
            except (PermissionError, FileNotFoundError) as exc:
                raise ConfigError(
                    f"cannot create {p}: {exc}\n"
                    f"  This profile writes to /kaggle/working, which only exists inside a "
                    f"Kaggle notebook. Either run the notebook on Kaggle, or set "
                    f"GF_REMAP=\"/kaggle/working=/some/writable/dir\" for a local dry-run."
                ) from exc
        return self


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    paths: Paths
    seed: int = 42

    # ---------------- conveniences ---------------- #
    def section(self, name: str) -> dict[str, Any]:
        if name not in self.raw:
            raise ConfigError(f"configs/{self.path.name} has no section '{name}'")
        return self.raw[name]

    @property
    def is_kaggle(self) -> bool:
        return Path("/kaggle").exists()

    def free_disk_gb(self) -> float:
        """Free space on the volume that will hold the outputs (walks up if absent)."""
        p = self.paths.working_root
        while not p.exists() and p != p.parent:
            p = p.parent
        try:
            return shutil.disk_usage(p).free / 1e9
        except OSError:
            return float("nan")

    # ---------------- datasets ---------------- #
    def dataset(self, key: str) -> dict[str, Any]:
        try:
            return self.raw["datasets"][key]
        except KeyError as exc:
            raise ConfigError(f"unknown dataset key '{key}'") from exc

    def dataset_path(self, key: str) -> Path | None:
        """Mounted path, or None when the (optional) dataset is absent."""
        d = self.dataset(key)
        p = _remap(Path(d["kaggle_input"]), _parse_remap())
        if p.exists():
            return p
        if d.get("required"):
            raise DatasetMissingError(
                f"Required dataset '{key}' not found at {p}.\n"
                f"  In the Kaggle notebook: right sidebar -> Add Input -> search for "
                f"'{key.replace('_', ' ')}' and attach it, or upload the archive as a "
                f"Kaggle Dataset and rename the mount to {p.name}.\n"
                f"  Then re-run this cell (the run resumes where it stopped)."
            )
        return None

    def dataset_report(self) -> list[dict[str, Any]]:
        """Status of every declared dataset, WITHOUT raising.

        Used by the preflight cell and by notebook 01 so a user sees the complete
        picture (which mounts are missing, which are optional) instead of tripping
        over the first required mount and never learning the rest.
        """
        remap = _parse_remap()
        rows = []
        for key, d in self.raw["datasets"].items():
            p = _remap(Path(d["kaggle_input"]), remap)
            if p.exists():
                status = "ok"
            elif d.get("required"):
                status = "MISSING (required)"
            else:
                status = "absent (optional)"
            rows.append({"dataset": key, "task": d.get("task"), "path": str(p),
                         "required": bool(d.get("required")), "status": status})
        return rows

    def missing_required(self) -> list[str]:
        return [r["dataset"] for r in self.dataset_report()
                if r["required"] and r["status"] != "ok"]

    def require_dataset(self, key: str) -> Path:
        p = self.dataset_path(key)
        if p is None:
            raise DatasetMissingError(f"dataset '{key}' is optional and not mounted")
        return p

    def available_datasets(self) -> dict[str, Path]:
        remap = _parse_remap()
        out = {}
        for key, d in self.raw["datasets"].items():
            p = _remap(Path(d["kaggle_input"]), remap)
            if p.exists():
                out[key] = p
        return out

    # ---------------- pretrained policy ---------------- #
    def require_pretrained(self) -> dict[str, Any]:
        pt = self.raw.get("pretrained", {})
        if not pt.get("require", False):
            raise ConfigError(
                "pretrained.require must stay true: this project never trains from scratch"
            )
        if pt.get("on_failure") != "raise":
            raise ConfigError("pretrained.on_failure must be 'raise' (no silent fallbacks)")
        remap = _parse_remap()
        for env, value in (pt.get("caches") or {}).items():
            cache_path = _remap(Path(value), remap)
            try:
                cache_path.mkdir(parents=True, exist_ok=True)
            except (PermissionError, FileNotFoundError) as exc:
                raise ConfigError(
                    f"cannot create the pretrained-weights cache {cache_path}: {exc}\n"
                    f"  Kaggle-only profile: run on Kaggle, or dry-run locally with "
                    f"GF_REMAP=\"/kaggle/working=/some/writable/dir\"."
                ) from exc
            os.environ.setdefault(env, str(cache_path))
        # everybody's cache lives under /kaggle/working/.cache
        os.environ.setdefault("TORCH_HOME", pt["caches"]["TORCH_HOME"])
        os.environ.setdefault("HF_HOME", pt["caches"]["HF_HOME"])
        return pt


def load_config(path: str | Path | None = None) -> Config:
    """Load the Kaggle profile (or $GF_CONFIG / an explicit path)."""
    p = Path(path or os.environ.get("GF_CONFIG", DEFAULT_CONFIG))
    if not p.exists():
        raise ConfigError(
            f"config file not found: {p}\n"
            f"  run notebooks from the repo root, or set GF_CONFIG=<path>"
        )
    data = yaml.safe_load(p.read_text())
    cfg = Config(raw=data, path=p, paths=Paths.from_dict(data["paths"]),
                 seed=int(data.get("seed", 42)))
    return cfg


def describe(cfg: Config) -> str:
    lines = [f"config      : {cfg.path}",
             f"profile     : {cfg.raw.get('profile')}",
             f"seed        : {cfg.seed}",
             f"is_kaggle   : {cfg.is_kaggle}",
             f"working     : {cfg.paths.working_root}  "
             f"(free {cfg.free_disk_gb():.1f} GB)",
             "datasets    :"]
    for key, d in cfg.raw["datasets"].items():
        p = _remap(Path(d["kaggle_input"]), _parse_remap())
        flag = "mounted" if p.exists() else ("MISSING (required)" if d.get("required")
                                             else "absent (optional)")
        lines.append(f"  - {key:22s} {flag:22s} {p}")
    return "\n".join(lines)
