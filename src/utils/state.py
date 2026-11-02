"""Resume machinery for long Kaggle runs.

A stage is identified by a string key. The tracker records, per stage:
`status` (running|done|failed), `started_at`, `finished_at`, `outputs` (list of
files), `metrics` (dict) and `error`. State lives in
`<outputs>/logs/state.json` on /kaggle/working and is written atomically after
every stage.

Resume model
------------
* Within one interactive session, re-running a cell skips stages already `done`.
* Across sessions, Kaggle wipes /kaggle/working. To continue a previous run you
  attach its output as a dataset and list that mount in
  `configs/kaggle.yaml -> resume.previous_outputs`; the tracker then also looks
  there for state + checkpoints and copies them forward.
* Any stage that wrote a checkpoint records `checkpoint` + `epoch`, so training
  restarts from the last saved epoch instead of epoch 0.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class StageRecord:
    key: str
    status: str = "pending"          # pending | running | done | failed
    started_at: str | None = None
    finished_at: str | None = None
    seconds: float | None = None
    outputs: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint: str | None = None
    epoch: int | None = None
    error: str | None = None
    note: str | None = None

    def done(self) -> bool:
        return self.status == "done"


class StageTracker:
    """JSON-backed, atomic, resume-aware stage bookkeeping."""

    def __init__(self, work_dir: str | Path, previous_outputs: list[str] | None = None,
                 notebook: str = "notebook"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.work_dir / "state.json"
        self.notebook = notebook
        self.previous_outputs = [Path(p) for p in (previous_outputs or [])]
        self.state: dict[str, Any] = {"notebook": notebook, "stages": {}, "history": []}
        self._load()
        if not self.state["stages"]:
            self._import_previous()
        self._t0 = time.time()

    # ------------------------- persistence ------------------------- #
    def _load(self) -> None:
        if self.path.exists():
            try:
                self.state = json.loads(self.path.read_text())
                self.state.setdefault("stages", {})
                self.state.setdefault("history", [])
            except json.JSONDecodeError:
                # corrupt state: keep it aside instead of silently losing it
                bad = self.path.with_suffix(".corrupt.json")
                shutil.move(self.path, bad)
                self.state["history"].append({"event": "state_reset", "saved_to": str(bad)})

    def _import_previous(self) -> None:
        """Resume from a previous Kaggle run attached as an input dataset."""
        for prev in self.previous_outputs:
            cand = prev / "state.json" if prev.is_dir() else prev
            if cand.exists():
                try:
                    prev_state = json.loads(cand.read_text())
                except json.JSONDecodeError:
                    continue
                for key, rec in prev_state.get("stages", {}).items():
                    if rec.get("status") == "done" and key not in self.state["stages"]:
                        self.state["stages"][key] = rec
                self.state["history"].append(
                    {"event": "resumed_from", "path": str(cand), "at": _now()})
                self.save()
                return

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str))
        os.replace(tmp, self.path)

    # ------------------------- queries ------------------------- #
    def get(self, key: str) -> StageRecord:
        rec = self.state["stages"].get(key) or {}
        fields = {k: v for k, v in rec.items() if k in StageRecord.__annotations__}
        fields.setdefault("key", key)
        return StageRecord(**fields)

    def is_done(self, key: str) -> bool:
        return self.get(key).status == "done"

    def summary(self) -> str:
        rows = [f"{'stage':38s} {'status':8s} {'secs':>8s}  metrics"]
        for key, rec in self.state["stages"].items():
            m = rec.get("metrics") or {}
            brief = ", ".join(f"{k}={_fmt(v)}" for k, v in list(m.items())[:4])
            rows.append(f"{key:38s} {rec.get('status', '?'):8s} {rec.get('seconds', 0):8.1f}  {brief}")
        return "\n".join(rows)

    # ------------------------- recording ------------------------- #
    def start(self, key: str, note: str | None = None) -> None:
        self.state["stages"][key] = {**self.get(key).__dict__, "key": key,
                                     "status": "running", "started_at": _now(),
                                     "note": note}
        self.state["history"].append({"event": "start", "stage": key, "at": _now()})
        self.save()

    def finish(self, key: str, outputs: list[str | Path] | None = None,
               metrics: dict[str, Any] | None = None, checkpoint: str | None = None,
               epoch: int | None = None, note: str | None = None) -> None:
        t0 = self.state["stages"].get(key, {}).get("started_at")
        secs = None
        if t0:
            try:
                secs = time.time() - time.mktime(time.strptime(t0, "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                secs = None
        base = self.get(key).__dict__
        base.update({"key": key, "status": "done", "finished_at": _now(), "seconds": secs,
                     "outputs": [str(o) for o in (outputs or [])],
                     "metrics": {**(base.get("metrics") or {}), **(metrics or {})},
                     "checkpoint": checkpoint or base.get("checkpoint"),
                     "epoch": epoch if epoch is not None else base.get("epoch"),
                     "note": note or base.get("note")})
        self.state["stages"][key] = base
        self.state["history"].append({"event": "finish", "stage": key, "at": _now()})
        self.save()

    def fail(self, key: str, error: str) -> None:
        rec = self.get(key).__dict__
        rec.update({"key": key, "status": "failed", "error": error, "finished_at": _now()})
        self.state["stages"][key] = rec
        self.state["history"].append({"event": "fail", "stage": key, "error": error,
                                      "at": _now()})
        self.save()

    def update_progress(self, key: str, epoch: int, checkpoint: str | None = None,
                        metrics: dict[str, Any] | None = None) -> None:
        """Called every epoch so an interrupted run restarts from here."""
        rec = self.state["stages"].setdefault(key, {"key": key, "status": "running"})
        rec["epoch"] = epoch
        if checkpoint:
            rec["checkpoint"] = checkpoint
        if metrics:
            rec.setdefault("metrics", {}).update(metrics)
        rec["updated_at"] = _now()
        self.save()

    def last_checkpoint(self, key: str) -> tuple[str | None, int]:
        rec = self.get(key)
        return rec.checkpoint, int(rec.epoch or 0)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def stage(tracker: StageTracker, key: str, skip_if_done: bool = True,
          note: str | None = None):
    """Context manager: `with stage(tr, "train.yolo11n") as s: ...`

    Yields a small object with `.skip` (True when the stage was already done) and
    `.record(outputs=..., metrics=...)`. Failures are recorded and re-raised.
    """

    class _Ctx:
        def __init__(self) -> None:
            self.skip = bool(skip_if_done and tracker.is_done(key))
            self._outputs: list[str | Path] = []
            self._metrics: dict[str, Any] = {}
            self._checkpoint: str | None = None
            self._epoch: int | None = None

        def out(self, *paths: str | Path) -> None:
            self._outputs.extend(paths)

        def metric(self, **kw: Any) -> None:
            self._metrics.update(kw)

        def checkpoint(self, path: str | Path, epoch: int | None = None) -> None:
            self._checkpoint = str(path)
            self._epoch = epoch

        def __enter__(self):
            if not self.skip:
                tracker.start(key, note=note)
            return self

        def __exit__(self, exc_type, exc, tb):
            if self.skip:
                return False
            if exc_type is None:
                tracker.finish(key, outputs=self._outputs, metrics=self._metrics,
                               checkpoint=self._checkpoint, epoch=self._epoch)
            else:
                tracker.fail(key, f"{exc_type.__name__}: {exc}"[:500])
            return False

    return _Ctx()


def validate_state(work_dir: str | Path) -> dict[str, Any]:
    """Health check used by the final notebook cell."""
    p = Path(work_dir) / "state.json"
    if not p.exists():
        return {"ok": False, "reason": "no state.json - did any stage run?"}
    st = json.loads(p.read_text())
    stages = st.get("stages", {})
    failed = [k for k, v in stages.items() if v.get("status") == "failed"]
    return {"ok": not failed, "stages": len(stages), "failed": failed,
            "done": sum(1 for v in stages.values() if v.get("status") == "done")}
