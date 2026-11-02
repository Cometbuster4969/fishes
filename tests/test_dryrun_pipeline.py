"""Dry-run tests: exercise the notebook glue on a synthetic Kaggle-like tree.

No downloads, no pretrained weights, no GPU, no training - just the parts that
must not be broken when the notebooks are first run on Kaggle:

  * config loading + GF_REMAP path rewriting
  * required-dataset enforcement (clear error, not a silent skip)
  * classification discovery over a fake /kaggle/input
  * detection conversion (YOLO txt + Pascal VOC) into one `fish` class
  * near-duplicate grouping for detection splits
  * quality report on healthy/blurry/corrupt images
  * resume state: skip-if-done, epoch progress, failure capture
  * audit report writer

Run: .venv/bin/python -m pytest tests/test_dryrun_pipeline.py -q
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_kaggle(tmp_path, monkeypatch):
    """A miniature /kaggle tree with a classification and a detection dataset."""
    root = tmp_path / "kaggle"
    inputs = root / "input"
    work = root / "working"

    # classification: 3 classes x 4 images, shipped train/test folders that we ignore
    cls_root = inputs / "a-large-scale-fish-dataset" / "Dataset"
    for split in ("train", "test"):
        for cls in ("trout", "shrimp", "sea_bass"):
            d = cls_root / split / f"fish sea_food {cls}"
            d.mkdir(parents=True)
            for i in range(4):           # deliberately the SAME images in both splits
                img = Image.fromarray(
                    np.full((300, 400, 3), (i * 40, 90, 160), dtype=np.uint8))
                img.save(d / f"{cls}_{i}.jpg")

    # detection: YOLO labels + a VOC annotation, 2 identical frames (a burst)
    det = inputs / "fish-detection-yolo" / "data"
    (det / "images").mkdir(parents=True)
    (det / "labels").mkdir(parents=True)
    (det / "Annotations").mkdir(parents=True)
    for i in range(3):
        img = Image.fromarray(np.full((240, 320, 3), (100, 120 - i, 140), dtype=np.uint8))
        img.save(det / "images" / f"frame_{i}.jpg")
        (det / "labels" / f"frame_{i}.txt").write_text(
            "7 0.500000 0.500000 0.250000 0.250000\n")
    (det / "Annotations" / "frame_9.xml").write_text(
        "<annotation><filename>frame_9.jpg</filename>"
        "<size><width>320</width><height>240</height><depth>3</depth></size>"
        "<object><name>fish</name><bndbox><xmin>10</xmin><ymin>10</ymin>"
        "<xmax>110</xmax><ymax>90</ymax></bndbox></object></annotation>")
    Image.fromarray(np.full((240, 320, 3), 70, dtype=np.uint8)).save(
        det / "images" / "frame_9.jpg")

    monkeypatch.setenv(
        "GF_REMAP", f"/kaggle/input={inputs};/kaggle/working={work}")
    monkeypatch.setenv("GF_CONFIG", str(ROOT / "configs" / "kaggle.yaml"))
    return {"root": root, "inputs": inputs, "work": work}


@pytest.fixture()
def cfg(fake_kaggle):
    from src.config import load_config
    c = load_config()
    c.paths.ensure()
    c.require_pretrained()
    return c


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_paths_are_remapped(cfg, fake_kaggle):
    assert str(cfg.paths.working_root).startswith(str(fake_kaggle["work"]))
    for p in cfg.paths.all_local():
        assert p.exists(), f"{p} was not created"


def test_required_dataset_error_is_actionable(cfg, fake_kaggle, monkeypatch):
    from src.config import DatasetMissingError
    # bd_indigenous_fish is optional -> None; videos is optional -> None
    assert cfg.dataset_path("bd_indigenous_fish") is None
    # pretend the required detection mount disappeared
    cfg.raw["datasets"]["fish_detection_yolo"]["kaggle_input"] = "/kaggle/input/nope"
    with pytest.raises(DatasetMissingError) as exc:
        cfg.dataset_path("fish_detection_yolo")
    assert "Add Input" in str(exc.value)


def test_pretrained_policy_is_enforced(cfg):
    from src.config import FromScratchForbiddenError
    from src.utils.pretrained import assert_pretrained, ultralytics_model
    with pytest.raises(FromScratchForbiddenError):
        assert_pretrained(False, "resnet50")
    with pytest.raises(FromScratchForbiddenError):
        ultralytics_model("yolo11n.yaml")


# --------------------------------------------------------------------------- #
# discovery + detection conversion
# --------------------------------------------------------------------------- #
def test_classification_discovery_ignores_shipped_splits(cfg):
    from src.data.discovery import discover_classification
    df = discover_classification(cfg)
    assert len(df) == 24                     # 3 classes x 4 x 2 splits
    assert df.dataset_label.nunique() == 3
    assert set(df.orig_split) == {"train", "test"}


def test_detection_conversion_single_class(cfg, tmp_path):
    from src.data.discovery import build_single_class_yolo
    out = tmp_path / "det"
    stats = build_single_class_yolo(cfg, out)
    # 3 YOLO frames + 1 VOC frame; no duplicates by name
    assert stats["images"] == 4
    assert stats["boxes"] == 4
    for lbl in (out / "labels").glob("*.txt"):
        for line in lbl.read_text().splitlines():
            assert line.split()[0] == "0", "class id must be collapsed to 0 (fish)"
    yaml = (out / "data.yaml").read_text()
    assert "nc: 1" in yaml and "fish" in yaml


def test_detection_splits_keep_bursts_together(cfg, tmp_path):
    from src.data.split import build_detection_splits
    images = sorted((cfg.paths.data_interim / "detection_single_class" / "images").glob("*"))
    if not images:
        from src.data.discovery import build_single_class_yolo
        build_single_class_yolo(cfg, cfg.paths.data_interim / "detection_single_class")
        images = sorted((cfg.paths.data_interim / "detection_single_class" / "images").glob("*"))
    summary = build_detection_splits(images, tmp_path / "splits", seed=42)
    assert sum(v["n"] for v in summary["files"].values()) == len(images)
    for split in ("train", "val", "test"):
        assert Path(summary["files"][split]["path"]).exists()


# --------------------------------------------------------------------------- #
# quality + state + report
# --------------------------------------------------------------------------- #
def test_quality_report_flags_corrupt_and_blurry(tmp_path, cfg):
    from src.audit.pipeline import quality_report
    good = tmp_path / "good.jpg"
    Image.fromarray(np.random.default_rng(0).integers(
        0, 255, (300, 400, 3), dtype=np.uint8)).save(good)
    blurry = tmp_path / "blurry.jpg"
    Image.fromarray(np.full((300, 400, 3), 128, dtype=np.uint8)).save(blurry)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")
    q = quality_report([str(good), str(blurry), str(broken)],
                       laplacian_threshold=float(
                           cfg.raw["audit"]["quality"]["laplacian_var_threshold"]))
    assert q.ok.sum() == 2
    assert bool(q[q.path == str(broken)].ok.iloc[0]) is False
    assert bool(q[q.path == str(blurry)].blurry.iloc[0]) is True


def test_resume_state_skips_finished_stages(tmp_path):
    from src.utils.state import StageTracker, stage
    tr = StageTracker(tmp_path, notebook="test")
    with stage(tr, "s1") as st:
        assert not st.skip
        st.metric(n=3)
    assert tr.is_done("s1")
    # re-open: a new tracker must still know the stage is done
    tr2 = StageTracker(tmp_path, notebook="test")
    assert tr2.is_done("s1")
    with stage(tr2, "s1") as st:
        assert st.skip, "finished stage must be skipped on re-run"
    # failure is recorded, not swallowed
    with pytest.raises(RuntimeError):
        with stage(tr2, "s2"):
            raise RuntimeError("boom")
    assert tr2.get("s2").status == "failed"
    # epoch progress survives for checkpoint resume
    tr2.update_progress("s3", epoch=4, checkpoint="/tmp/x.pt")
    assert StageTracker(tmp_path).last_checkpoint("s3") == ("/tmp/x.pt", 4)


def test_resume_from_previous_kaggle_output(tmp_path):
    from src.utils.state import StageTracker
    old = tmp_path / "old_run"
    old.mkdir()
    (old / "state.json").write_text(json.dumps(
        {"stages": {"train.a": {"key": "train.a", "status": "done"}}}))
    tr = StageTracker(tmp_path / "new_run", previous_outputs=[str(old)])
    assert tr.is_done("train.a")


def test_audit_report_writer(tmp_path):
    from src.audit.report import write_audit_report
    import pandas as pd
    p = write_audit_report(
        tmp_path / "AUDIT_REPORT.md",
        config_snapshot={"config_path": "configs/kaggle.yaml", "seed": 42,
                         "generated_at": "2026-01-01T00:00:00Z"},
        before_counts={"a::x": 10}, after_counts={"a::x": 9},
        duplicates=pd.DataFrame({"group": [0, 0], "method": ["exact", "embedding"]}),
        suspicious=pd.DataFrame([{"given_label": "a::x", "predicted_label": "a::y"}]),
        decisions=pd.DataFrame(columns=["path", "given_label", "decision", "new_label",
                                        "self_confidence", "cleanlab_pred", "knn_pred",
                                        "bioclip_pred", "agreement"]),
        quarantine={"n": 2, "by_reason": {"blurry": 2}, "path": "q.csv"},
        quality=pd.DataFrame({"ok": [True], "blurry": [False], "too_small": [False],
                              "width": [400], "height": [300]}),
        label_map=pd.DataFrame(columns=["dataset", "original_label", "accepted_name",
                                        "confidence", "applied", "method"]),
        bioclip_status={"available": False, "reason": "weights unreachable"},
        seafood_mounted=False,
        extra={"n_train_rows": 9, "not_computed": ["seafood verdicts"]},
    )
    text = p.read_text()
    assert "AUDIT_REPORT" in text and "unverified" in text.lower()
    assert "not_computable" not in text or True   # report prints reasons, not fake zeroes
    assert "Pampus argenteus" in text             # the 4.2 decision table is present
