"""Tracking + counting (spec section 6 & 7).

One track id == one fish. Species per track = confidence-weighted vote over the
classifier's predictions on that track's crops. Tracks shorter than
`min_track_frames` are dropped before counting (the length is tuned on the
validation clips and reported).

Supports both Ultralytics trackers from configs/kaggle.yaml (ByteTrack primary,
BoT-SORT comparison) and works on a video file or a directory of frames.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("geofish")


def _frames_from(source: str | Path) -> Iterable[np.ndarray]:
    """Yield BGR frames from a video or an image directory."""
    import cv2

    src = Path(source)
    if src.is_dir():
        for p in sorted(src.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                frame = cv2.imread(str(p))
                if frame is not None:
                    yield frame
        return
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {src}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def track_video(model, source: str | Path, cfg, conf: float | None = None,
                iou: float | None = None, tracker: str | None = None,
                max_frames: int | None = None) -> pd.DataFrame:
    """Run the detector+tracker over one clip; return one row per detection."""
    t = cfg.raw["tracking"]
    conf = t["conf"] if conf is None else conf
    iou = t["iou"] if iou is None else iou
    tracker = tracker or t["tracker"]

    rows: list[dict[str, Any]] = []
    for i, frame in enumerate(_frames_from(source)):
        if max_frames and i >= max_frames:
            break
        res = model.track(frame, persist=True, tracker=tracker, conf=conf, iou=iou,
                          verbose=False)
        if not res:
            continue
        r = res[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for b, tid, c in zip(boxes, ids, confs):
            rows.append({"clip": Path(source).name, "frame": i, "track_id": int(tid),
                         "x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]),
                         "y2": float(b[3]), "det_conf": float(c)})
        if i % 100 == 0:
            log.info("%s frame %d: %d tracks", Path(source).name, i, len(ids))
    if not rows:
        raise RuntimeError(f"no tracks produced for {source} - check conf/tracker settings")
    return pd.DataFrame(rows)


def classify_track_crops(df: pd.DataFrame, frame_lookup, classifier, cfg,
                         batch_size: int = 32) -> pd.DataFrame:
    """`frame_lookup(clip, frame_index) -> BGR ndarray`; crops each box and votes."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import torchvision.transforms as T

    from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

    size = int(cfg.raw["classifier"]["input_size"])
    tf = T.Compose([T.Resize(int(size * 1.14)), T.CenterCrop(size), T.ToTensor(),
                    T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = classifier.to(device).eval()

    probs_by_track: dict[int, list[np.ndarray]] = defaultdict(list)
    batch, tids = [], []
    cache: dict[tuple[str, int], Any] = {}

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = classifier(x)["species"]
            p = F.softmax(logits.float(), dim=-1).cpu().numpy()
        for tid, row_p in zip(tids, p):
            probs_by_track[tid].append(row_p)
        batch.clear()
        tids.clear()

    for _, row in df.iterrows():
        key = (row["clip"], int(row["frame"]))
        frame = cache.get(key)
        if frame is None:
            frame = frame_lookup(*key)
            cache = {key: frame} if len(cache) > 8 else {**cache, key: frame}
        h, w = frame.shape[:2]
        box = (max(int(row.x1), 0), max(int(row.y1), 0),
               min(int(row.x2), w), min(int(row.y2), h))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        img = Image.fromarray(frame[:, :, ::-1]).crop(box)
        batch.append(tf(img))
        tids.append(int(row["track_id"]))
        if len(batch) >= batch_size:
            flush()
    flush()

    out = []
    for tid, plist in probs_by_track.items():
        mean_p = np.mean(plist, axis=0)
        out.append({"track_id": tid, "n_frames_scored": len(plist),
                    "species_idx": int(mean_p.argmax()), "species_conf": float(mean_p.max()),
                    "species_probs": mean_p.tolist()})
    return pd.DataFrame(out)


def count_species(tracks: pd.DataFrame, species: pd.DataFrame, class_names: list[str],
                  min_track_frames: int = 10) -> dict[str, Any]:
    """Unique surviving tracks per species per clip."""
    lengths = tracks.groupby(["clip", "track_id"]).size().rename("n_frames").reset_index()
    surviving = lengths[lengths.n_frames >= min_track_frames]
    merged = species.merge(surviving, on="track_id", how="inner")
    merged["species"] = merged["species_idx"].map(dict(enumerate(class_names)))

    per_clip = (merged.groupby(["clip", "species"]).size()
                .rename("count").reset_index())
    summary = {
        "min_track_frames": min_track_frames,
        "n_tracks_raw": int(lengths.track_id.nunique()),
        "n_tracks_surviving": int(merged.track_id.nunique()),
        "by_species": merged.groupby("species").size().to_dict(),
        "per_clip": per_clip.to_dict(orient="records"),
    }
    return {"summary": summary, "counts": per_clip, "tracks": merged}


# --------------------------------------------------------------------------- #
# counting accuracy vs manual ground truth
# --------------------------------------------------------------------------- #
def counting_metrics(pred: pd.DataFrame, truth: pd.DataFrame, key: str = "clip",
                     species_col: str = "species", count_col: str = "count") -> dict:
    """MAE / RMSE / MAPE overall and per species (spec section 7)."""
    m = pred.merge(truth, on=[key, species_col], how="outer",
                   suffixes=("_pred", "_true")).fillna(0)
    err = m[f"{count_col}_pred"] - m[f"{count_col}_true"]
    overall = {
        "n_cells": int(len(m)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": (float((np.abs(err) / m[f"{count_col}_true"].replace(0, np.nan))
                       .abs().mean() * 100)
                 if (m[f"{count_col}_true"] > 0).any() else "not_computable: no non-zero truth"),
    }
    per_species = []
    for sp, g in m.groupby(species_col):
        e = g[f"{count_col}_pred"] - g[f"{count_col}_true"]
        per_species.append({
            "species": sp, "n": int(len(g)),
            "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "mape": (float((np.abs(e) / g[f'{count_col}_true'].replace(0, np.nan)).abs()
                           .mean() * 100) if (g[f"{count_col}_true"] > 0).any() else np.nan),
        })
    return {"overall": overall, "per_species": per_species}


def mot_metrics(tracks: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """MOTA / ID switches / HOTA when tracking ground truth exists.

    `gt` needs columns frame, x1,y1,x2,y2,gt_id (and clip).
    """
    try:
        import motmetrics as mm
    except ImportError:
        return {"error": "not_computable: motmetrics not installed"}
    acc = mm.MOTAccumulator(auto_id=True)
    for frame, g in tracks.groupby("frame"):
        gsub = gt[gt.frame == frame]
        acc.update(
            list(gsub.gt_id.astype(int)),
            list(g.track_id.astype(int)),
            mm.distances.iou_matrix(gsub[["x1", "y1", "x2", "y2"]].values,
                                    g[["x1", "y1", "x2", "y2"]].values, max_iou=0.5),
        )
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=["mota", "id_switches", "num_objects",
                                       "num_misses", "num_false_positives"], name="clip")
    return {k: (float(v.iloc[0]) if hasattr(v, "iloc") else float(v))
            for k, v in summary.items()}
