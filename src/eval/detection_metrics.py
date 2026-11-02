"""Detection metrics (spec section 7) without pycocotools.

Inputs are plain lists of boxes, so the same code scores YOLO11, YOLOv12 and
RT-DETR on the identical split.

Reported per model: precision / recall / F1 at the best-F1 confidence, mAP@50,
mAP@50-95, the PR curve and the F1-confidence curve (saved as CSV + PNG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

log = logging.getLogger("geofish")

NC = "not_computable"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def match_image(pred: np.ndarray, gt: np.ndarray, iou_thr: float) -> tuple[int, int, int]:
    """Greedy one-to-one matching: returns TP, FP, FN for one image."""
    if len(gt) == 0:
        return 0, len(pred), 0
    if len(pred) == 0:
        return 0, 0, len(gt)
    ious = iou_matrix(pred, gt)
    used_gt = set()
    tp = 0
    for i in np.argsort(-ious.max(axis=1)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and j not in used_gt:
            used_gt.add(j)
            tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def _ap_from_pr(prec: np.ndarray, rec: np.ndarray) -> float:
    """101-point interpolated AP (COCO style)."""
    if len(prec) == 0:
        return 0.0
    mrec = np.linspace(0, 1, 101)
    out = []
    for t in mrec:
        m = rec >= t
        out.append(prec[m].max() if m.any() else 0.0)
    return float(np.mean(out))


def evaluate_detection(predictions: Sequence[dict], ground_truth: Sequence[dict],
                       conf_thresholds: np.ndarray | None = None,
                       iou_thresholds: np.ndarray | None = None
                       ) -> dict[str, Any]:
    """`predictions`/`ground_truth`: per-image dicts {boxes: (N,4) xyxy, scores: (N,)}.

    Boxes must be in the same pixel space.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("predictions and ground truth have different image counts")
    n_gt = sum(len(g["boxes"]) for g in ground_truth)
    if n_gt == 0:
        return {"error": f"{NC}: ground truth has no boxes"}

    conf_thresholds = (np.linspace(0.001, 0.99, 100) if conf_thresholds is None
                       else conf_thresholds)
    iou_thresholds = (np.arange(0.5, 0.96, 0.05) if iou_thresholds is None
                      else iou_thresholds)

    # ---- sweep confidence at IoU 0.5 for the P/R/F1 curves ----
    pr_rows = []
    for conf in conf_thresholds:
        tp = fp = fn = 0
        for pred, gt in zip(predictions, ground_truth):
            keep = pred["scores"] >= conf
            t, f, n = match_image(pred["boxes"][keep], gt["boxes"], 0.5)
            tp += t
            fp += f
            fn += n
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        pr_rows.append({"conf": float(conf), "precision": prec, "recall": rec,
                        "f1": f1, "tp": tp, "fp": fp, "fn": fn})
    best = max(pr_rows, key=lambda r: r["f1"])

    # ---- mAP over IoU thresholds, all detections (conf>=0.001) ----
    aps = {}
    for iou in iou_thresholds:
        scores, matches = [], []
        for pred, gt in zip(predictions, ground_truth):
            keep = pred["scores"] >= 0.001
            boxes, sc = pred["boxes"][keep], pred["scores"][keep]
            if len(boxes) == 0:
                continue
            ious = iou_matrix(boxes, gt["boxes"])
            used = set()
            for i in np.argsort(-sc):
                m = 0
                if len(gt["boxes"]):
                    j = int(np.argmax(ious[i]))
                    if ious[i, j] >= iou and j not in used:
                        used.add(j)
                        m = 1
                scores.append(float(sc[i]))
                matches.append(m)
        if not scores:
            aps[f"{iou:.2f}"] = 0.0
            continue
        order = np.argsort(-np.asarray(scores))
        m_arr = np.asarray(matches)[order]
        tp_cum = np.cumsum(m_arr)
        fp_cum = np.cumsum(1 - m_arr)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        rec = tp_cum / n_gt
        aps[f"{iou:.2f}"] = _ap_from_pr(prec, rec)

    map50 = aps.get("0.50", 0.0)
    map5095 = float(np.mean(list(aps.values())))

    return {
        "n_images": len(predictions),
        "n_gt_boxes": int(n_gt),
        "n_pred_boxes_at_best_conf": int(best["tp"] + best["fp"]),
        "best_f1_conf": best["conf"],
        "precision_at_best_f1": best["precision"],
        "recall_at_best_f1": best["recall"],
        "f1_at_best_f1": best["f1"],
        "tp": best["tp"], "fp": best["fp"], "fn": best["fn"],
        "map50": map50,
        "map50_95": map5095,
        "ap_per_iou": aps,
        "pr_curve": pr_rows,
    }


def save_detection_metrics(metrics: dict[str, Any], out_dir: Path, name: str,
                           meta: dict[str, Any] | None = None) -> dict[str, Path]:
    import json
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = metrics.pop("pr_curve", [])
    js = out_dir / f"{name}.json"
    js.write_text(json.dumps({"meta": meta or {}, "metrics": metrics}, indent=2, default=str))
    flat = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))}
    pd.DataFrame([flat]).to_csv(out_dir / f"{name}.csv", index=False)
    if curve:
        pd.DataFrame(curve).to_csv(out_dir / f"{name}_pr_curve.csv", index=False)
    return {"json": js, "curve_csv": out_dir / f"{name}_pr_curve.csv"}


def plot_pr_and_f1(metrics: dict[str, Any] | list[dict], labels: Sequence[str],
                   out_path: Path) -> Path:
    """PR curve and F1-vs-confidence curve for one or several models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = metrics if isinstance(metrics, list) else [metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, label in zip(entries, labels):
        curve = m.get("pr_curve") or []
        if not curve:
            continue
        rec = [r["recall"] for r in curve]
        prec = [r["precision"] for r in curve]
        f1 = [r["f1"] for r in curve]
        conf = [r["conf"] for r in curve]
        ax1.plot(rec, prec, label=f"{label} (mAP50={m.get('map50', float('nan')):.3f})")
        ax2.plot(conf, f1, label=f"{label} (best F1={m.get('f1_at_best_f1', 0):.3f}"
                                f" @ {m.get('best_f1_conf', 0):.2f})")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_title("PR curve (IoU 0.5)")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("F1")
    ax2.set_title("F1 vs confidence")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return Path(out_path)
