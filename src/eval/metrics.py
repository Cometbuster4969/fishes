"""Every metric required by spec section 7 for classification, plus plotting.

Nothing here invents a value: a metric that cannot be computed (e.g. top-5 with
fewer than 5 classes) returns the string "not_computable" with a reason.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

log = logging.getLogger("geofish")

NC = "not_computable"


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray,
                              n_bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def open_set_auroc(known_energy: np.ndarray, ood_energy: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(known_energy)), np.zeros(len(ood_energy))]
    s = np.r_[known_energy, ood_energy]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def classification_metrics(y_true: Sequence[int], probs: np.ndarray,
                           class_names: Sequence[str],
                           family_of_class: Sequence[int] | None = None,
                           family_names: Sequence[str] | None = None,
                           known_energy: np.ndarray | None = None,
                           ood_energy: np.ndarray | None = None) -> dict[str, Any]:
    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score, top_k_accuracy_score)

    y = np.asarray(y_true)
    p = np.asarray(probs)
    pred = p.argmax(axis=1)
    n_classes = len(class_names)
    labels = list(range(n_classes))
    out: dict[str, Any] = {
        "n_samples": int(len(y)),
        "n_classes": int(n_classes),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0,
                                                 labels=labels)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0,
                                           labels=labels)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0, labels=labels)),
        "precision_weighted": float(precision_score(y, pred, average="weighted",
                                                    zero_division=0, labels=labels)),
        "recall_weighted": float(recall_score(y, pred, average="weighted", zero_division=0,
                                              labels=labels)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0,
                                      labels=labels)),
        "ece": expected_calibration_error(y, p),
        "mean_confidence": float(p.max(axis=1).mean()),
    }
    if n_classes >= 5:
        out["top5_accuracy"] = float(top_k_accuracy_score(
            y, p, k=5, labels=labels))
    else:
        out["top5_accuracy"] = f"{NC}: only {n_classes} classes"

    rep = classification_report(y, pred, labels=labels, target_names=list(class_names),
                                output_dict=True, zero_division=0)
    out["per_class"] = {k: v for k, v in rep.items() if k not in ("accuracy", "macro avg",
                                                                  "weighted avg")}
    out["support_total"] = int(len(y))

    cm = confusion_matrix(y, pred, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    out["confusion_matrix"] = cm.tolist()
    out["confusion_matrix_normalized"] = cm_norm.round(6).tolist()

    # top-20 most confused pairs (off-diagonal)
    pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": str(class_names[i]), "pred": str(class_names[j]),
                              "count": int(cm[i, j]),
                              "rate": float(cm[i, j] / max(cm[i].sum(), 1))})
    out["top_confused_pairs"] = sorted(pairs, key=lambda d: -d["count"])[:20]

    if family_of_class is not None and family_names is not None:
        fam = np.asarray(family_of_class)
        y_f, p_f = fam[y], fam[pred]
        cmf = confusion_matrix(y_f, p_f, labels=list(range(len(family_names))))
        out["family_confusion_matrix"] = cmf.tolist()
        out["family_names"] = list(family_names)
        out["family_accuracy"] = float((y_f == p_f).mean())

    if known_energy is not None and ood_energy is not None:
        out["open_set_auroc"] = open_set_auroc(np.asarray(known_energy),
                                               np.asarray(ood_energy))
    else:
        out["open_set_auroc"] = f"{NC}: no OOD/unknown samples evaluated"

    return out


def save_metrics(metrics: dict[str, Any], out_dir: Path, name: str,
                 meta: dict[str, Any] | None = None) -> dict[str, Path]:
    """Write <name>.json + flattened <name>.csv. `meta` documents provenance."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta or {}, "metrics": metrics}
    js = out_dir / f"{name}.json"
    js.write_text(json.dumps(payload, indent=2, default=str))

    flat = {k: v for k, v in metrics.items()
            if isinstance(v, (int, float, str)) and k not in ("per_class",)}
    cs = out_dir / f"{name}.csv"
    pd.DataFrame([flat]).to_csv(cs, index=False)

    if "per_class" in metrics:
        pd.DataFrame(metrics["per_class"]).T.to_csv(
            out_dir / f"{name}_per_class.csv", index=True)
    if "top_confused_pairs" in metrics:
        pd.DataFrame(metrics["top_confused_pairs"]).to_csv(
            out_dir / f"{name}_top_confused.csv", index=False)
    log.info("metrics saved: %s", js)
    return {"json": js, "csv": cs}


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def plot_confusion(cm: np.ndarray, class_names: Sequence[str], out_path: Path,
                   normalize: bool = True, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    m = cm.astype(float)
    if normalize:
        with np.errstate(invalid="ignore"):
            m = m / m.sum(axis=1, keepdims=True)
        m = np.nan_to_num(m)
    n = len(class_names)
    size = max(6.0, min(0.42 * n + 2, 26))
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(m, cmap="Blues", cbar=True, square=True, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                vmin=0, vmax=1 if normalize else None)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or ("confusion matrix (row-normalised)" if normalize
                           else "confusion matrix (counts)"))
    ax.tick_params(labelsize=max(5, 10 - n // 20))
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return Path(out_path)


def plot_per_class_f1(metrics: dict[str, Any], out_path: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per = metrics["per_class"]
    names = [k for k in per if isinstance(per[k], dict)]
    f1 = [per[k]["f1-score"] for k in names]
    order = np.argsort(f1)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.22 * len(names) + 1.5)))
    ax.barh([names[i] for i in order], [f1[i] for i in order], color="#3b6ea5")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title or "per-class F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return Path(out_path)
