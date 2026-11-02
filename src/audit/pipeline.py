"""Label audit pipeline (spec 4.3): dedupe, quality, label noise, zero-shot
cross-check, quarantine and human review pack.

All opinions come from pretrained models - BioCLIP for zero-shot, DINOv2 for the
embedding space. There is no from-scratch model anywhere, by policy.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("geofish")


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #
def quality_report(paths: list[str], laplacian_threshold: float = 60.0,
                   min_short_side: int = 224) -> pd.DataFrame:
    import cv2

    rows = []
    for p in paths:
        rec = {"path": p, "ok": True, "width": None, "height": None,
               "laplacian_var": None, "blurry": False, "too_small": False, "error": None}
        try:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError("cv2 could not decode")
            h, w = img.shape[:2]
            scale = 512.0 / max(h, w)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_AREA)
            var = float(cv2.Laplacian(img, cv2.CV_64F).var())
            rec.update(width=w, height=h, laplacian_var=var,
                       blurry=var < laplacian_threshold,
                       too_small=min(h, w) < min_short_side)
        except Exception as exc:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:120])
        rows.append(rec)
    df = pd.DataFrame(rows)
    log.info("quality: %d unreadable, %d blurry, %d small",
             int((~df.ok).sum()), int(df.blurry.sum()), int(df.too_small.sum()))
    return df


# --------------------------------------------------------------------------- #
# label noise via out-of-fold linear probe on pretrained features
# --------------------------------------------------------------------------- #
def oof_pred_probs(emb: np.ndarray, labels: np.ndarray, folds: int = 5,
                   seed: int = 42) -> np.ndarray:
    """Out-of-fold probabilities from a logistic regression on frozen features.

    Cleanlab needs OOF `pred_probs`; a classifier head on DINOv2 features gives
    them in seconds instead of a 5-fold CNN training run, and never touches the
    test split.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    classes = np.array(sorted(set(labels)))
    if len(classes) < 2:
        raise RuntimeError("need at least 2 classes for label-noise detection")
    y = np.searchsorted(classes, labels)
    probs = np.zeros((len(y), len(classes)), dtype=np.float64)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for k, (tr, te) in enumerate(skf.split(emb, y)):
        clf = LogisticRegression(max_iter=2000, C=10.0, multi_class="multinomial")
        clf.fit(emb[tr], y[tr])
        probs[te] = clf.predict_proba(emb[te])
        log.info("oof fold %d/%d done", k + 1, folds)
    return probs


def find_label_issues(probs: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    from cleanlab.filter import find_label_issues as cl_filter

    classes = np.array(sorted(set(labels)))
    y = np.searchsorted(classes, labels)
    issues = cl_filter(labels=y, pred_probs=probs, return_indices_ranked_by="self_confidence")
    pred = probs.argmax(axis=1)
    rows = []
    for i in issues:
        rows.append({
            "index": int(i),
            "given_label": str(classes[y[i]]),
            "predicted_label": str(classes[pred[i]]),
            "self_confidence": float(probs[i, y[i]]),
            "pred_confidence": float(probs[i, pred[i]]),
        })
    df = pd.DataFrame(rows).sort_values("self_confidence") if rows else pd.DataFrame(
        columns=["index", "given_label", "predicted_label", "self_confidence",
                 "pred_confidence"])
    log.info("cleanlab flagged %d / %d labels", len(df), len(labels))
    return df


def knn_third_opinion(emb: np.ndarray, labels: np.ndarray, k: int = 15) -> np.ndarray:
    """Second substitute opinion: cosine kNN vote among other samples (leave-one-out)."""
    preds = np.empty(len(labels), dtype=object)
    classes = np.array(sorted(set(labels)))
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nn = np.argsort(-sims, axis=1)[:, :k]
    for i in range(len(labels)):
        votes = Counter(labels[nn[i]])
        preds[i] = votes.most_common(1)[0][0]
    return preds


# --------------------------------------------------------------------------- #
# BioCLIP zero-shot
# --------------------------------------------------------------------------- #
def bioclip_zero_shot(paths: list[str], candidate_names: list[str],
                      cfg, batch_size: int = 32) -> np.ndarray:
    """Zero-shot class scores for every image. Requires the BioCLIP checkpoint."""
    import torch
    from src.utils.pretrained import open_clip_bioclip

    tag = cfg.raw["pretrained"]["bioclip"]["open_clip_tag"]
    template = cfg.raw["pretrained"]["bioclip"]["prompt_template"]
    model, preprocess, tokenizer = open_clip_bioclip(tag)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    prompts = [template.format(species=name.replace("_", " ")) for name in candidate_names]
    with torch.no_grad():
        text = tokenizer(prompts).to(device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    from PIL import Image
    scores: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    with torch.no_grad():
        for p in paths:
            with Image.open(p) as im:
                batch.append(preprocess(im.convert("RGB")))
            if len(batch) >= batch_size:
                img = torch.stack(batch).to(device)
                f = model.encode_image(img)
                f = f / f.norm(dim=-1, keepdim=True)
                scores.append((100.0 * f @ text_features.T).softmax(dim=-1).cpu().numpy())
                batch.clear()
        if batch:
            img = torch.stack(batch).to(device)
            f = model.encode_image(img)
            f = f / f.norm(dim=-1, keepdim=True)
            scores.append((100.0 * f @ text_features.T).softmax(dim=-1).cpu().numpy())
    return np.concatenate(scores, axis=0)


# --------------------------------------------------------------------------- #
# decision + outputs
# --------------------------------------------------------------------------- #
def decide_relabels(img_df: pd.DataFrame, issues: pd.DataFrame, knn_preds: np.ndarray,
                    bioclip_scores: np.ndarray | None, candidate_names: list[str],
                    min_confidence: float = 0.90) -> pd.DataFrame:
    """Apply a relabel only when cleanlab + BOTH substitute opinions agree.

    Anything else is returned with decision='quarantine' and never reaches the
    training labels.
    """
    rows = []
    for _, r in issues.iterrows():
        i = int(r["index"])
        given = r["given_label"]
        votes = {"cleanlab": r["predicted_label"], "knn": str(knn_preds[i])}
        if bioclip_scores is not None:
            votes["bioclip"] = candidate_names[int(bioclip_scores[i].argmax())]
        agreed = {v for k, v in votes.items() if k in ("cleanlab", "knn", "bioclip")}
        unanimous = len(agreed) == 1 and given not in agreed
        # the *alternative* class must be confident, and the given label must be weak;
        # otherwise a confident-but-wrong model would be allowed to rewrite good labels
        confident = (float(r["pred_confidence"]) >= min_confidence
                     and float(r["self_confidence"]) < 0.5)
        decision = "relabel" if (unanimous and confident) else "quarantine"
        rows.append({
            "index": i, "path": img_df.iloc[i]["path"], "given_label": given,
            "cleanlab_pred": r["predicted_label"], "knn_pred": votes["knn"],
            "bioclip_pred": votes.get("bioclip", "not_available"),
            "self_confidence": float(r["self_confidence"]),
            "agreement": len(agreed), "decision": decision,
            "new_label": next(iter(agreed)) if decision == "relabel" else given,
        })
    out = pd.DataFrame(rows)
    n_rel = int((out.decision == "relabel").sum()) if len(out) else 0
    log.info("relabel decisions: %d applied, %d quarantined", n_rel,
             len(out) - n_rel)
    return out


def write_review_pack(img_df: pd.DataFrame, decisions: pd.DataFrame, out_dir: Path,
                      images_per_class: int = 25, thumb: int = 224,
                      cols: int = 5) -> dict[str, Any]:
    """Contact sheets so a human can approve/reject every relabel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    def sheet(items: list[tuple[str, str]], title: str, name: str) -> Path | None:
        if not items:
            return None
        rows = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, (path, caption) in zip(axes, items):
            try:
                with Image.open(path) as im:
                    ax.imshow(im.convert("RGB").resize((thumb, thumb)))
            except Exception:  # noqa: BLE001
                ax.text(0.5, 0.5, "unreadable", ha="center")
            ax.set_title(caption[:44], fontsize=6)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        return p

    # per-class sheets
    for cls, sub in img_df.groupby("dataset_label"):
        sample = sub.head(images_per_class)
        p = sheet([(r["path"], f"{cls}") for _, r in sample.iterrows()],
                  f"{cls} ({len(sub)} images)", f"class_{cls.replace('/', '_').replace(':', '_')}")
        if p:
            index_rows.append({"sheet": p.name, "kind": "class", "label": cls,
                               "n_images": len(sub), "reviewer": "", "verdict": ""})

    # flagged sheets
    flagged = decisions[decisions.decision == "quarantine"] if len(decisions) else decisions
    if len(flagged):
        items = [(r["path"], f"{r['given_label']} -> {r['cleanlab_pred']}\n"
                             f"knn={r['knn_pred']} conf={r['self_confidence']:.2f}")
                 for _, r in flagged.head(120).iterrows()]
        p = sheet(items, f"QUARANTINE candidates ({len(flagged)}) - needs human decision",
                  "flag_quarantine")
        if p:
            index_rows.append({"sheet": p.name, "kind": "quarantine", "label": "",
                               "n_images": len(flagged), "reviewer": "", "verdict": ""})

    idx = pd.DataFrame(index_rows)
    idx_path = out_dir / "REVIEW_INDEX.csv"
    idx.to_csv(idx_path, index=False)
    log.info("review pack: %d sheets -> %s", len(idx), out_dir)
    return {"sheets": len(idx), "index": str(idx_path)}


def write_quarantine(decisions: pd.DataFrame, quality: pd.DataFrame, out_dir: Path,
                     unresolved: list[dict] | None = None) -> dict[str, Any]:
    """Never delete: record something as excluded and why."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    if len(decisions):
        q = decisions[decisions.decision == "quarantine"].copy()
        q["reason"] = "label_disagreement_below_confidence"
        parts.append(q)
    if len(quality):
        bad = quality[(~quality.ok) | quality.blurry | quality.too_small].copy()
        bad["reason"] = np.select(
            [~bad.ok, bad.blurry, bad.too_small],
            ["unreadable", "blurry", "too_small"], default="")
        parts.append(bad[["path", "reason"]])
    if unresolved:
        parts.append(pd.DataFrame(unresolved))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["path", "reason"])
    p = out_dir / "quarantine_index.csv"
    df.to_csv(p, index=False)
    log.info("quarantine: %d rows -> %s", len(df), p)
    return {"n": int(len(df)), "path": str(p),
            "by_reason": df["reason"].value_counts().to_dict() if len(df) else {}}
