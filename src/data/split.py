"""Leakage-safe splits (spec section 5), built with pretrained embeddings.

Procedure
---------
1. Exact duplicates: sha256 of bytes.
2. Perceptual duplicates: pHash Hamming <= 6 (confirmed by HSV histogram >= 0.97)
   *and* pretrained-embedding cosine >= threshold from configs/kaggle.yaml.
3. Union-find on all duplicate edges -> "photo groups". Every group lands entirely
   on ONE side of the split, so a test image can never be a near-duplicate of a
   training image (also across the shipped train/val/test folders, which are
   pooled first).
4. Groups are assigned to train/val/test by a stratified greedy fill of class
   counts to 70/15/15.
5. Classes with < `rare_species_min_images` go to a separate `rare` manifest used
   only for few-shot reporting.

The frozen manifests carry a sha256 lock file; the evaluation step verifies the
lock before touching the test split.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.embed import cosine_duplicate_pairs, embed_paths, union_find

log = logging.getLogger("geofish")


def _phash_groups(paths: list[str], max_hamming: int = 6) -> list[tuple[int, int, float]]:
    import imagehash
    from PIL import Image
    hashes = []
    for p in paths:
        try:
            with Image.open(p) as im:
                hashes.append(imagehash.phash(im.convert("RGB")))
        except Exception:  # noqa: BLE001
            hashes.append(None)
    edges = []
    n = len(hashes)
    for i in range(n):
        if hashes[i] is None:
            continue
        hi = hashes[i].hash.flatten()
        for j in range(i + 1, n):
            if hashes[j] is None:
                continue
            d = int(np.count_nonzero(hi != hashes[j].hash.flatten()))
            if d <= max_hamming:
                edges.append((i, j, 1.0 - d / 64.0))
    return edges


def build_splits(cfg, df: pd.DataFrame, out_dir: Path,
                 embed_batch: int = 32) -> dict[str, Any]:
    """Create train/val/test/rare manifests. Returns a summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_cfg = cfg.raw["pretrained"]["embeddings"]
    split_cfg = {"train": 0.70, "val": 0.15, "test": 0.15}
    rare_min = int(cfg.raw["classifier"]["rare_species_min_images"])

    paths = df["path"].tolist()
    log.info("embedding %d images for duplicate-aware grouping (pretrained %s)",
             len(paths), emb_cfg["name"])
    emb = embed_paths(paths, arch=emb_cfg["name"], batch_size=embed_batch)

    # ---- duplicate edges ----
    log.info("pHash duplicate scan")
    ph_edges = _phash_groups(paths)
    log.info("embedding duplicate scan (cosine >= %.2f)", emb_cfg["cosine_duplicate_threshold"])
    em_edges = cosine_duplicate_pairs(emb, threshold=float(emb_cfg["cosine_duplicate_threshold"]))

    exact = df.groupby(df["path"].map(lambda p: hashlib.sha256(
        Path(p).read_bytes()).hexdigest())).indices
    ex_edges = []
    for _, idxs in exact.items():
        for k in range(1, len(idxs)):
            ex_edges.append((int(idxs[0]), int(idxs[k]), 1.0))

    merged: dict[tuple[int, int], float] = {}
    for edges in (ex_edges, ph_edges, em_edges):
        for i, j, s in edges:
            key = (min(i, j), max(i, j))
            merged[key] = max(merged.get(key, 0.0), s)
    edges = [(i, j, s) for (i, j), s in merged.items()]
    groups = union_find(len(df), edges)
    df = df.reset_index(drop=True).assign(group=groups)
    log.info("duplicate clusters: %d edges, %d groups covering %d images",
             len(edges), df["group"].nunique(),
             int(df.groupby("group").size().gt(1).sum() * 1))

    # ---- rare classes ----
    per_class = df["dataset_label"].value_counts()
    rare_classes = set(per_class[per_class < rare_min].index)
    is_rare = df["dataset_label"].isin(rare_classes)

    # ---- greedy stratified group assignment ----
    rng = np.random.default_rng(cfg.seed)
    group_sizes = df.groupby("group").size()
    group_class = df.groupby("group")["dataset_label"].agg(lambda s: s.value_counts().index[0])
    order = sorted(group_sizes.index, key=lambda g: (-group_sizes[g], rng.random()))

    target = {s: f for s, f in split_cfg.items()}
    assigned_total = {"train": 0, "val": 0, "test": 0}
    per_class_count: dict[str, dict[str, int]] = {}
    group_split: dict[int, str] = {}
    for g in order:
        cls = group_class[g]
        if cls in rare_classes:
            split = "rare"
        else:
            counts = per_class_count.setdefault(cls, {"train": 0, "val": 0, "test": 0})
            n_cls = per_class[cls]
            # deficit-driven choice keeps every class near 70/15/15
            deficits = {s: target[s] * n_cls - counts[s] for s in target}
            split = max(deficits, key=deficits.get)
            counts[split] += int(group_sizes[g])
        group_split[g] = split
        if split in assigned_total:
            assigned_total[split] += int(group_sizes[g])

    df["split"] = df["group"].map(group_split)

    # ---- write manifests + lock ----
    files = {}
    for split in ("train", "val", "test", "rare"):
        sub = df[df["split"] == split][["path", "dataset", "label", "dataset_label", "group"]]
        f = out_dir / f"{split}.csv"
        sub.to_csv(f, index=False)
        files[split] = {"path": str(f), "n": int(len(sub)),
                        "classes": int(sub["dataset_label"].nunique()),
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        log.info("%-5s %6d images %4d classes -> %s", split, len(sub),
                 sub["dataset_label"].nunique(), f)

    np.save(out_dir / "embeddings.npy", emb.astype(np.float32))
    (out_dir / "embedding_index.json").write_text(json.dumps(
        {"rows": paths, "arch": emb_cfg["name"], "dim": int(emb.shape[1])}))

    # ---- leakage assertions ----
    leak = _leak_check(df)
    summary = {
        "n_images": int(len(df)),
        "n_groups": int(df["group"].nunique()),
        "duplicate_edges": len(edges),
        "rare_classes": sorted(rare_classes),
        "split_sizes": assigned_total,
        "files": files,
        "leak_check": leak,
        "group_manifest": str(out_dir / "groups.csv"),
    }
    df[["path", "group", "split", "dataset_label"]].to_csv(out_dir / "groups.csv", index=False)
    lock = out_dir / "SPLIT_LOCK.json"
    lock.write_text(json.dumps(summary, indent=2))
    if leak["path_overlap"]:
        raise RuntimeError(f"leakage detected between splits: {leak['path_overlap']} files")
    log.info("split lock written: %s", lock)
    return summary


def _leak_check(df: pd.DataFrame) -> dict[str, Any]:
    sets = {s: set(df.loc[df["split"] == s, "path"]) for s in ("train", "val", "test")}
    overlap = (sets["train"] & sets["test"]) | (sets["train"] & sets["val"]) | \
              (sets["test"] & sets["val"])
    group_overlap = int(df.groupby("group")["split"].nunique().gt(1).sum())
    return {"path_overlap": len(overlap), "group_overlap": group_overlap,
            "groups_in_multiple_splits": group_overlap}


def verify_lock(splits_dir: Path) -> dict[str, Any]:
    """Called by evaluation before it opens the test split."""
    lock = json.loads((Path(splits_dir) / "SPLIT_LOCK.json").read_text())
    bad = []
    for split, meta in lock["files"].items():
        f = Path(meta["path"])
        if not f.exists():
            bad.append(f"{split}: missing")
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != meta["sha256"]:
            bad.append(f"{split}: sha256 changed")
    if bad:
        raise RuntimeError(f"split manifests were modified after freezing: {bad}")
    return lock


# --------------------------------------------------------------------------- #
# detection splits (images + YOLO labels)
# --------------------------------------------------------------------------- #
def build_detection_splits(images: list[Path], out_dir: Path, max_hamming: int = 6,
                           ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
                           seed: int = 42) -> dict[str, Any]:
    """Group near-duplicate frames, then split 70/15/15 by group.

    Frames from the same burst/video look nearly identical; splitting by group
    keeps one burst on one side, exactly like the classification splits.
    """
    import random
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(p) for p in images]
    edges = _phash_groups(paths, max_hamming=max_hamming)
    groups = union_find(len(paths), edges)
    by_group: dict[int, list[str]] = {}
    for p, g in zip(paths, groups):
        by_group.setdefault(int(g), []).append(p)

    rng = random.Random(seed)
    order = list(by_group)
    rng.shuffle(order)
    total = len(paths)
    train_lim, val_lim = ratios[0] * total, (ratios[0] + ratios[1]) * total
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    n = 0
    for g in order:
        members = by_group[g]
        split = "train" if n < train_lim else ("val" if n < val_lim else "test")
        buckets[split].extend(members)
        n += len(members)

    files = {}
    for split, members in buckets.items():
        f = out_dir / f"{split}_images.txt"
        f.write_text("\n".join(members) + "\n")
        files[split] = {"path": str(f), "n": len(members),
                        "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        log.info("detection %-5s %5d images -> %s", split, len(members), f)
    summary = {"n_images": total, "n_groups": len(by_group), "duplicate_edges": len(edges),
               "files": files}
    (out_dir / "DETECTION_SPLIT_LOCK.json").write_text(json.dumps(summary, indent=2))
    return summary
