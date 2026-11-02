"""Pretrained embedding extraction (DINOv2 / timm) - the backbone of the audit and
of the leakage-safe split.

Weights are **required**; `src.utils.pretrained` raises if they cannot be loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from src.utils.pretrained import dinov2_encoder

log = logging.getLogger("geofish")


@torch.no_grad()
def embed_paths(paths: Sequence[str | Path], arch: str = "vit_small_patch14_dinov2.lvd142m",
                image_size: int = 224, batch_size: int = 32, device: str | None = None,
                l2_normalize: bool = True) -> np.ndarray:
    """Return (N, D) float32 embeddings. Deterministic: eval mode, no augmentation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = dinov2_encoder(arch, image_size=image_size).to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    out: list[np.ndarray] = []
    batch: list[torch.Tensor] = []
    bad: list[str] = []

    def flush() -> None:
        if not batch:
            return
        x = torch.stack(batch).to(device)
        x = (x - mean) / std
        if image_size != 224 and hasattr(model, "set_input_size"):
            pass
        f = model(x)                      # timm with num_classes=0 -> pooled features
        if l2_normalize:
            f = torch.nn.functional.normalize(f, dim=-1)
        out.append(f.float().cpu().numpy())
        batch.clear()

    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
                arr = np.asarray(im, dtype=np.float32) / 255.0
            batch.append(torch.from_numpy(arr).permute(2, 0, 1))
        except Exception as exc:  # noqa: BLE001 - unreadable file is data, not a crash
            bad.append(f"{p}: {type(exc).__name__}")
            batch.append(torch.zeros(3, image_size, image_size))
        if len(batch) >= batch_size:
            flush()
        if (i + 1) % 1000 == 0:
            log.info("embedded %d/%d", i + 1, len(paths))
    flush()
    if bad:
        log.warning("%d unreadable images were embedded as zeros: %s", len(bad), bad[:5])
    if not out:
        raise RuntimeError("no images were embedded")
    return np.concatenate(out, axis=0)


def cosine_duplicate_pairs(emb: np.ndarray, threshold: float = 0.97,
                           block: int = 2048) -> list[tuple[int, int, float]]:
    """All pairs with cosine similarity >= threshold (blocked to bound memory)."""
    pairs: list[tuple[int, int, float]] = []
    n = len(emb)
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        sims = emb[i0:i1] @ emb.T
        for r in range(i1 - i0):
            row = sims[r]
            idx = np.where(row >= threshold)[0]
            for j in idx:
                gi, gj = i0 + r, int(j)
                if gj > gi:
                    pairs.append((gi, gj, float(row[j])))
    return pairs


def union_find(n: int, pairs: Sequence[tuple[int, int, float]]) -> np.ndarray:
    """Cluster ids (0-based, dense) from a list of edges."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    roots = {}
    labels = np.empty(n, dtype=np.int64)
    for k in range(n):
        r = find(k)
        labels[k] = roots.setdefault(r, len(roots))
    return labels
