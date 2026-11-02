"""Torch datasets used by the classifier notebook.

Two entry points:
  * `ManifestDataset`  - images listed in a split manifest (data/splits/*.csv)
  * `CropDataset`      - crops produced by the detector (bounding boxes cut out),
                         used for the "train on detector crops" ablation (spec 6)

Transforms are deterministic given the seed; augmentation (incl. the under-water
simulations from spec 6) is applied to train only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

log = logging.getLogger("geofish")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path | pd.DataFrame, class_to_idx: dict[str, int],
                 transform: Callable | None = None, path_col: str = "path",
                 label_col: str = "dataset_label"):
        self.df = (pd.read_csv(manifest) if isinstance(manifest, (str, Path))
                   else manifest.copy()).reset_index(drop=True)
        missing = [c for c in (path_col, label_col) if c not in self.df.columns]
        if missing:
            raise KeyError(f"manifest is missing columns {missing}")
        self.df = self.df[self.df[label_col].isin(class_to_idx)].reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.labels = [class_to_idx[l] for l in self.df[label_col]]
        self.transform = transform
        self.path_col, self.label_col = path_col, label_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(row[self.path_col]) as im:
            img = im.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, self.labels[i]

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=len(self.class_to_idx))


class CropDataset(ManifestDataset):
    """Same interface; manifest must have x1,y1,x2,y2 columns (detector crops)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for c in ("x1", "y1", "x2", "y2"):
            if c not in self.df.columns:
                raise KeyError(f"crop manifest is missing '{c}'")

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        with Image.open(row["path"]) as im:
            im = im.convert("RGB")
            box = (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
            img = im.crop(box)
            if self.transform:
                img = self.transform(img)
        return img, self.labels[i]


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def build_transforms(image_size: int, train: bool, underwater_sim: bool = True):
    import torchvision.transforms as T

    ops: list[Callable] = []
    if train:
        ops += [T.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
                T.RandomHorizontalFlip(0.5),
                T.ColorJitter(0.25, 0.25, 0.2, 0.03)]
        if underwater_sim:
            # cheap proxies for the pond conditions in spec 6: haze (turbidity),
            # gamma (low light) and slight blur are applied stochastically
            ops += [T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.6))], p=0.25),
                    T.RandomApply([T.Grayscale(3)], p=0.05)]
        ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        if underwater_sim:
            ops += [RandomTurbidity(p=0.2), RandomGamma(p=0.2)]
    else:
        ops += [T.Resize(int(image_size * 1.14)), T.CenterCrop(image_size),
                T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class RandomTurbidity:
    """Blend toward a light-blue haze - a crude but deterministic-seed turbidity sim."""

    def __init__(self, p: float = 0.2, strength: float = 0.35):
        self.p, self.strength = p, strength

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        a = torch.rand(1).item() * self.strength
        haze = torch.tensor([0.62, 0.74, 0.70]).view(3, 1, 1)
        return (1 - a) * x + a * haze


class RandomGamma:
    def __init__(self, p: float = 0.2, lo: float = 0.6, hi: float = 1.6):
        self.p, self.lo, self.hi = p, lo, hi

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        g = self.lo + torch.rand(1).item() * (self.hi - self.lo)
        return torch.clamp(x, 0, 1) ** g


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
def make_loader(ds: Dataset, batch_size: int, train: bool, workers: int = 4,
                balanced: bool = True, seed: int = 42) -> DataLoader:
    sampler = None
    if train and balanced:
        counts = np.bincount(np.asarray(ds.labels), minlength=len(ds.class_to_idx))
        w = 1.0 / np.maximum(counts, 1)
        sample_w = torch.as_tensor([w[l] for l in ds.labels], dtype=torch.double)
        g = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(ds), replacement=True,
                                        generator=g)
    return DataLoader(ds, batch_size=batch_size, shuffle=(train and sampler is None),
                      sampler=sampler, num_workers=workers, pin_memory=True,
                      drop_last=False, persistent_workers=workers > 0)
