"""Species classifier: pretrained backbone (required) + hierarchical head.

Head layout (spec section 1): species -> genus -> family, with an extra `unknown`
output used for open-set rejection at inference time.

Initialisation policy: `build_classifier` calls `src.utils.pretrained.timm_model`,
which refuses random init. There is no `pretrained=False` branch anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.pretrained import timm_model

log = logging.getLogger("geofish")


@dataclass
class TaxonomyHeads:
    species: list[str]
    genus_of_species: np.ndarray          # (n_species,) index into genera
    family_of_genus: np.ndarray           # (n_genera,) index into families
    genera: list[str]
    families: list[str]

    @classmethod
    def from_table(cls, species: list[str], genus: list[str], family: list[str]):
        genera = sorted(set(genus))
        families = sorted(set(family))
        gidx = {g: i for i, g in enumerate(genera)}
        fidx = {f: i for i, f in enumerate(families)}
        return cls(species=species,
                   genus_of_species=np.array([gidx[g] for g in genus]),
                   family_of_genus=np.array([fidx[f] for f in family]),
                   genera=genera, families=families)


def heads_from_approved(df, label_col: str = "dataset_label",
                        taxon_col: str = "accepted_name") -> TaxonomyHeads:
    """Build the three-level head from an audit-approved manifest.

    Genus/family come from the accepted taxon when it is a binomial; otherwise the
    dataset label is used as its own genus *and* family, so the hierarchy never
    invents a parent that the audit did not establish.
    """
    import pandas as pd

    if label_col not in df.columns:
        raise KeyError(f"manifest has no column {label_col!r}")
    classes = sorted(df[label_col].unique())
    genus_of, family_of = {}, {}
    for cls in classes:
        rows = df.loc[df[label_col] == cls, taxon_col] if taxon_col in df.columns else None
        name = ""
        if rows is not None and len(rows.dropna()):
            name = str(rows.dropna().iloc[0]).strip()
        parts = name.split()
        genus_of[cls] = parts[0] if len(parts) >= 2 else cls
        family_of[cls] = genus_of[cls]        # family unresolved -> own node
    return TaxonomyHeads.from_table(classes,
                                    [genus_of[c] for c in classes],
                                    [family_of[c] for c in classes])


def family_index_of_species(heads: "TaxonomyHeads") -> list[int]:
    """For every species index, the index of its family (for family-level matrices)."""
    return [int(heads.family_of_genus[g]) for g in heads.genus_of_species]


class HierarchicalClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, heads: TaxonomyHeads,
                 dropout: float = 0.2):
        super().__init__()
        self.backbone = backbone
        self.heads = heads
        self.drop = nn.Dropout(dropout)
        self.fc_species = nn.Linear(feat_dim, len(heads.species))
        self.fc_genus = nn.Linear(feat_dim, len(heads.genera))
        self.fc_family = nn.Linear(feat_dim, len(heads.families))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        f = self.backbone(x)
        if isinstance(f, (tuple, list)):
            f = f[0]
        f = self.drop(f)
        return {"species": self.fc_species(f),
                "genus": self.fc_genus(f),
                "family": self.fc_family(f),
                "features": f}

    @torch.no_grad()
    def energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Energy score for open-set detection (higher = more in-distribution)."""
        return temperature * torch.logsumexp(logits / temperature, dim=-1)


def build_classifier(arch_tag: str, heads: TaxonomyHeads, image_size: int = 288,
                     dropout: float = 0.2) -> HierarchicalClassifier:
    backbone = timm_model(arch_tag, pretrained=True, num_classes=0,
                          img_size=image_size, drop_rate=dropout)
    feat_dim = int(getattr(backbone, "num_features", 0) or 0)
    if feat_dim == 0:
        raise RuntimeError(f"could not determine feature dim for {arch_tag}")
    model = HierarchicalClassifier(backbone, feat_dim, heads, dropout)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log.info("built %s | %.1fM params | %d species / %d genera / %d families",
             arch_tag, n_params, len(heads.species), len(heads.genera), len(heads.families))
    return model


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #
class ClassBalancedCE(nn.Module):
    """Cross-entropy weighted by inverse class frequency (optionally 'effective
    number' smoothing as in Cui et al. 2019)."""

    def __init__(self, counts: np.ndarray, beta: float = 0.9999,
                 label_smoothing: float = 0.1):
        super().__init__()
        counts = np.maximum(np.asarray(counts, dtype=np.float64), 1.0)
        eff = 1.0 - np.power(beta, counts)
        w = (1.0 - beta) / np.maximum(eff, 1e-12)
        w = w / w.sum() * len(w)
        self.register_buffer("weight", torch.tensor(w, dtype=torch.float32))
        self.ls = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, weight=self.weight, label_smoothing=self.ls)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma, self.ls = gamma, label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none", label_smoothing=self.ls)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def hierarchical_loss(out: dict[str, torch.Tensor], y_species: torch.Tensor,
                      heads: TaxonomyHeads, weights: dict[str, float],
                      species_loss: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    g_of_s = torch.as_tensor(heads.genus_of_species, device=y_species.device)
    f_of_g = torch.as_tensor(heads.family_of_genus, device=y_species.device)
    y_genus = g_of_s[y_species]
    y_family = f_of_g[y_genus]

    l_sp = species_loss(out["species"], y_species)
    l_ge = F.cross_entropy(out["genus"], y_genus, label_smoothing=0.1)
    l_fa = F.cross_entropy(out["family"], y_family, label_smoothing=0.1)
    total = (weights.get("species", 1.0) * l_sp + weights.get("genus", 0.3) * l_ge
             + weights.get("family", 0.15) * l_fa)
    return total, {"loss_species": float(l_sp), "loss_genus": float(l_ge),
                   "loss_family": float(l_fa)}


# --------------------------------------------------------------------------- #
# mixup / cutmix
# --------------------------------------------------------------------------- #
def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, num_classes: int,
                 mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0,
                 prob: float = 0.5, seed: int | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Returns (x_mixed, y_a, y_b, lam); lam=1 means 'no mixing'."""
    if torch.rand(1, generator=None).item() > prob or (mixup_alpha <= 0 and cutmix_alpha <= 0):
        return x, y, y, 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if cutmix_alpha > 0 and torch.rand(1).item() < 0.5:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        h, w = x.shape[-2:]
        rh, rw = int(h * np.sqrt(1 - lam)), int(w * np.sqrt(1 - lam))
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, h)
        x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, w)
        x[:, :, y1:y2, x1:x2] = x[perm][:, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1 - lam) * x[perm]
    return x, y, y[perm], lam


def mixed_ce(loss_fn: nn.Module, logits: torch.Tensor, ya: torch.Tensor,
             yb: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * loss_fn(logits, ya) + (1 - lam) * loss_fn(logits, yb)


# --------------------------------------------------------------------------- #
# open-set thresholding (tuned on validation only)
# --------------------------------------------------------------------------- #
def tune_open_set_threshold(energy_in: np.ndarray, energy_ood: np.ndarray,
                            target_tpr: float = 0.95) -> dict[str, Any]:
    """Pick the in-distribution threshold that keeps `target_tpr` of knowns."""
    if len(energy_in) == 0:
        raise ValueError("no in-distribution energies")
    thr = float(np.quantile(energy_in, 1.0 - target_tpr))
    tpr = float((energy_in >= thr).mean())
    tnr = float((energy_ood < thr).mean()) if len(energy_ood) else float("nan")
    return {"threshold": thr, "target_tpr": target_tpr, "val_tpr": tpr,
            "val_tnr": tnr, "n_val_in": int(len(energy_in)),
            "n_val_ood": int(len(energy_ood))}
