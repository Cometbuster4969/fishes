"""Classifier training loop with two-stage fine-tuning and epoch-level resume.

* Stage 1: backbone frozen, head only.
* Stage 2: full network, layer-wise LR decay.
* Every epoch: checkpoint (`last.pt`), best checkpoint on val macro-F1
  (`best.pt`), TensorBoard scalars, CSV log, and the resume record is updated so
  a killed Kaggle session restarts at the last finished epoch.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.classifier import (ClassBalancedCE, FocalLoss, hierarchical_loss,
                                   mixup_cutmix, mixed_ce)

log = logging.getLogger("geofish")


def optimizer_param_groups(model, lr_backbone: float, lr_head: float,
                           decay: float = 0.75) -> list[dict]:
    """Layer-wise LR decay: earlier blocks get lr_backbone * decay^(n-i)."""
    blocks = []
    for name, mod in model.backbone.named_children():
        blocks.append((name, list(mod.parameters())))
    n = max(len(blocks), 1)
    groups = []
    for i, (_, params) in enumerate(blocks):
        groups.append({"params": params, "lr": lr_backbone * (decay ** (n - 1 - i))})
    # any remaining backbone params (stems/norms not in named_children)
    seen = {id(p) for _, ps in blocks for p in ps}
    rest = [p for p in model.backbone.parameters() if id(p) not in seen]
    if rest:
        groups.append({"params": rest, "lr": lr_backbone})
    head = [p for n_, p in model.named_parameters() if not n_.startswith("backbone.")]
    groups.append({"params": head, "lr": lr_head})
    return groups


def set_backbone_trainable(model, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def _class_counts(dataset) -> np.ndarray:
    """Per-class image counts of the TRAIN split (drives class-balanced weights)."""
    labels = np.asarray(dataset.labels, dtype=np.int64)
    if labels.size == 0:
        raise RuntimeError("training dataset is empty")
    return np.bincount(labels, minlength=int(labels.max()) + 1)


def fit(cfg, model, loaders: dict[str, DataLoader], heads, ckpt_dir: Path,
        run_name: str, tracker=None, stage_key: str | None = None,
        epochs_stage1: int | None = None, epochs_stage2: int | None = None,
        eval_fn: Callable | None = None) -> dict[str, Any]:
    """Train `model`; returns a result dict with history and best metrics."""
    c = cfg.raw["classifier"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = ckpt_dir / f"{run_name}_last.pt"
    best_ckpt = ckpt_dir / f"{run_name}_best.pt"
    csv_path = ckpt_dir / f"{run_name}_history.csv"

    counts = _class_counts(loaders["train"].dataset)
    species_loss = (ClassBalancedCE(counts, label_smoothing=c["label_smoothing"])
                    if c["loss"] == "class_balanced_ce"
                    else FocalLoss(c["focal_gamma"], c["label_smoothing"])).to(device)

    n1 = c["epochs_stage1_head"] if epochs_stage1 is None else epochs_stage1
    n2 = c["epochs_stage2_full"] if epochs_stage2 is None else epochs_stage2
    total_epochs = n1 + n2

    start_epoch, best_f1 = 0, -1.0
    history: list[dict[str, Any]] = []
    amp = bool(cfg.raw.get("amp", True)) and device == "cuda"

    def make_opt(stage: int):
        if stage == 0:
            params = [p for p in model.parameters() if p.requires_grad]
            return torch.optim.AdamW(params, lr=c["lr_head"], weight_decay=c["weight_decay"])
        return torch.optim.AdamW(
            optimizer_param_groups(model, c["lr_backbone"], c["lr_head"],
                                   c["layerwise_lr_decay"]),
            weight_decay=c["weight_decay"])

    # ---------------- resume ---------------- #
    if last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = int(state.get("epoch", -1)) + 1
        best_f1 = float(state.get("best_f1", -1.0))
        history = state.get("history", [])
        log.info("resumed %s at epoch %d (best macro-F1 %.4f)", run_name, start_epoch, best_f1)
        if start_epoch >= total_epochs:
            log.info("%s already finished all %d epochs - nothing to do", run_name, total_epochs)
            return {"run_name": run_name, "resumed_complete": True,
                    "best_macro_f1": best_f1, "checkpoint": str(best_ckpt),
                    "history": history}

    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(log_dir=str(Path(cfg.paths.runs) / run_name))
    except Exception as exc:  # noqa: BLE001
        log.warning("tensorboard unavailable (%s); CSV logging only", exc)

    t0 = time.time()
    for epoch in range(start_epoch, total_epochs):
        stage = 0 if epoch < n1 else 1
        if epoch == 0 or (epoch == n1 and stage == 1):
            if stage == 0:
                set_backbone_trainable(model, False)
                log.info("stage 1: head only (%d epochs)", n1)
            else:
                set_backbone_trainable(model, True)
                log.info("stage 2: full network with layer-wise LR decay (%d epochs)", n2)
            opt = make_opt(stage)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(n2, 1), eta_min=c["lr_backbone"] * 0.01)

        model.train()
        run_loss, n_seen, correct = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, ya, yb, lam = mixup_cutmix(
                x, y, num_classes=len(heads.species),
                mixup_alpha=c["mixup"], cutmix_alpha=c["cutmix"],
                prob=c["mixup_prob"])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=amp):
                out = model(x)
                if lam < 1.0:
                    g_map = torch.as_tensor(heads.genus_of_species, device=device)
                    f_map = torch.as_tensor(heads.family_of_genus, device=device)
                    w = c["hierarchical_loss_weights"]

                    def _species_ce(lg, t):
                        return species_loss(lg, t)

                    def _plain_ce(lg, t):
                        return torch.nn.functional.cross_entropy(lg, t, label_smoothing=0.1)

                    loss = (
                        w["species"] * mixed_ce(_species_ce, out["species"], ya, yb, lam)
                        + w["genus"] * mixed_ce(_plain_ce, out["genus"], g_map[ya], g_map[yb], lam)
                        + w["family"] * mixed_ce(_plain_ce, out["family"],
                                                 f_map[g_map[ya]], f_map[g_map[yb]], lam)
                    )
                else:
                    loss, _ = hierarchical_loss(out, y, heads,
                                                c["hierarchical_loss_weights"], species_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run_loss += float(loss) * x.size(0)
            n_seen += x.size(0)
            correct += int((out["species"].argmax(1) == ya).sum())
        if stage == 1:
            sched.step()

        metrics: dict[str, float] = {"train_loss": run_loss / max(n_seen, 1),
                                     "train_acc": correct / max(n_seen, 1), "stage": stage}
        if eval_fn is not None:
            metrics.update(eval_fn(model, loaders["val"], device))
        history.append({"epoch": epoch, **metrics})
        with open(csv_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            if fh.tell() == 0:
                w.writeheader()
            w.writerow(history[-1])
        if tb:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    tb.add_scalar(k, v, epoch)

        f1 = float(metrics.get("macro_f1", metrics.get("train_acc", 0.0)))
        torch.save({"model": model.state_dict(), "epoch": epoch, "best_f1": max(f1, best_f1),
                    "history": history, "arch": run_name, "heads": heads},
                   last_ckpt)
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics,
                        "arch": run_name, "heads": heads}, best_ckpt)
        if tracker and stage_key:
            tracker.update_progress(stage_key, epoch, str(last_ckpt),
                                    {"epoch": epoch, **{k: v for k, v in metrics.items()
                                                        if isinstance(v, (int, float))}})
        log.info("epoch %2d/%d stage=%d loss=%.4f %s", epoch + 1, total_epochs, stage,
                 metrics["train_loss"],
                 " ".join(f"{k}={v:.4f}" for k, v in metrics.items()
                          if isinstance(v, float) and k != "train_loss"))
        _prune(ckpt_dir, run_name, keep=int(cfg.raw["resume"]["keep_last_k_checkpoints"]))

    if tb:
        tb.close()
    return {"run_name": run_name, "best_macro_f1": best_f1, "checkpoint": str(best_ckpt),
            "last_checkpoint": str(last_ckpt), "history_csv": str(csv_path),
            "epochs": total_epochs, "wall_clock_s": round(time.time() - t0, 1)}


def _prune(ckpt_dir: Path, run_name: str, keep: int = 3) -> None:
    snaps = sorted(ckpt_dir.glob(f"{run_name}_ep*.pt"), key=lambda p: p.stat().st_mtime)
    for p in snaps[:-keep]:
        p.unlink(missing_ok=True)


@torch.no_grad()
def predict_proba(model, loader: DataLoader, device: str = "cuda") -> tuple[np.ndarray, ...]:
    model.eval()
    logits_all, y_all = [], []
    for batch in loader:
        x = batch[0].to(device)
        y = batch[1]
        out = model(x)
        logits_all.append(out["species"].float().cpu().numpy())
        y_all.append(np.asarray(y))
    return np.concatenate(logits_all), np.concatenate(y_all)
