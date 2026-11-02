"""Pretrained-weight loading. Strict by design.

There is deliberately **no from-scratch path** in this repository. Every builder
here either returns a model whose weights came from a checkpoint, or raises
`PretrainedDownloadError` with the exact reason and the fix.

Why: the project's numbers are only meaningful with transfer learning, and a
silent random-init fallback was the failure mode we most wanted to make
impossible.
"""

from __future__ import annotations

import functools
import time
from typing import Any

from src.config import FromScratchForbiddenError

__all__ = [
    "PretrainedDownloadError",
    "timm_model",
    "torchvision_model",
    "dinov2_encoder",
    "ultralytics_model",
    "open_clip_bioclip",
    "assert_pretrained",
]


class PretrainedDownloadError(RuntimeError):
    """A required pretrained checkpoint could not be obtained."""


_HINT = (
    "\nHow to fix on Kaggle:"
    "\n  1) notebook Settings -> Internet: ON (required for pretrained weights);"
    "\n  2) re-run the setup cell - weights are cached under /kaggle/working/.cache;"
    "\n  3) if the host is blocked by your network, upload the checkpoint as a Kaggle"
    "\n     Dataset and point configs/kaggle.yaml -> pretrained.* at the file."
)

# make sure every cache lives on the writable volume *before* torch/timm import
import os  # noqa: E402


def set_caches(caches: dict[str, str]) -> None:
    for k, v in caches.items():
        os.makedirs(v, exist_ok=True)
        os.environ.setdefault(k, v)


def _fail(what: str, exc: BaseException) -> "PretrainedDownloadError":
    return PretrainedDownloadError(
        f"could not load pretrained weights for {what}: {type(exc).__name__}: {exc}"
        f"\nThis repository refuses to fall back to random initialisation." + _HINT
    )


def _retry(fn, what: str, retries: int = 3, wait: float = 5.0):
    last: BaseException | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - re-raised below with context
            last = exc
            if attempt < retries:
                time.sleep(wait * attempt)
    raise _fail(what, last)  # type: ignore[arg-type]


def assert_pretrained(pretrained: bool | str | None, what: str) -> None:
    """Central guard: refuse anything that means 'random init'."""
    if pretrained is False or pretrained is None or pretrained == "" or pretrained == "none":
        raise FromScratchForbiddenError(
            f"{what} was asked to initialise WITHOUT pretrained weights "
            f"(pretrained={pretrained!r}). From-scratch training is disabled in this "
            f"repository by policy (configs/kaggle.yaml: pretrained.require=true)."
        )


# --------------------------------------------------------------------------- #
# timm / torchvision
# --------------------------------------------------------------------------- #
def timm_model(arch: str, pretrained: str | bool = True, retries: int = 3, **kwargs: Any):
    """Create a timm model with a real checkpoint. Never returns a random-init net."""
    assert_pretrained(pretrained, f"timm:{arch}")
    import timm

    def _build():
        model = timm.create_model(arch, pretrained=pretrained, **kwargs)
        _verify_checkpoint(model, arch)
        return model

    return _retry(_build, f"timm:{arch}", retries=retries)


def torchvision_model(name: str, weights: str = "DEFAULT", retries: int = 3, **kwargs: Any):
    assert_pretrained(weights, f"torchvision:{name}")
    import torchvision.models as tvm

    if not hasattr(tvm, name):
        raise PretrainedDownloadError(f"torchvision has no model '{name}'")

    def _build():
        model = getattr(tvm, name)(weights=weights, **kwargs)
        _verify_checkpoint(model, name)
        return model

    return _retry(_build, f"torchvision:{name}", retries=retries)


def _verify_checkpoint(model, what: str) -> None:
    """Sanity check that weights are not the default random init.

    A freshly initialised ResNet/ViT has near-identical weight magnitudes across
    parameter tensors; a trained checkpoint does not. We only refuse the clearly
    degenerate case (all-zero or all-identical init) so the check never misfires.
    """
    import torch

    tensors = [p.detach().float().flatten() for p in model.parameters() if p.dim() > 1]
    if not tensors:
        return
    sample = tensors[0][:4096]
    if float(sample.std()) == 0.0:
        raise RuntimeError(f"{what}: loaded weights are constant (random/degenerate init)")


def dinov2_encoder(arch: str = "vit_small_patch14_dinov2.lvd142m", image_size: int = 224,
                   retries: int = 3):
    """Frozen DINOv2 feature extractor used by the audit + split builder."""
    model = timm_model(arch, pretrained=True, num_classes=0, img_size=image_size,
                       retries=retries)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Ultralytics (YOLO11 / YOLOv12 / RT-DETR)
# --------------------------------------------------------------------------- #
def ultralytics_model(weights: str, task: str = "detect", retries: int = 3):
    """Load a *checkpoint* (.pt) through Ultralytics.

    Passing a `.yaml` (which would mean random init) is rejected.
    """
    if not weights.endswith(".pt"):
        raise FromScratchForbiddenError(
            f"ultralytics_model('{weights}'): only pretrained .pt checkpoints are allowed. "
            f"Configuring a .yaml here would build the network from scratch, which this "
            f"repository forbids (configs/kaggle.yaml: pretrained.require=true)."
        )
    try:
        import ultralytics
        from packaging.version import Version
        from ultralytics import RTDETR, YOLO
        need = "8.4.0"
        have = getattr(ultralytics, "__version__", "0")
        if Version(have) < Version(need):
            raise PretrainedDownloadError(
                f"ultralytics {have} is too old for this weight set (need >= {need} for "
                f"YOLOv12/RT-DETR checkpoint names). Run: pip install -U ultralytics"
            )
    except ImportError as exc:
        raise _fail("ultralytics", exc)

    def _build():
        cls = RTDETR if weights.startswith("rtdetr") else YOLO
        return cls(weights)

    return _retry(_build, f"ultralytics:{weights}", retries=retries)


# --------------------------------------------------------------------------- #
# BioCLIP (zero-shot cross-check, spec 4.3.3)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=2)
def open_clip_bioclip(tag: str = "hf-hub:imageomics/bioclip", retries: int = 3):
    """Return (model, preprocess, tokenizer) for BioCLIP. Required, never skipped."""
    assert_pretrained(tag, "open_clip:bioclip")
    try:
        import open_clip
    except ImportError as exc:
        raise _fail("open_clip (pip install open-clip-torch)", exc)

    def _build():
        model, _, preprocess = open_clip.create_model_and_transforms(tag)
        tokenizer = open_clip.get_tokenizer(tag)
        model.eval()
        return model, preprocess, tokenizer

    return _retry(_build, f"open_clip:{tag}", retries=retries)
