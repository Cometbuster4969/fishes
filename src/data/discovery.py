"""Dataset discovery: turn Kaggle mounts into one uniform image manifest.

Layouts are declared in configs/kaggle.yaml (`datasets.<key>.layout`) and parsed
here. Nothing is guessed silently: an unparseable mount raises with the expected
layout printed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
log = logging.getLogger("geofish")


class LayoutError(RuntimeError):
    """A mounted dataset does not match its declared layout."""


def _iter_images(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            yield p


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def _label_from_path(path: Path, root: Path) -> str:
    """Deepest directory under `root` that is not a split folder."""
    parts = path.relative_to(root).parts[:-1]
    split_names = {"train", "val", "valid", "validation", "test", "testing", "images"}
    while parts and parts[-1].lower() in split_names:
        parts = parts[:-1]
    return parts[-1] if parts else "(root)"


def discover_classification(cfg) -> pd.DataFrame:
    """Every classification image on disk, with its original folder label."""
    rows: list[dict[str, Any]] = []
    for key in cfg.raw["datasets"]:
        meta = cfg.dataset(key)
        if meta.get("task") != "classification":
            continue
        root = cfg.dataset_path(key)
        if root is None:
            continue
        # some uploads nest one more level (e.g. Fish_Data/images/<species>)
        layout = meta.get("layout", "")
        candidates = [root]
        if "Fish_Data" in layout:
            for extra in ("Fish_Data/images", "images", "Fish_Data"):
                if (root / extra).is_dir():
                    candidates = [root / extra]
                    break
        base = next((c for c in candidates if any(_iter_images(c))), None)
        if base is None:
            raise LayoutError(
                f"dataset '{key}' at {root} contains no images; declared layout "
                f"was {layout!r}")
        imgs = list(_iter_images(base))
        for p in imgs:
            rel = p.relative_to(root).parts
            rows.append({
                "path": str(p),
                "dataset": key,
                "label": _label_from_path(p, base),
                "orig_split": next((s for s in rel if s.lower() in
                                    {"train", "val", "valid", "test"}), ""),
                "bytes": p.stat().st_size,
            })
        log.info("discovered %-22s %6d images  %3d folders  (%s)",
                 key, len(imgs), len({r['label'] for r in rows if r['dataset'] == key}), base)
    if not rows:
        raise LayoutError("no classification images found in any mounted dataset")
    df = pd.DataFrame(rows)
    df["dataset_label"] = df["dataset"] + "::" + df["label"]
    return df


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def discover_detection(cfg, counts_only: bool = False) -> dict[str, Any]:
    """Report the detection datasets present and how their labels are stored."""
    report: dict[str, Any] = {}
    for key in cfg.raw["datasets"]:
        meta = cfg.dataset(key)
        if meta.get("task") != "detection":
            continue
        root = cfg.dataset_path(key)
        if root is None:
            continue
        images = [p for p in _iter_images(root)]
        label_dirs = [d for d in root.rglob("*") if d.is_dir() and d.name == "labels"]
        xml_dirs = [d for d in root.rglob("*") if d.is_dir() and d.name in
                    {"annotations", "Annotations", "xml"}]
        report[key] = {
            "root": str(root),
            "images": len(images),
            "label_dirs": [str(d) for d in label_dirs],
            "voc_dirs": [str(d) for d in xml_dirs],
            "n_label_txt": sum(len(list(d.glob("*.txt"))) for d in label_dirs),
            "n_xml": sum(len(list(d.glob("*.xml"))) for d in xml_dirs),
        }
        log.info("detection %-20s images=%d yolo_txt=%d voc_xml=%d", key,
                 report[key]["images"], report[key]["n_label_txt"], report[key]["n_xml"])
        if counts_only:
            continue
    if not report:
        raise LayoutError("no detection dataset mounted")
    return report


def build_single_class_yolo(cfg, out_dir: Path, imgsz_check: bool = True) -> dict[str, Any]:
    """Convert every mounted detection set to ONE class `fish`, ready for YOLO/RT-DETR.

    * YOLO txt labels: all class ids -> 0, boxes copied verbatim.
    * Pascal-VOC xml: converted to YOLO boxes.
    * Images are symlinked (not copied) to stay inside the 20 GB working quota;
      labels are real files because they are rewritten.
    Returns a dict with per-source counts, which goes into the metrics JSON so no
    reader mistakes this for DePondFi.
    """
    from xml.etree import ElementTree as ET

    out_dir = Path(out_dir)
    images_dir, labels_dir = out_dir / "images", out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {"sources": {}, "boxes": 0, "images": 0,
                             "collapsed_classes": {}}
    for key in cfg.raw["datasets"]:
        meta = cfg.dataset(key)
        if meta.get("task") != "detection":
            continue
        root = cfg.dataset_path(key)
        if root is None:
            continue
        src_stats = {"images": 0, "boxes": 0, "dropped_without_label": 0,
                     "class_ids_seen": set()}

        yolo_dirs = [d for d in root.rglob("*") if d.is_dir() and d.name == "labels"]
        voc_dirs = [d for d in root.rglob("*") if d.is_dir() and d.name in
                    {"annotations", "Annotations", "xml"}]

        # ---- YOLO sources ----
        for ldir in yolo_dirs:
            idir_candidates = [ldir.parent / "images", ldir.parent.parent / "images",
                               root / "images", ldir.parent]
            idir = next((c for c in idir_candidates if c.is_dir()), None)
            if idir is None:
                continue
            for lp in sorted(ldir.glob("*.txt")):
                img = next((idir / f"{lp.stem}{e}" for e in IMG_EXT
                            if (idir / f"{lp.stem}{e}").exists()), None)
                if img is None:
                    continue
                boxes = []
                for line in lp.read_text(errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    src_stats["class_ids_seen"].add(parts[0])
                    boxes.append("0 " + " ".join(parts[1:5]))
                if not boxes:
                    src_stats["dropped_without_label"] += 1
                    continue
                name = f"{key}_{img.stem}"
                (labels_dir / f"{name}.txt").write_text("\n".join(boxes) + "\n")
                _symlink(img, images_dir / f"{name}{img.suffix}")
                src_stats["images"] += 1
                src_stats["boxes"] += len(boxes)

        # ---- Pascal-VOC sources ----
        for xdir in voc_dirs:
            for xp in sorted(xdir.glob("*.xml")):
                try:
                    tree = ET.parse(xp)
                except ET.ParseError:
                    continue
                size = tree.find("size")
                if size is None:
                    continue
                w = float(size.findtext("width", "0"))
                h = float(size.findtext("height", "0"))
                if w <= 0 or h <= 0:
                    continue
                boxes = []
                for obj in tree.findall("object"):
                    bb = obj.find("bndbox")
                    if bb is None:
                        continue
                    x1, y1 = float(bb.findtext("xmin", 0)), float(bb.findtext("ymin", 0))
                    x2, y2 = float(bb.findtext("xmax", 0)), float(bb.findtext("ymax", 0))
                    xc, yc = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    boxes.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                if not boxes:
                    continue
                fname = Path(tree.findtext("filename", xp.stem)).stem
                img = next((p for p in _iter_images(root) if p.stem == fname), None)
                if img is None:
                    continue
                name = f"{key}_{img.stem}"
                if (labels_dir / f"{name}.txt").exists():
                    continue                      # YOLO labels take precedence
                (labels_dir / f"{name}.txt").write_text("\n".join(boxes) + "\n")
                _symlink(img, images_dir / f"{name}{img.suffix}")
                src_stats["images"] += 1
                src_stats["boxes"] += len(boxes)

        src_stats["class_ids_seen"] = sorted(src_stats["class_ids_seen"])
        stats["collapsed_classes"][key] = len(src_stats["class_ids_seen"])
        stats["sources"][key] = src_stats
        stats["images"] += src_stats["images"]
        stats["boxes"] += src_stats["boxes"]

    if stats["images"] == 0:
        raise LayoutError("no detection images with labels could be converted")

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir.resolve()}\ntrain: images\nval: images\n"
        f"names:\n  0: fish\nnc: 1\n")
    log.info("single-class detection set: %d images, %d boxes -> %s",
             stats["images"], stats["boxes"], out_dir)
    return stats


def _symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:            # filesystems without symlink support
        import shutil
        shutil.copy2(src, dst)
