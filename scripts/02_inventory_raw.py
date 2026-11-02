#!/usr/bin/env python3
"""Phase 2, step 2: measure what actually landed on disk.

No class counts are trusted from papers, dataset cards or READMEs - they are
counted from files here. This is the baseline that AUDIT_REPORT.md compares against.

Outputs
  data/interim/raw_inventory.json          structured inventory
  outputs/tables/raw_inventory_by_class.csv
  outputs/tables/raw_inventory_summary.csv

Run: .venv/bin/python scripts/02_inventory_raw.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import registry
from src.utils.paths import INTERIM, RAW, TABLES, ensure_dirs, rel, write_csv, write_json

# dataset subtree (relative to data/raw/<key>) that actually holds the data,
# so app screenshots / docs in a repo never get counted as dataset images
ROOTS = {m["key"]: (m.get("image_root"), m.get("label_root")) for m in registry.MIRRORS}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_info(path: Path, timeout_s: float = 0.0) -> dict:
    """Read header only (no full decode) - enough to catch truncated/missing files."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
            fmt = im.format
            im.verify()
        return {"ok": True, "w": w, "h": h, "format": fmt, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "w": None, "h": None, "format": None,
                "error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def scan_tree(root: Path) -> tuple[list[dict], list[dict]]:
    """Return (image records, label-file records) for a subtree."""
    imgs, txts = [], []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXT:
            info = image_info(p)
            imgs.append({"path": rel(p), "bytes": p.stat().st_size, **info})
        elif ext == ".txt":
            txts.append({"path": rel(p)})
    return imgs, txts


def classify_structure(key: str, imgs: list[dict], root: Path) -> dict:
    """Infer the layout: class-per-folder (classification) vs images+labels (detection)."""
    parent = Counter(Path(r["path"]).parent.as_posix() for r in imgs)
    rec = {"dataset": key, "n_images": len(imgs),
           "n_unreadable": sum(1 for r in imgs if not r["ok"]),
           "bytes": sum(r["bytes"] for r in imgs),
           "top_image_dirs": [{"dir": d, "n": n} for d, n in parent.most_common(12)]}
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    ensure_dirs()

    manifest_path = RAW / "MANIFEST.json"
    if not manifest_path.exists():
        print("no manifest - run scripts/01_download_datasets.py first")
        return 2
    manifest = json.loads(manifest_path.read_text())

    inventory: dict = {"datasets": {}, "generated_from": rel(manifest_path)}
    class_rows, summary_rows = [], []

    for m in manifest.get("mirrors", []):
        key = m["key"]
        if not m.get("ok"):
            continue
        if args.only and key not in args.only:
            continue
        ds_root = RAW / key
        img_root, label_root = ROOTS.get(key, (".", None))
        if img_root is None:
            # taxonomy dump - no imagery by design
            inventory["datasets"][key] = {"dataset": key, "type": "taxonomy",
                                          "n_images": 0, "n_classes": 0,
                                          "n_unreadable": 0, "bytes": 0,
                                          "note": "no imagery (taxonomy source)"}
            summary_rows.append({"dataset": key, "type": "taxonomy", "n_images": 0,
                                 "n_classes": 0, "n_unreadable": 0, "bytes": 0})
            print(f"== {key}: taxonomy dump, no imagery")
            continue
        root = ds_root / img_root
        print(f"== scanning {key} ({rel(root)})")
        imgs, txts = scan_tree(root)
        struct = classify_structure(key, imgs, root)

        if key == "fish_detection_proxy":
            # detection layout: images/ + labels/ (+ Pascal-VOC xml)
            label_dir = (ds_root / label_root) if label_root else None
            label_files = sorted(label_dir.glob("*.txt")) if label_dir else []
            stems = {Path(r["path"]).stem for r in imgs}
            label_stems = {p_.stem for p_ in label_files}
            class_hits: Counter = Counter()
            n_boxes = 0
            for lp in label_files:
                for line in lp.read_text(errors="ignore").splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        class_hits[int(parts[0])] += 1
                        n_boxes += 1
            xml_dir = ds_root / "data" / "Annotations"
            struct.update({
                "type": "detection",
                "n_label_txt": len(label_files),
                "n_boxes": n_boxes,
                "label_dir": rel(label_dir) if label_dir else None,
                "xml_annotations": len(list(xml_dir.glob("*.xml"))) if xml_dir.exists() else 0,
                "n_images_without_yolo_label": len(stems - label_stems),
                "n_yolo_labels_without_image": len(label_stems - stems),
                "yolo_class_id_histogram": dict(sorted(class_hits.items())),
                "n_classes_in_labels": len(class_hits),
                "boxes_per_image_mean": round(n_boxes / max(len(label_files), 1), 2),
            })
            summary_rows.append({"dataset": key, "type": "detection",
                                 "n_images": struct["n_images"],
                                 "n_classes": len(class_hits),
                                 "n_unreadable": struct["n_unreadable"],
                                 "bytes": struct["bytes"]})
        else:
            # classification layout: deepest folder that holds images == the class
            by_class: dict[str, list[dict]] = defaultdict(list)
            for r in imgs:
                p = Path(r["path"])
                parts = p.parts
                # ignore the dataset key prefix (data/raw/<key>/...)
                try:
                    i = parts.index(key)
                    sub = parts[i + 1:-1]
                except ValueError:
                    sub = parts[-2:-1]
                if img_root not in (".", ""):
                    # the layout (train/val/test/class) is kept as part of the class key
                    sub = tuple(sub)
                cls = sub[-1] if sub else "(root)"
                by_class[cls].append(r)
            struct.update({"type": "classification", "n_classes": len(by_class),
                           "classes": {c: len(v) for c, v in sorted(by_class.items())}})
            for cls, recs in sorted(by_class.items()):
                class_rows.append({"dataset": key, "class_dir": cls, "n_images": len(recs),
                                   "n_unreadable": sum(1 for r in recs if not r["ok"]),
                                   "bytes": sum(r["bytes"] for r in recs),
                                   "min_bytes": min(r["bytes"] for r in recs),
                                   "max_bytes": max(r["bytes"] for r in recs)})
            summary_rows.append({"dataset": key, "type": "classification",
                                 "n_images": struct["n_images"],
                                 "n_classes": len(by_class),
                                 "n_unreadable": struct["n_unreadable"],
                                 "bytes": struct["bytes"]})

        inventory["datasets"][key] = struct
        print(f"   images={struct['n_images']} unreadable={struct['n_unreadable']} "
              f"dirs={struct.get('n_classes') or struct.get('n_classes_in_labels')}")

    write_json(INTERIM / "raw_inventory.json", inventory)
    write_csv(TABLES / "raw_inventory_by_class.csv", class_rows)
    write_csv(TABLES / "raw_inventory_summary.csv", summary_rows)
    for r in summary_rows:
        print(f"   {r['dataset']:22s} {r['type']:15s} imgs={r['n_images']:6d} "
              f"classes={r['n_classes']:4d} bad={r['n_unreadable']}")
    print(f"wrote {rel(INTERIM / 'raw_inventory.json')}, "
          f"{rel(TABLES / 'raw_inventory_by_class.csv')}, "
          f"{rel(TABLES / 'raw_inventory_summary.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
