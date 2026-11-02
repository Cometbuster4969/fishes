# GeoFish-AI

**AI-IoT framework for fish species identification, population-density estimation and
habitat mapping in Indian waters** — detector → tracker → species classifier → counting →
density → spatial map.

> Status: see `outputs/DECISIONS.md` (blockers + fallbacks) and `outputs/DATASETS.md`
> (what is actually on disk). This environment is CPU-only and can reach only GitHub/PyPI,
> which constrains every number in `outputs/metrics/`.

## Quick start

```bash
bash scripts/setup_env.sh          # phase 1: venv + requirements.txt  (~10 min, 6.4 GB)
```

## One command per phase

| phase | what it does | command |
|---|---|---|
| 1 | repo structure, environment | `bash scripts/setup_env.sh` |
| 2a | probe every official dataset host | `.venv/bin/python scripts/00_probe_network.py` |
| 2b | download datasets (official → verified GitHub copy), manifest + sha256 | `.venv/bin/python scripts/01_download_datasets.py` |
| 2c | measure what is on disk (class counts, corrupt files) | `.venv/bin/python scripts/02_inventory_raw.py` |
| 2d | write `outputs/DATASETS.md` | `.venv/bin/python scripts/03_write_dataset_docs.py` |
| 3a | build `data/taxonomy/taxonomy_master.csv` (FishBase-derived) | `.venv/bin/python scripts/10_build_taxonomy.py` |
| 3b | resolve every dataset label → accepted name (`label_map.csv`) | `.venv/bin/python scripts/11_build_label_map.py` |
| 3c | automated label audit (dedupe, cleanlab, zero-shot, quality) | `.venv/bin/python scripts/12_audit_labels.py` |
| 3d | review pack + quarantine + `AUDIT_REPORT.md` | `.venv/bin/python scripts/13_build_review_pack.py` |
| 4 | leakage-safe splits (70/15/15, cluster-grouped) | `.venv/bin/python scripts/20_make_splits.py` |
| 5a | train detector (YOLO11 / YOLOv12 / RT-DETR) | `.venv/bin/python scripts/30_train_detector.py --config configs/det_yolo11n.yaml` |
| 5b | train classifier zoo (hierarchical head, `unknown` class) | `.venv/bin/python scripts/31_train_classifier.py --config configs/cls_*.yaml` |
| 6a | track + count (ByteTrack / BoT-SORT) | `.venv/bin/python scripts/40_track_count.py` |
| 6b | every §7 metric → `outputs/metrics/` | `.venv/bin/python scripts/41_evaluate.py` |
| 7 | density estimation + bootstrap CIs | `.venv/bin/python scripts/50_density.py` |
| 8 | maps: GeoJSON / PNG / interactive HTML | `.venv/bin/python scripts/60_map.py` |

Each script is idempotent, seeds at 42 and writes a JSON record of what it used.

## Layout

```
data/raw/           downloaded archives + extracted datasets (git-ignored, sha256 in MANIFEST.json)
data/taxonomy/      taxonomy_master.csv, label_map.csv
data/curated/       cleaned labels, dedupe map, quarantine index
data/splits/        train/val/test manifests (frozen before training)
src/                data · models · train · eval · track · density · mapping · iot · utils
configs/            one yaml per experiment (saved next to every run)
outputs/            metrics · figures · tables · models · review · quarantine · reports
scripts/            the phase commands in the table above
```

## Non-negotiables encoded in the code

1. No number in `outputs/` is hand-written; everything is computed from files or predictions.
2. Unavailable → the string `"not_computable"` plus a reason. Never a guess.
3. Test split is opened only by the final evaluation step.
4. Low-confidence label relabels are never applied — they go to `outputs/quarantine/` and
   `outputs/review/` for human sign-off.
