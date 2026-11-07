# GeoFish-AI

**AI-IoT framework for fish species identification, population-density estimation and
habitat mapping in Indian waters.** Detector → tracker → species classifier → counting →
density → spatial map.

The project runs on **Kaggle notebooks** (GPU T4 × 2). It is deliberately strict about two
things: **every model starts from pretrained weights** (there is no from-scratch fallback
anywhere — the code raises instead), and **no number is ever written that was not computed
from files on disk**.

```
notebooks/01_data_download.ipynb      →  02_taxonomy_audit.ipynb  →  03_detector.ipynb
                                     →  04_classifier.ipynb  →  05_tracking_counting.ipynb
                                     →  06_density_map.ipynb
```

---

> **Full step-by-step (private repo, dataset uploads, CLI, quota, troubleshooting):**
> [`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md). The summary is below.

## 1. One-time Kaggle setup

### 1.1 Get the code onto Kaggle

The repository is **private**, so cell `0a` needs one of these (see
[`docs/KAGGLE_SETUP.md`](docs/KAGGLE_SETUP.md) §3):

1. a Kaggle **secret named `GITHUB_TOKEN`** (GitHub PAT with `repo` scope) — cell `0a` clones
   with it;
2. the repo uploaded as a **Kaggle Dataset** — cell `0a` finds it under `/kaggle/input`
   (works with Internet off);
3. making the repo **public**.

Upload/import `notebooks/01…06` as six notebooks; each one bootstraps the code itself.

### 1.2 Enable the two settings that matter

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | `GPU T4 x 2` | notebooks 02–05 require CUDA and stop with a clear message otherwise; 01 and 06 run on CPU |
| **Internet** | **On** | pretrained weights (timm, Ultralytics, BioCLIP) are downloaded to `/kaggle/working/.cache`. Without internet the notebooks **fail on purpose** rather than training from scratch |
| **Persistence** | `Files only` | keeps checkpoints across *Save version*, which is what makes resume work |

### 1.3 Attach the data (*Add Data*)

Search Kaggle by name, or upload the archive as your own dataset. Mount names must match
`configs/kaggle.yaml → datasets.<key>.kaggle_input` — if your mount is named differently,
edit **that one line** in the config.

| key | what to attach | required | how |
|---|---|---|---|
| `large_scale_fish` | **A Large Scale Fish Dataset** (Kaggle, `crowww/a-large-scale-fish-dataset`) | ✅ yes | Add Data → search the title → attach (mounts as `/kaggle/input/a-large-scale-fish-dataset`) |
| `fish_detection_yolo` | a YOLO-format detection set, e.g. the `data/` folder of `1wenjinjie/yolov10-fish-detection` (2,184 images / 2,360 boxes / 19 classes) | ✅ yes | download → *New Dataset* → upload the `data` folder → title it `fish-detection-yolo` |
| `depondfi` | DePondFi pond-detection archive (official challenge data) | optional | upload as `depondfi`; if attached it is used **in addition**, and the metrics name their source |
| `indian_seafood_fish` | Mendeley `mxf2c45yb5` v1 — the 8-folder Indian market set that drives the §4.2 audit | optional | download the archive, upload as `indian-seafood-fish-dataset` |
| `as_fishfresh` | Mendeley `sdhp9bmrz5` v2 — 48 Assam freshwater species | optional | upload as `as-fishfresh` (keep augmented copies in a subfolder) |
| `bd_indigenous_fish` | `BDIndigenousFish2019` (8 Bangladeshi freshwater species) | optional | GitHub → *New Dataset* → `bdindigenousfish2019` |
| `qut_fish_468` | **Fish Species Image Data** (`sripaadsrinivasan`, 468 species) | optional | Add Data → attach |
| `videos` | your clips (`.mp4`) for tracking + counting | needed for 05 | upload as `fish-videos`; include `ground_truth_counts.csv` (`clip,species,count`) if you want counting metrics |
| `jh_fish_field` | field/JH-Fish data: `field_schema.csv` (spec §9 columns), `sensor.csv` / `mqtt.log`, `calibration/p_detect.csv` | needed for 06 | upload as `jh-fish-field` |

A missing **required** dataset raises `DatasetMissingError` with the exact fix in the
message — it never silently skips a phase. Missing optional datasets are reported in
`outputs/tables/mounts.csv` and the affected metrics are marked `not_computable`.

### 1.4 Storage budget

Kaggle gives ~20 GB of writable `/kaggle/working`. Notebook 01 symlinks images instead of
copying them, and checkpoints are pruned to the last `keep_last_k_checkpoints`. Clean up
between runs with:

```python
!rm -rf /kaggle/working/fishes /kaggle/working/outputs/runs /kaggle/working/.cache
```

---

## 2. Run order and what each notebook produces

Run the notebooks **in order**; each is resumable and idempotent.

| # | notebook | GPU | produces |
|---|---|---|---|
| 01 | `01_data_download` | no | `outputs/tables/{mounts,class_counts,DATASETS_SNAPSHOT}.json`, single-class detection set + splits |
| 02 | `02_taxonomy_audit` | yes | `data/taxonomy/{taxonomy_master,label_map}.csv`, `outputs/tables/{duplicates,suspect_labels,image_quality,relabel_decisions}.csv`, `outputs/quarantine/`, `outputs/review/`, `outputs/AUDIT_REPORT.md`, `data/curated/approved_labels.csv` |
| 03 | `03_detector` | yes | `outputs/models/det_*_best.pt`, `outputs/metrics/detection_*.json`, `outputs/figures/detection_pr_f1.png`, detector crops |
| 04 | `04_classifier` | yes | `outputs/models/cls_*_best.pt`, `outputs/metrics/classifier_*_{val,test}.json`, confusion matrices, ECE, open-set threshold, ablation, cross-domain drop |
| 05 | `05_tracking_counting` | yes | `outputs/tables/tracks_*.csv`, `outputs/tables/counts_*.csv`, `outputs/metrics/counting_{summary,metrics}.json` |
| 06 | `06_density_map` | no | `outputs/metrics/density_estimates.csv`, `outputs/figures/maps/*` (GeoJSON, PNG, interactive HTML), `outputs/RUN_MANIFEST.json` |

Nothing is trained before **02** has written `approved_labels.csv`; notebook 04 refuses to
start without it. The test split is read exactly once, in notebook 04 step 8, after the
frozen split lock is verified.

### Resume behaviour

Every stage writes to `/kaggle/working/outputs/logs/state.json` and **skips itself if it
already finished**. If a session dies:

1. `Save version` so `/kaggle/working` is preserved, then
2. re-run the notebook — finished stages are skipped, and training restarts from the last
   saved epoch (`cls_*_last.pt` / Ultralytics `last.pt` with `resume=True`).
3. If the working directory is gone entirely, attach the previous saved output as a
   dataset and add its mount to `configs/kaggle.yaml → resume.previous_outputs`.

### Rough wall-clock (T4, estimates — measure your own first run)

| notebook | estimate |
|---|---|
| 01 | 5–15 min |
| 02 | 25–45 min (embedding pass dominates) |
| 03 | 2–5 h for the full detector zoo (drop models from `detector.models` to shorten) |
| 04 | 3–6 h for six architectures × 30 epochs (each arch is one resumable stage) |
| 05 | 10–30 min per clip |
| 06 | 5 min |

---

## 3. Configuration

Everything is in **`configs/kaggle.yaml`** — paths, datasets, pretrained checkpoint names,
training budgets, audit thresholds, tracking and density parameters. Notebooks read it via
`src/config.py`; nothing is hard-coded.

Key blocks:

```yaml
pretrained:
  require: true        # a missing checkpoint raises PretrainedDownloadError
  on_failure: raise    # there is no silent fallback
  classifiers:         # spec §1 zoo -> real timm tags
    resnet50: resnet50.a1_in1k
    ...
detector: {imgsz: 960, epochs: 60, models: [yolo11n, yolo11s, ...]}
classifier: {input_size: 288, epochs_stage2_full: 26, hierarchical_loss_weights: {...}}
audit: {quality: {...}, label_noise: {...}}
resume: {enabled: true, previous_outputs: []}
```

## 4. The pretrained-only policy

* `src/utils/pretrained.py` is the **only** place models are created. It refuses
  `pretrained=False`, refuses an Ultralytics `.yaml` (which would mean random init), verifies
  that the loaded weights are not a degenerate/random initialisation, retries downloads, and
  raises `PretrainedDownloadError` with the Kaggle fix instructions.
* `src/config.py` refuses a config where `pretrained.require` is not true.
* `scripts/validate_notebooks.py` fails the build if a notebook reintroduces
  `pretrained=False`, `weights=None`, a `.yaml` model, or a hard-coded `/kaggle/input` path.

Turn the internet **on**, or nothing will run. That is intentional.

## 5. Repository layout

```
configs/kaggle.yaml        single source of truth (paths, datasets, weights, budgets)
notebooks/                 01…06, generated by scripts/build_notebooks.py
src/config.py              config loading, path creation, dataset enforcement
src/utils/pretrained.py    strict pretrained loading (no from-scratch path)
src/utils/state.py         resume: stage tracker, atomic state.json
src/data/                  discovery, splits (leakage-safe), datasets, embeddings
src/audit/                 dedupe, quality, cleanlab, zero-shot, review pack, report
src/models/classifier.py   hierarchical head (species/genus/family), losses, open-set
src/train/                 two-stage fine-tuning with layer-wise LR decay + resume
src/eval/                  §7 metrics (classification + a COCO-style detector evaluator)
src/track/                 ByteTrack/BoT-SORT wrapper, species voting, counting, MOT
src/density/, src/mapping/ density with bootstrap CIs, GeoJSON/PNG/HTML maps
src/iot/                   JH-Fish schema, sensor/MQTT ingestion, presence model
tests/test_dryrun_pipeline.py  synthetic end-to-end checks (no data, no GPU, no training)
```

## 6. Developer commands (no Kaggle needed)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build_notebooks.py        # regenerate notebooks/01…06
.venv/bin/python scripts/validate_notebooks.py     # static checks on all six
.venv/bin/python -m pytest tests/ -q               # dry-run pipeline tests (~2 s)
```

`tests/test_dryrun_pipeline.py` builds a synthetic `/kaggle` tree and exercises config
enforcement, discovery, detection conversion, split grouping, quality flags, resume
semantics and the audit report — so a broken notebook fails here, not after 40 minutes of
GPU time.

### Local dry-run of a notebook

Set a path rewrite and the config will point at a writable tree:

```bash
GF_REMAP="/kaggle/working=/tmp/kgwork;/kaggle/input=/tmp/kginput" \
  .venv/bin/python -c "from src.config import load_config, describe; print(describe(load_config()))"
```

## 7. What this repository does *not* claim

* Nothing here has been trained yet in this checkout — the sandbox that produced it had no
  GPU and could reach only GitHub/PyPI, so **all training runs happen on Kaggle**.
* `outputs/DECISIONS.md` records every known limitation (substitute datasets, missing
  FishBase checklist, label-confidence policy). Read it before quoting any number.
* `outputs/AUDIT_PLAN.md` documents the §4.2/§4.3 audit logic, including which relabels are
  auto-applied (high confidence only) and which are quarantined for human review.

## 8. Licences and attribution

Datasets keep their own licences and attribution (recorded per-dataset in
`configs/kaggle.yaml` and in `outputs/DATASETS.md`). FishBase-derived taxonomy is
CC BY-NC 4.0 → **this project is non-commercial**. The detection fallback set is AGPL-3.0.
