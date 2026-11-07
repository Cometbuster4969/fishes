# Kaggle setup — complete walkthrough

Everything you need to turn this repo on in Kaggle, in order. Takes ~30 min the first time,
mostly waiting for uploads.

> **Read this first:** the GitHub repo is **private**. Kaggle cannot clone a private repo
> without help, so step 3 (get the code onto Kaggle) is the one that trips people up. You
> have three options and only need one.

---

## 0. What you need

| requirement | why | how to check |
|---|---|---|
| Kaggle account | notebooks + datasets | kaggle.com |
| **verified phone number** | Kaggle only unlocks **Internet** and GPUs after phone verification | *Settings → Phone verification* |
| GitHub access to `Cometbuster4969/fishes` | the notebooks import the code | `git ls-remote https://github.com/Cometbuster4969/fishes.git` |
| ~5 GB local disk (optional) | only if you use the Kaggle CLI to upload datasets | — |

Kaggle free tier, for budgeting:

| resource | limit |
|---|---|
| session length | **12 h** (9 h on TPU) |
| `/kaggle/working` (saved disk) | **20 GB** |
| GPU quota | ~**30 h / week** (Kaggle doesn't publish the number; yours is in *Settings → Your quota*) |
| accelerator options | `GPU P100` (1× 16 GB) or `GPU T4 ×2` (2× 16 GB) |
| RAM | ~29 GB with a GPU |

The code in this repo uses **one GPU**. `T4 ×2` therefore burns quota twice as fast for the
same throughput — pick **P100** unless you specifically need the newer card.

---

## 1. Create the notebook

1. kaggle.com → **Create → Notebook**.
2. Name it e.g. `geofish-01-data`.
3. **File → Import Notebook → File** → upload `notebooks/01_data_download.ipynb`.
   (Repeat for `02`…`06`, each as its own notebook — they are separate stages.)

Alternative for all six at once: **File → Import → GitHub**, connect your GitHub account
then pick `Cometbuster4969/fishes` → `notebooks/…`. This only shows private repos if you
linked GitHub in *Settings → Linked accounts*.

---

## 2. Turn on the three settings that matter

Right-hand sidebar → **Settings** (call it *Session options* on newer UI):

| setting | value | if you skip it |
|---|---|---|
| **Accelerator** | `GPU P100` (or `T4 ×2`) | notebook 02–05 stop immediately with a clear message; 01/06 run on CPU |
| **Internet** | **On** | pretrained weights can't download → `PretrainedDownloadError` (by design, no from-scratch fallback) |
| **Persistence** | **Files only** | checkpoints die when the session ends, resume breaks |

Also useful: *Environment → Pin to original* so a future Kaggle image change doesn't move
your library versions mid-project.

> **Internet is greyed out?** You haven't verified a phone number. Do that first
> (*Settings → Phone verification*), then reload the notebook.

---

## 3. Get the code onto Kaggle (pick ONE)

### Option A — `GITHUB_TOKEN` secret *(recommended: keeps you on the live branch)*

1. GitHub → *Settings → Developer settings → Personal access tokens → Tokens (classic)* →
   **Generate new token** → scope **`repo`** → copy it.
2. Kaggle notebook → **Add-ons → Secrets → Add a secret**:
   - Label: `GITHUB_TOKEN`
   - Value: paste the token
   - Toggle it **on** for this notebook.
3. Run cell **0a**. It detects the secret and clones the private repo:

```python
# what cell 0a does, in short
tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
url = f"https://x-access-token:{tok}@github.com/Cometbuster4969/fishes.git"
subprocess.run(["git", "clone", "--depth", "1", "-b", "arena/3dc076ac-fishes", url,
                "/kaggle/working/fishes"])
```

### Option B — upload the repo as a Kaggle **Dataset** *(works with Internet off, best for reproducibility)*

```bash
git clone -b arena/3dc076ac-fishes https://github.com/Cometbuster4969/fishes.git geofish
cd geofish && rm -rf .git data/raw        # 133 KB of source, nothing large
kaggle datasets init -p .
# edit dataset-metadata.json:
#   "title": "GeoFish-AI source"
#   "id": "<your-kaggle-username>/geofish-source"
#   "licenses": [{"name": "other"}]
kaggle datasets create -p .              # or: kaggle datasets version -p . -m "update"
```

Then in the notebook: **Add Input → Your Datasets → geofish-source**. The repo appears
read-only at `/kaggle/input/geofish-source`, cell 0a finds it and adds it to `sys.path`
(outputs still go to `/kaggle/working`, so read-only is fine).

### Option C — make the repo public

GitHub → repo → *Settings → General → Danger zone → Change visibility → Public*. Then cell
0a's plain `git clone` works with no token.

> Whichever you choose, cell 0a prints `repo: … | commit: …` — note that commit for your
> paper's reproducibility section.

---

## 4. Upload the datasets

`configs/kaggle.yaml` expects exact mount paths. Kaggle mounts a dataset at
**`/kaggle/input/<dataset-slug>`**, so create your datasets with these slugs.

| `configs/kaggle.yaml` key | expected mount | required | where it comes from |
|---|---|---|---|
| `large_scale_fish` | `/kaggle/input/a-large-scale-fish-dataset` | ✅ | public Kaggle dataset `crowww/a-large-scale-fish-dataset` — just **Add Input** and search the title |
| `fish_detection_yolo` | `/kaggle/input/fish-detection-yolo` | ✅ | your upload (see below) |
| `indian_seafood_fish` | `/kaggle/input/indian-seafood-fish-dataset` | optional | Mendeley `mxf2c45yb5` v1 archive |
| `as_fishfresh` | `/kaggle/input/as-fishfresh` | optional | Mendeley `sdhp9bmrz5` v2 archive |
| `bd_indigenous_fish` | `/kaggle/input/bdindigenousfish2019` | optional | GitHub `knowaminul/BDIndigenousFish2019` |
| `qut_fish_468` | `/kaggle/input/fish-species-image-data` | optional | public Kaggle `sripaadsrinivasan/fish-species-image-data` |
| `depondfi` | `/kaggle/input/depondfi` | optional | DePondFi challenge archive |
| `videos` | `/kaggle/input/fish-videos` | for 05 | your clips + `ground_truth_counts.csv` |
| `jh_fish_field` | `/kaggle/input/jh-fish-field` | for 06 | field rows + `sensor.csv` + `calibration/p_detect.csv` |

### Adding a public dataset (2 clicks)
Notebook → **Add Input** → search the title → **+ Add**. Done.

### Uploading your own files (UI)
Kaggle → **Datasets → New Dataset** → drag the folder → set the **title** so the slug
matches the table above (e.g. title `fish-detection-yolo` → slug `fish-detection-yolo`) →
**Create**. Then **Add Input → Your Datasets**.

### Uploading your own files (CLI, better for 500 MB+)

```bash
pip install kaggle
mkdir -p ~/.kaggle && cp ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# required detection set: the 'data' folder of 1wenjinjie/yolov10-fish-detection
kaggle datasets init -p fish-detection-yolo/data
#   set "id": "<username>/fish-detection-yolo" in fish-detection-yolo/data/dataset-metadata.json
kaggle datasets create -p fish-detection-yolo/data
kaggle datasets status <username>/fish-detection-yolo     # wait for "ready"
```

If a mount name doesn't match, **don't rename files** — edit the one line in
`configs/kaggle.yaml → datasets.<key>.kaggle_input`. That is the only place paths live.

---

## 5. Run

Order matters; each notebook is resumable and skips finished stages.

```
01_data_download  →  02_taxonomy_audit  →  03_detector  →  04_classifier
                  →  05_tracking_counting  →  06_density_map
```

**First run = smoke test** (do this before spending GPU hours):

1. Run cell `0a` → `repo: /kaggle/working/fishes | commit: …`
2. Run cell `0b` → pip installs `requirements-kaggle.txt`
3. Run cell `0c` → mount table + GPU report
4. Paste the preflight cell from the chat (weights load: `yolo11n.pt`, `resnet50.a1_in1k`,
   DINOv2, BioCLIP) — expect four `PASS` lines.

Only then run the rest, top to bottom.

**For runs longer than your browser session:** *Save Version → **Save & Run All (Commit)***
(note: *Commit* does **not** get Internet on some account tiers; if so, keep the browser tab
open and use *Save & Run All (Quick Save)* / interactive instead). A commit keeps running
after you close the tab, up to the 12 h cap.

---

## 6. Resume after a session dies

Within the same working dir: just re-run — `outputs/logs/state.json` makes finished stages
skip and training continues from the last saved epoch (`cls_*_last.pt`, Ultralytics
`last.pt`).

After `/kaggle/working` is wiped, use the saved output of the previous version:

1. Notebook's **Output** tab of the old version → **New Dataset** from it (name it e.g.
   `geofish-run-2026-11`).
2. In the new session: **Add Input** that dataset.
3. In `configs/kaggle.yaml`:

```yaml
resume:
  enabled: true
  previous_outputs: ["/kaggle/input/geofish-run-2026-11"]
```

4. Re-run: the tracker imports the old `state.json` and continues.

---

## 7. Quota hygiene

- 03 trains 5 detectors × 60 epochs and 04 trains 6 architectures × 30 epochs — that is more
  than one week's quota. Start with **one** entry in `detector.models` (`yolo11n`) and
  `classifier.models` (`resnet50`), get the full path to metrics, then add models.
- Cut `detector.epochs` / `classifier.epochs_stage2_full` in `configs/kaggle.yaml`.
- Delete run dirs between experiments: `!rm -rf /kaggle/working/outputs/runs` (keep
  `outputs/models`, `outputs/metrics`).
- Check remaining quota: Kaggle → *Settings → Your quota*.

---

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `could not clone … (private)` | repo is private, no token | step 3 Option A/B/C |
| `PretrainedDownloadError` | Internet off | Settings → Internet → On (needs phone verification) |
| `ModuleNotFoundError: ultralytics` / stale API | pip cell didn't run | run cell `0b`, then **Restart & Run All** |
| `DatasetMissingError: large_scale_fish` | dataset not attached | Add Input → search `A Large Scale Fish Dataset` |
| `DatasetMissingError: fish_detection_yolo` | your upload missing/renamed | step 4 (CLI); or edit `datasets.fish_detection_yolo.kaggle_input` |
| `RuntimeError: 02_taxonomy_audit needs a GPU` | accelerator off | Settings → Accelerator → GPU P100 |
| out of memory in 03/04 | `imgsz`/`batch` too big for 16 GB | `detector.batch: 8 → 4`, `classifier.batch: 32 → 16` |
| session killed at 12 h | Kaggle cap | resumable by design: re-run, or split `detector.models` |
| `No space left on device` | 20 GB working disk | `!du -sh /kaggle/working/*`, delete `outputs/runs`, `data/raw/_archives` |
| `outputs/review/` contact sheets missing | notebook 02 didn't reach step 8 | run 02 to the end; it needs the GPU stages before it |

---

## 9. Final checklist

- [ ] phone verified, Internet **On**, accelerator set, **Persistence → Files only**
- [ ] code on Kaggle via Option A / B / C, cell `0a` prints a commit hash
- [ ] `large_scale_fish` + `fish_detection_yolo` mounted (the two `required: true`)
- [ ] preflight cell: 4 × `PASS`
- [ ] `01` finished: `outputs/tables/mounts.csv` shows `ok` for both required datasets
- [ ] `02` finished: `outputs/AUDIT_REPORT.md` exists **before** any training notebook
- [ ] `configs/kaggle.yaml` still has `pretrained.require: true` (do not "fix" it to false — the code will refuse to run)

Record for reproducibility: repo commit (`0a`), `outputs/tables/DATASETS_SNAPSHOT.json`,
`outputs/AUDIT_REPORT.md`, `outputs/RUN_MANIFEST.json`.
