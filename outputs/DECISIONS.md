# DECISIONS.md — assumptions, blockers, fallbacks

Project: **GeoFish-AI** (Indian fish species ID, detection, counting, density mapping)
Branch: `arena/3dc076ac-fishes` · Seed: **42** everywhere · Environment: CPU-only sandbox.

Format: `D-xx | phase | decision | why | what it costs us`.

---

## Enforcement environment (measured 2026-11-02)

`outputs/reports/network_probe.json` is the evidence file. Reachable: `github.com`,
`api.github.com`, `codeload.github.com`, `pypi.org` / `files.pythonhosted.org`.
Everything else fails at the TLS layer (`TLS/SSL connection has been closed`).

Blocked hosts that the spec depends on: `data.mendeley.com`, `www.kaggle.com`,
`api.inaturalist.org`, `api.gbif.org`, `www.fishbase.se`, `www.marinespecies.org`,
`researcharchive.calacademy.org`, `huggingface.co`, `download.pytorch.org`,
`zenodo.org`, `github.com/.../releases/download/*` (redirects to blocked asset hosts).

---

## D-01 · Phase 2 · Official dataset hosts unreachable → verified GitHub copies only
**Decision.** Every official URL in `src/data/registry.py` is probed and logged. Where the
official host is blocked, a GitHub copy reachable through `codeload.github.com` is used
**only** if its provenance is checkable (folder names / class list match the official
dataset). The copy is pinned to a commit sha and sha256-recorded in `data/raw/MANIFEST.json`.
When no faithful copy exists, the item is recorded as **unavailable**. Nothing is ever
presented as an official download.
**Cost.** Two spec-critical classification sets and the primary detection set are missing
(D-02, D-03, D-05).

## D-02 · Phase 2 · DePondFi images unavailable → substitute detection set
**Decision.** The DePondFi GitHub repo contains only documentation (PDF/PPTX/DOCX); the
annotated archive lives off-GitHub and is unreachable. The detector is trained on
`1wenjinjie/yolov10-fish-detection` (2,184 images + YOLO label files, AGPL-3.0) as an
explicit **substitute**. All multi-class labels are collapsed to the single class `fish`
(spec §6).
**Cost.** No DePondFi number may be claimed anywhere. Detector results are
"fish detection on the substitute set", and any comparison with published DePondFi mAP is
invalid. Recorded in every metrics table with `dataset=fish_detection_proxy`.

## D-03 · Phase 2 · No pretrained backbone weights are reachable
**Decision.** torchvision (`download.pytorch.org`), timm/Hugging Face, open_clip (BioCLIP)
and Ultralytics release assets are all blocked. Therefore:
- all classifiers are trained **from scratch** on the available data (small models, low
  resolution) — no ImageNet/BioCLIP initialisation is claimed;
- BioCLIP zero-/few-shot baseline and the BioCLIP zero-shot audit cross-check **cannot be
  executed**; the code path exists (`src/audit/zero_shot.py`) but is disabled with
  `available=False` and every report says so;
- the audit's "third opinion" is taken from a from-scratch CNN's out-of-fold predictions
  plus a classical colour/texture-feature kNN (both run offline), instead of BioCLIP.
**Cost.** Much lower absolute accuracy than the literature, no open-set work based on a
pretrained embedding space, and the spec's BioCLIP deliverables are marked N/A with reason.

## D-04 · Phase 3 · FishBase / WoRMS / Eschmeyer unreachable → FishDex (FishBase-derived)
**Decision.** `taxonomy_master.csv` is built from the FishDex sharded dump
(`ScottMorris123/FishDex`: per-genus CSV with `ScientificName`, FishBase `SpecCode`,
`CommonName`, `Environments`; per-family `index.json` with `Family` + `Order`), plus manual
curation of every label in `label_map.csv`. FishBase is CC BY-NC 4.0 → this project is
therefore non-commercial.
**Cost.** No country distribution, no IUCN status, no synonym table, no state-level
occurrence. The **FishBase India checklist cannot be reproduced**, so "coverage vs FishBase
India" is reported as **not computable** (we do not estimate a denominator). `iucn_status`
and `states` columns stay empty with an explicit `source` note instead of being filled with
guesses.

## D-05 · Phase 3 · Indian Seafood (8 species) + AS-FishFresh (48 species) missing
**Decision.** The §4.2 folder-by-folder audit is written as a *decision table* whose
verdicts are marked `pending imagery` — the mapping logic is specified precisely enough to
be executed the moment the images arrive, but no image-derived claim is made. The §4.3
automated audit still runs end-to-end on every dataset that *is* on disk, so the machinery
is proven. The paper's Fig. 1 vs text discrepancy (9,007 vs 8,488) cannot be re-checked
and is reported as unverified.
**Cost.** The headline Indian-market label audit is a plan, not an executed result.
South-Asian coverage comes from the BDIndigenousFish2019 proxy (8 Bengali-named freshwater
species) — labelled as a proxy everywhere.

## D-06 · Phase 3 · Bengali folder labels in the proxy dataset
**Decision.** Folders are Bengali common names (`Koi`, `Sing`, `Sol`, `Taki`, `Tengra`,
`Foli`, `Sorputi`, `Byen`). Each is resolved to a scientific name in `label_map.csv` with an
explicit `confidence` and `method=native_name_literature`. Any mapping whose confidence is
not **high** goes to `outputs/quarantine/` + `outputs/review/` and is **not** applied to
training labels (spec §4.3.5).
**Cost.** Class count for training may shrink if the reviewer rejects a mapping.

## D-07 · Phase 2 · Kaggle "Fish Species Image Data" (QUT, 468 species) unavailable
**Decision.** No Kaggle credentials exist in the environment and the host is blocked.
Long-tail/468-species evaluation is therefore **not run**. `QUT FISH` appears in
`DATASETS.md` as unavailable; no long-tail numbers are fabricated.
**Cost.** Species breadth in this run is limited to what the reachable datasets contain.

## D-08 · Phase 2 · Dried-fish dataset identifier not verifiable → skipped
**Decision.** The spec gives only the article (Data in Brief 55 (2024) 110563); the Mendeley
dataset id cannot be resolved offline and is **not guessed**.
**Cost.** Optional item omitted.

## D-09 · Phase 2 · Pre-existing splits in mirrored datasets are discarded
**Decision.** `tar-ang-2004/Fish_Classification` ships a random `train/val/test` split (and
the repo's own `Dataset/test` contains images that may duplicate `train`). That split is
**ignored**: all splits are rebuilt from scratch by `src/data/split.py` using
embedding-cluster grouping + near-duplicate removal (spec §5), so no test image can be a
near-duplicate of a training image.
**Cost.** Slower (embedding pass) but the leakage guarantee is what §5 demands.

## D-10 · Phase 1 · Environment and disk
**Decision.** CPU-only (2 vCPU, ~3 GB RAM, ~12 GB free disk). Python 3.11 venv at `.venv`
(git-ignored, snapshot-excluded). PyPI `torch==2.5.1+cu124` wheel is installed because the
CPU index is blocked; it runs on CPU (`torch.cuda.is_available() == False`).
Training budgets are capped accordingly and **wall-clock limits are reported with results**.
**Cost.** Small models, low resolution, few epochs, 1–3 seeds where the spec asks for 3.

## D-11 · All phases · Honesty rules encoded in code
**Decision.** (a) No metric is ever written unless computed from files/predictions on disk;
(b) every metrics JSON carries `dataset`, `split`, `n_test_images`, `seed`, `wall_clock_s`
and `caveats`; (c) missing values are the string `"not_computable"` with a reason, never 0
or a plausible-looking number; (d) the test split is only read by the final evaluation
script, guarded by a checksum of the split file.

---

## Open items to re-check when the network opens
1. Run `scripts/01_download_datasets.py` again — Mendeley/Kaggle/GBIF/iNat blocks are then
   the only gate; execute the §4.2 audit on the real 8 folders, count real files, and
   correct the 9,007 / 8,488 discrepancy.
2. Pull the FishBase India checklist and fill `iucn_status`, `states`, synonyms, then
   report real coverage against it.
3. Enable BioCLIP zero-shot (`src/audit/zero_shot.py --enable`) and ImageNet-initialised
   backbones; re-run §6 with the same splits so numbers stay comparable.

## D-12 · Phase 3 · Duplicate-detection embedding substitute
**Decision.** Spec §4.3.1 asks for pHash **or** CLIP/DINOv2 cosine > 0.97. CLIP/DINOv2
weights are unreachable (D-03), so the embedding branch is replaced by a documented weaker
substitute: pHash Hamming ≤ 6 **confirmed by** HSV-histogram correlation ≥ 0.97, a 32×32
normalised-cross-correlation catch for resized copies, and (after the first audit CNN
exists) its penultimate-layer cosine ≥ 0.97 plus ORB inlier matching. The `configs/audit.yaml`
entry `dedupe.embedding.available` stays `false` so nobody mistakes the substitute for the
real thing.
**Cost.** Duplicates that survive resizing/JPEG recompression *and* colour shifts may be
missed; a leakage check on the final split is therefore run explicitly and reported.

## D-13 · Phase 3 · Bengali folder labels: only high-confidence mappings are applied
**Decision.** `Koi`→*Anabas testudineus* and `Sol`→*Channa striata* are high-confidence
standard Bengali market names and are applied. `Sing`, `Sorputi`, `Foli`, `Tengra`, `Taki`
are medium/low and `Byen` is unresolved: all six go to `outputs/quarantine/` +
`outputs/review/` and are **not** applied. If the reviewer does not promote them, the
Bengali proxy supplies 2 classes / 620 images to supervised training and the remaining
1,990 images are kept as unlabelled/quarantined. That reduced number is reported, never
padded.
**Cost.** Far fewer South-Asian classes than the 48-species AS-FishFresh set would have
provided.

---

# Pivot: Kaggle execution profile (2026-11-02, later the same day)

The sandbox was confirmed unfit for training (CPU-only, 2 vCPU, no pretrained weights
reachable). The project now ships a **Kaggle execution profile** instead of running phases
locally. The entries above stay as history of the restricted-environment attempt; the ones
they contradict are marked superseded.

## D-14 · Supersedes D-03 · Pretrained weights are mandatory, from-scratch is removed
**Decision.** `configs/kaggle.yaml` sets `pretrained.require: true` and
`pretrained.on_failure: raise`, and `src/utils/pretrained.py` is now the only construction
path for models. It refuses `pretrained=False`, refuses Ultralytics `.yaml` model files
(random init), verifies that loaded tensors are not a degenerate initialisation, retries
transient download failures, and raises `PretrainedDownloadError` containing the exact
Kaggle fix (enable internet, re-run, or attach the checkpoint as a dataset).
`src/config.py` refuses to run if `pretrained.require` is not true.
`scripts/validate_notebooks.py` fails on any reintroduction of `pretrained=False`,
`weights=None`, `.yaml` models, hard-coded `/kaggle/input` paths or literal seeds.
**Consequence.** Every number produced on Kaggle benefits from ImageNet/BioCLIP-style
initialisation; the "from scratch because the network was blocked" caveat no longer applies.
**Cost.** The notebooks need *Internet: On*. Offline Kaggle runs are impossible by design.

## D-15 · Supersedes D-10 · Execution environment is Kaggle, not this sandbox
**Decision.** Six notebooks (`01_data_download` … `06_density_map`) run top to bottom on a
Kaggle T4 × 2 kernel, write everything to `/kaggle/working`, and resume from
`outputs/logs/state.json`. Datasets come from `/kaggle/input` mounts declared in
`configs/kaggle.yaml`; a missing required mount raises `DatasetMissingError` with the fix,
never a silent skip. Ultralytics is pinned to `>=8.4.0` in `requirements-kaggle.txt`
because that is where YOLOv12 and RT-DETR checkpoint names exist.
**Cost.** None for correctness; wall-clock is bounded by Kaggle session limits, so training
budgets live in the config and each architecture is its own resumable stage.

## D-16 · Supersedes the "audit could not run" parts of D-05/D-12 · Audit now runs for real
**Decision.** With internet enabled, the audit executes as specified: DINOv2 embeddings for
duplicate/near-duplicate grouping (cosine ≥ 0.97) *and* for out-of-fold label-noise
probabilities, **BioCLIP zero-shot** as the third opinion, cleanlab on the OOF probabilities,
contact-sheet review packs, and a quarantine index. A relabel is applied only when cleanlab
and every available opinion agree and confidence passes
`audit.label_noise.min_confidence_to_apply_relabel` (0.90); everything else is quarantined.
The previously "unavailable" substitutions (CNN/ORB instead of CLIP embeddings, no BioCLIP)
are retired; `configs/audit.yaml` was deleted and its thresholds moved into
`configs/kaggle.yaml → audit`.
**Still true from D-04/D-05:** without the Mendeley datasets mounted, the eight §4.2 folder
verdicts remain *pending imagery*, and coverage vs the FishBase India checklist is
`not_computable`. `outputs/AUDIT_REPORT.md` states both explicitly.

## D-17 · Label-map confidence does not rewrite training labels by itself
**Decision.** Resolving a folder to a scientific name (`label_map.csv`) is metadata work and
does not change what the classifier learns. Only an **audit relabel** (unanimous cleanlab +
kNN + BioCLIP evidence) changes `dataset_label`. This keeps a medium-confidence name
resolution from silently rewriting a label, which was the failure mode §4.2 is about.
**Cost.** Taxonomy coverage and label coverage are reported as two separate numbers.

## D-18 · Legacy local scripts are kept but are not part of the Kaggle flow
**Decision.** `scripts/00_probe_network.py` … `scripts/03_write_dataset_docs.py`,
`scripts/setup_env.sh` and `requirements.txt` document the restricted-sandbox attempt
(they produced `outputs/DATASETS.md`). They are not used by any notebook, are not required
for a Kaggle run, and must not be mistaken for it. `outputs/DATASETS.md` remains as the
record of what that environment could and could not reach.

---

## D-19 · Per-package environment installer supersedes `pip install -r`
**Observed problem.** On Kaggle (Python 3.13, torch 2.11+cu130) `pip install -q -r
requirements-kaggle.txt` left `open_clip_torch` uninstalled with no visible reason: `-r` is
atomic, so one unresolvable package aborts the whole transaction, and the caller piped the
output through `tail`, hiding the resolver error.
**Decision.** `scripts/kaggle_env.py` is now the canonical installer and notebook cell `0b`
calls it. It installs only missing packages, one at a time, prints the real pip error, writes
`outputs/logs/env_report.json` and exits non-zero when a *required* package is absent.
open-clip is installed with `--no-deps` plus its pure-python dependencies
(`tiktoken, ftfy, wcwidth, regex, huggingface-hub, safetensors`) so pip can never replace the
image's CUDA build of torch/torchvision; versions are tried in order
`open-clip-torch>=4.0` → `==3.3.0` → `==2.32.0`. `requirements-kaggle.txt` stays as a
single-command fallback and no longer pins torch/torchvision.
**Unchanged.** BioCLIP remains *required* for the audit (spec 4.3.3): if it cannot be
installed the audit raises `PretrainedDownloadError` with the exact command — it does not
silently downgrade the audit. Optional packages (motmetrics, geopandas, folium, lightgbm,
contextily, mapclassify, seaborn) are reported as `missing_optional` and the affected
metrics state `not_computable` instead of inventing a value.

---

## D-20 · Notebook generator wrote source lines without their newlines
**Observed problem.** `md()`/`code()` in `scripts/build_notebooks.py` did
`text.split("\n")`, dropping the newline from every line. nbformat rejoins `source` with
`"".join(...)`, so each cell collapsed into **one line**. Because every code cell starts
with a `# @title` comment, the collapsed cell was one big comment: it parsed cleanly and
then silently did nothing. `scripts/validate_notebooks.py` used the same `"".join(...)`, so
`ast.parse` saw a valid comment and reported "all checks passed". Fixed in the generator
(`_lines()` keeps the trailing newline on every line but the last) and the validator now
also asserts the newline structure. `tests/test_dryrun_pipeline.py` gained
`test_notebooks_are_not_collapsed`, and a real `ast.parse` of the joined cell text caught
three cells (0a, 0b, 3a) whose error messages contained `\n` escapes that the generator's
f-strings turned into physical newlines inside a string literal.
**Lesson recorded.** Never trust "the cell compiles" when the first character is a comment;
assert structure as well as syntax.

## D-21 · Weights are proven before a missing dataset can stop the notebook
Cell 1a used to raise `DatasetMissingError` immediately, so on a kernel with zero datasets
the pretrained-weight checks in 2a never ran and "bad download" was indistinguishable from
"missing data". Cell 1a now only reports (`MOUNT_GAPS`), 2a verifies the three checkpoints,
2b prints `WEIGHTS OK, DATA MISSING -> attach [...]`, and the hard stop moved to 3a, the
first cell that actually needs pixels. Verified offline with nothing mounted:
`2 required dataset(s) MISSING: ['large_scale_fish', 'fish_detection_yolo']`,
`WEIGHTS OK, DATA MISSING -> attach [...]`, then `DatasetMissingError` at 3a.
