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
