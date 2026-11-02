# AUDIT_PLAN.md — label audit & mismatch fixes (spec §4.2, §4.3)

**Nothing is trained until this plan has run.** Every parameter below is fixed in
`configs/audit.yaml` so the run is reproducible; every count below was measured from disk
(`outputs/tables/raw_inventory_by_class.csv`), not read from a paper.

---

## 0. Scope, after Phase 2

| dataset on disk | images (counted) | classes (counted) | source status |
|---|---|---|---|
| `kaggle_large_scale_fish` | **10,504** | 11 folders | verified GitHub copy of Kaggle `crowww/a-large-scale-fish-dataset` |
| `bd_indigenous_fish` | **2,610** | 8 folders (Bengali names) | substitute for AS-FishFresh (official unreachable) |
| `fish_detection_proxy` | **2,184** | 19 YOLO classes, 2,360 boxes | substitute for DePondFi (images not in its repo) |
| `taxonomy_fishdex` | 0 (taxonomy dump) | 5,176 genus CSVs | FishBase-derived substitute for the FishBase API |

Not auditable because not on disk: Indian Seafood Fish (8 market species), AS-FishFresh
(48 species), QUT-468, iNat/GBIF India. Their §4.2 table is specified but **cannot be
executed** (see §1).

---

## 1. §4.2 — Indian Seafood 8-folder audit: decision table, verdicts = *pending imagery*

The images are unreachable, so no verdict below is an image-derived claim. This is the
executable decision logic; each row has the exact criterion the moment the files exist.
Treatment unit = **the folder (visual class)**, never the printed scientific name.

| # | folder / printed label | printed scientific name | problem | decision rule when images arrive | confidence needed |
|---|---|---|---|---|---|
| 1 | Pomfret, "Pampus argeneus" | *Pampus argeneus* | spelling error | rename → *Pampus argenteus*; no visual check needed (pure nomenclatural fix) | n/a — automatic |
| 2 | Mackerel, "Scomber scombrus" / local "Bangda" | *Scomber scombrus* | Atlantic species; Indian "Bangda" is normally *Rastrelliger kanagurta* | check for vertical bars + spotted belly (→ *R. kanagurta*) vs. plain metallic flanks (→ stay *S. scombrus*); relabel only if ≥2 of 3 independent signals agree | high |
| 3 | "Black Snapper", *Apsilus dentatus*, local "Tilapia" | *Apsilus dentatus* | African species; "Tilapia" = *Oreochromis* spp. → name and local name contradict | look for cichlid body shape, vertical bars, dorsal-fin spination → *Oreochromis niloticus* vs *O. mossambicus* (count dorsal rays / tail bands); relabel only with a decisive morphological cue | high |
| 4 | "Indian Carp", *Cyprinus carpio*, local "Rohu" | *Cyprinus carpio* | Rohu is *Labeo rohita*; *C. carpio* is common carp | barbel count + dorsal-fin base length + colour ⇒ *Labeo rohita* vs *C. carpio*; relabel only if decisive | high |
| 5 | Prawn, *Dendrobranchiata* | suborder, not a species | rank error | label `Penaeoidea_prawn (species unresolved)`; resolve only if the full carapace + rostrum + pereiopods are visible in ≥N images; otherwise keep as unresolved node in the hierarchy | n/a — rank fix automatic, species resolution needs high |
| 6 | "Pink Perch", *Zalembius rosaceus*, local "Rani" | *Zalembius rosaceus* | Pacific surfperch; Indian "Rani" = *Nemipterus japonicus* | check for filamentous upper caudal lobe + pink body (→ *N. japonicus*) | high |
| 7 | "Indian flathead", *Lates calcarifer*, local "Bhetki" | *Lates calcarifer* | English name and scientific name disagree | keep *L. calcarifer* **iff** images show barramundi (silver body, large mouth, distinct lateral line); otherwise correct the common name only, not the species | high |
| 8 | Black Pomfret, *Parastromateus niger* | correct name | sample image shows red fins → possible look-alike (pacu) | flag folder as mixed; split per-image rather than per-folder; any pacu/colossoma image → quarantine | high |

Also required by §4.2 and **not computable here**: the paper's Fig. 1 counts
(1,579 / 1,248 / 1,328 / 752 / 1,136 / 519 / 974 / 1,471, sum 9,007) vs stated 8,488.
Recorded in `AUDIT_REPORT.md` as **unverified — images unavailable**, with both numbers
quoted as *paper statements*, never as measurements.

---

## 2. §4.3 — automated audit (runs on every dataset that is on disk)

Six stages, all logged to `outputs/tables/`; final artefact `outputs/AUDIT_REPORT.md`.

### Stage A — exact + near-duplicate detection (§4.3.1)
- **Exact**: sha256 of file bytes; also byte-identical after EXIF strip.
- **Perceptual**: `pHash` (64-bit, `imagehash`, DCT-based). Candidate pair if
  **Hamming ≤ 6**; confirmed duplicate if additionally **HSV-histogram correlation ≥ 0.97**.
- **Resized-copy catcher**: 32×32 grayscale, normalized cross-correlation ≥ 0.995.
- **Embedding stage** (spec asks for CLIP/DINOv2 cosine > 0.97): *not available* — those
  weights cannot be downloaded (DECISIONS D-03). Substitute: penultimate-layer embeddings
  of the from-scratch audit CNN **plus** ORB keypoint matching, cosine ≥ 0.97 / ≥ 40
  inlier matches. Recorded explicitly as a weaker substitute in `AUDIT_REPORT.md`.
- Scope: **within and across splits** — the shipped `Dataset/{train,val,test}` folders of
  the Kaggle copy are pooled first, so cross-split duplicates are found and all of a
  cluster goes to one side of the new split (§5).
- Output: `outputs/tables/duplicates.csv` (`group_id, paths, n, method, score`),
  `data/curated/dedup_map.csv`.

### Stage B — label-noise detection (§4.3.2)
- 5-fold out-of-fold predictions from a small from-scratch CNN (128 px, ~6 epochs per
  fold, seed 42) — this is *not* the final model.
- `cleanlab.filter.find_label_issues` (confident learning) on the OOF `pred_probs`.
- Output: `outputs/tables/suspect_labels.csv`
  (`path, given_label, predicted_label, self_confidence, confidence, cleanlab_rank`).

### Stage C — third-opinion cross-check (§4.3.3)
- **BioCLIP zero-shot: unavailable** (weights blocked; DECISIONS D-03). Code path exists in
  `src/audit/zero_shot.py`, reports `available: false`.
- Substitute opinions, both offline:
  1. colour/texture descriptor (HSV histogram + LBP + HOG-lite) with a class centroid /
     kNN vote;
  2. the OOF CNN's top-1.
- A relabel is only applied if **cleanlab flags it AND both substitute opinions agree with
  the same alternative class**. Otherwise → quarantine.

### Stage D — corrupt / blurry / size filter (§4.3.4)
- Unreadable: PIL header+`verify()` — measured 0 corrupt files in all three datasets today.
- Blur: variance of Laplacian on the 512-px long side; flag **< 60** (documented threshold,
  swept once on a held-out margin sample and frozen).
- Size: flag `< 224 px` on the short side; record every image's width/height.
- Output: `outputs/tables/image_quality.csv`, summary histograms in
  `outputs/figures/quality_*.png`.

### Stage E — human review pack (§4.3.5)
- `outputs/review/` contact sheets: **25 images per class per dataset** (grid PNG, class
  name burned in) + **one sheet per flagged cluster** (duplicates, suspects, quarantined
  relabels, unresolved names) with a `REVIEW_INDEX.csv` the reviewer signs off in.
- `outputs/quarantine/` — every relabel below high confidence, every unresolved name, every
  low-quality image that is *not* deleted (we never delete; we exclude + record).

### Stage F — report
`outputs/AUDIT_REPORT.md`: before/after class counts, duplicates removed, relabels applied
(with the agreeing evidence), quarantined items, corrupt/blurry counts, and a
`NOT COMPUTED` section listing everything the environment prevented.

---

## 3. What this run will actually change (evidence gathered so far)

Already measured, will be reflected in `AUDIT_REPORT.md`:

1. **`fish_detection_proxy`: two misspelled class names** in `data/fish.yaml` —
   `Cephalopholis_fulvaw` → *Cephalopholis fulva*, `Rhincodon_typusw` → *Rhincodon typus*
   (trailing `w`). `nc: 19` matches the 19 listed names and all 19 ids 0–18 occur; every
   filename prefix maps to exactly one id → the annotation set is internally consistent.
2. **`fish_detection_proxy` is ~1 fish per image** (2,360 boxes / 2,184 images = 1.08), so
   it is a *classification-grade* detection set, not a dense-scene one — counting metrics
   will say so.
3. **`kaggle_large_scale_fish` folders are not the official 9**: 11 folders, including
   `animal fish` (1,803) and `animal fish bass` (53) which do not exist in the Kaggle
   9-class release; the shipped `train/val/test` split is discarded (DECISIONS D-09).
4. **`bd_indigenous_fish` labels are Bengali market names**, not scientific names
   (Stage G below).

### Stage G — Bengali label resolution (this is the Indian-species proxy, so it matters)
`label_map.csv` gets one row per folder with an explicit confidence; **only `high` is
applied**, everything else goes to `outputs/quarantine/` + review pack.

| folder | n images | proposed accepted name | basis | confidence | applied? |
|---|---|---|---|---|---|
| Koi | 500 | *Anabas testudineus* | standard Bengali market name for climbing perch (কৈ) | high | yes |
| Sol | 120 | *Channa striata* | standard Bengali name (শোল) for striped snakehead | high | yes |
| Sing | 400 | *Heteropneustes fossilis* | "Singhi" stinging catfish | medium | **no → review** |
| Sorputi | 200 | *Systomus sarana* | "Sarputi" olive barb | medium | **no → review** |
| Foli | 300 | *Notopterus notopterus* | "Pholi" bronze featherback | medium | **no → review** |
| Tengra | 320 | *Mystus* cf. *vittatus* | "Tengra" spans several *Mystus* spp. | low | **no → review** |
| Taki | 390 | *Channa punctata* | "Taki" also used for other *Channa* | low | **no → review** |
| Byen | 500 | unresolved | no verifiable match | unresolved | **no → review** |

Consequence, stated up front: if the reviewer does not promote the `medium` rows, the
Bengali proxy contributes **2 classes / 620 images** to supervised training (plus the rest
as unlabelled/quarantined). That is reported as-is; the number will not be padded.

### Stage H — naming policy for the Kaggle copy
Common-name folders (`black_sea_sprat`, `hourse_mackerel`, …) are resolved to scientific
names in `label_map.csv` with a confidence. **Resolution does not change the training
class**: the classifier trains on the dataset's own visual folders, and the taxonomy join is
reported separately, so a low-confidence name resolution can never silently rewrite a label.

---

## 4. Gate before Phase 5

`scripts/12_audit_labels.py` writes `outputs/AUDIT_REPORT.md` and refuses to emit a
`data/curated/approved_labels.csv` unless: (a) every dataset has a class count, (b) every
duplicate cluster is resolved to one split side, (c) every applied relabel has the required
agreement, (d) the quarantine index exists. Phase 5 reads only
`data/curated/approved_labels.csv`.

**Estimated runtime** (2 CPU cores): Stage A pHash over 15,298 images ≈ 6 min; Stage B
5-fold CPU training at 128 px ≈ 55 min; Stage D ≈ 4 min; Stage E contact sheets ≈ 2 min.
