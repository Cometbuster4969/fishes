#!/usr/bin/env python3
"""Generate notebooks/0X_*.ipynb from one place so the six notebooks stay
consistent (same bootstrap, same resume handling, same output conventions).

Run:  .venv/bin/python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

REPO = "https://github.com/Cometbuster4969/fishes.git"
BRANCH = "arena/3dc076ac-fishes"


def _lines(text: str) -> list[str]:
    """nbformat source lines: every line keeps its trailing newline except the last.

    Jupyter/nbformat rejoin a source array with ''.join(...), so lines WITHOUT their
    newline collapse into a single line - and because each of our code cells starts with
    a `# @title` comment, the whole cell would silently become one comment.
    """
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": _lines(text)}


def setup_cells(nb: str, require_gpu: bool, pip: str = "-r requirements-kaggle.txt") -> list[dict]:
    """The identical first three cells of every notebook."""
    return [
        md(f"""
# {nb.replace('_', ' ').title()}
**GeoFish-AI** · runs top to bottom on a Kaggle GPU (T4 x2 or P100).

**Before you run**
1. *Settings → Accelerator*: **GPU T4 x2**
2. *Settings → Internet*: **On** (pretrained weights are mandatory — the code raises rather than training from scratch)
3. *Add Data*: attach the datasets listed in cell 3 (required ones are enforced)
4. *Settings → Persistence*: **Files only**, so checkpoints survive between saves

Every stage writes into `/kaggle/working`, records itself in
`/kaggle/working/outputs/logs/state.json`, and **skips itself on a re-run**.
An interrupted session resumes from the last finished epoch/stage.
"""),
        code(f"""
# @title 0a · Enter the repo (private repo: needs a GITHUB_TOKEN secret, or the repo as a Dataset)
import os, subprocess, sys
from pathlib import Path

REPO_SLUG = "{REPO.split('github.com/')[1].replace('.git','')}"
BRANCH = "{BRANCH}"
CLONE_DIR = Path("/kaggle/working/fishes")

def _is_repo(p):
    return (p / "configs" / "kaggle.yaml").exists()

def _token():
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("GITHUB_TOKEN").strip()
    except Exception:
        return None

# 1) already inside a checkout?
REPO = next((p for p in [Path.cwd(), *Path.cwd().parents] if _is_repo(p)), None)
# 2) repo attached as a (read-only) Kaggle Dataset?
if REPO is None:
    for pattern in ("*/configs/kaggle.yaml", "*/*/configs/kaggle.yaml"):
        hits = sorted(Path("/kaggle/input").glob(pattern)) if Path("/kaggle/input").exists() else []
        if hits:
            REPO = hits[0].parent.parent
            break
# 3) clone
if REPO is None and _is_repo(CLONE_DIR):
    subprocess.run(["git", "-C", str(CLONE_DIR), "pull", "-q", "origin", BRANCH], check=False)
    REPO = CLONE_DIR
if REPO is None:
    CLONE_DIR.parent.mkdir(parents=True, exist_ok=True)
    tok = _token()
    url = (f"https://x-access-token:{{tok}}@github.com/{{REPO_SLUG}}.git" if tok
           else f"https://github.com/{{REPO_SLUG}}.git")
    res = subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, str(CLONE_DIR)],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            "could not clone " + REPO_SLUG + "\\n  git: " + (res.stderr or "")[:300] +
            "\\n  The repo is PRIVATE. Do ONE of these, then re-run this cell:" +
            "\\n   1) Add-ons -> Secrets -> new secret GITHUB_TOKEN = a GitHub PAT with 'repo'"
            " scope; enable it for this notebook." +
            "\\n   2) Upload the repo as a Kaggle Dataset (read-only under /kaggle/input)." +
            "\\n   3) Make the repository public on GitHub.")
    REPO = CLONE_DIR

os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
writable = os.access(REPO, os.W_OK)
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                     text=True).stdout.strip() or "unknown"
print("repo:", REPO, "| commit:", sha, "| writable:", writable)
if not writable:
    print("note: code dir is read-only (attached dataset); all outputs still go to "
          "/kaggle/working via configs/kaggle.yaml")
"""),
        code(f"""
# @title 0b · Dependencies (per-package install, real errors shown)
# `pip install -r ...` is atomic: one bad package installs nothing and the reason is easy
# to miss. scripts/kaggle_env.py installs ONLY what is missing, one package at a time,
# prints the actual pip error, and exits non-zero if something REQUIRED is absent
# (timm, ultralytics, open-clip/BioCLIP, cleanlab, imagehash, lap, cv2, sklearn, matplotlib).
import json, subprocess, sys
from pathlib import Path

if Path("/kaggle").exists():
    rc = subprocess.run([sys.executable, "scripts/kaggle_env.py"]).returncode
    report = Path("outputs/logs/env_report.json")
    if report.exists():
        rep = json.loads(report.read_text())
        print("missing required:", rep["missing_required"] or "none")
        print("missing optional:", rep["missing_optional"] or "none")
    if rc != 0:
        raise RuntimeError(
            "environment incomplete - read the REQUIRED PACKAGES MISSING block above.\\n"
            "Then Run -> Restart (not just re-run), and run cells 0a + 0b again.")
else:
    print("not on Kaggle: skipped (use requirements.txt for local dev)")
"""),
        code(f"""
# @title 0c · Config, seeding, resume state, mount report
from src.utils.notebook import bootstrap
from src.config import describe

env = bootstrap("{nb}", require_gpu={require_gpu})
cfg, tr, log = env.cfg, env.tracker, env.log
print(describe(cfg))
print()
from src.utils.state import stage, validate_state
print("stages already done:", [k for k, v in tr.state["stages"].items()
                               if v.get("status") == "done"])
"""),
    ]


# =========================================================================== #
# 01 - data download / inventory
# =========================================================================== #
nb01 = setup_cells("01_data_download", require_gpu=False) + [
    md("""
## 1 · Check the mounts
Kaggle datasets are read-only under `/kaggle/input`. The table below is generated from
`configs/kaggle.yaml`.

**This cell never stops the notebook.** It reports every declared dataset, then records
`MOUNT_GAPS` - the required ones you still have to attach. The hard stop happens later,
in the first cell that actually needs the data (section 3), so that the environment
(section 0) and the pretrained weights (section 2) are still proven to work while you are
chasing datasets. That is deliberate: a broken download and a missing dataset look
identical if you only see the first error.
"""),
    code("""
# @title 1a · Mount report (reports, never raises)
import pandas as pd
from src.config import DatasetMissingError  # used by 3a

# Report EVERY declared dataset - do not abort on the first missing one. A missing
# mount is recorded in MOUNT_GAPS and only becomes fatal in 3a, so the weight checks
# in 2a still run.
mounts = pd.DataFrame(cfg.dataset_report())
display(mounts[["dataset", "task", "required", "status", "path"]])
mounts.to_csv(cfg.paths.tables / "mounts.csv", index=False)

MOUNT_GAPS = cfg.missing_required()          # required datasets not attached yet
n_ok = int((mounts.status == "ok").sum())

if MOUNT_GAPS:
    print(f"{len(MOUNT_GAPS)} required dataset(s) MISSING: {MOUNT_GAPS}")
    print("   Kaggle right sidebar -> Add Input -> attach each one:")
    for r in mounts.to_dict("records"):
        if r["required"] and r["status"] != "ok":
            print(f"     {r['dataset']:22s} expected at {r['path']}")
    print("   Re-run this cell after attaching. Sections 0 and 2 can be run NOW -")
    print("   the notebook stops at 3a, the first place that needs images.")
else:
    print(f"all {len(mounts)} declared datasets mounted ({n_ok} ok)")
"""),
    md("""
## 2 · Pretrained weights warm-up
This **proves** the network path works before any training starts: detector, classifier,
embedding encoder and BioCLIP. If any checkpoint cannot be fetched, the cell raises
`PretrainedDownloadError` — there is no random-init fallback anywhere in this repo.
"""),
    code("""
# @title 2a · Fetch one checkpoint of every family (cached in /kaggle/working/.cache)
from src.utils.pretrained import (PretrainedDownloadError, timm_model,
                                  ultralytics_model, open_clip_bioclip)

key = "pretrained.warmup"
if tr.is_done(key):
    print("already verified in this working dir:", tr.get(key).metrics)
else:
    with stage(tr, key) as st:
        if st.skip:
            print("skipped (done)")
        else:
            report = {}
            try:
                m = ultralytics_model(cfg.raw["pretrained"]["detectors"]["yolo11n"])
                report["detector"] = "yolo11n.pt loaded"
                del m
            except PretrainedDownloadError:
                raise
            m = timm_model(cfg.raw["pretrained"]["classifiers"]["resnet50"])
            report["classifier"] = cfg.raw["pretrained"]["classifiers"]["resnet50"]
            del m
            bio, _, _ = open_clip_bioclip(cfg.raw["pretrained"]["bioclip"]["open_clip_tag"])
            report["bioclip"] = "loaded"
            del bio
            print(report)
            st.metric(**{k: 1 for k in report})
"""),
    code("""
# @title 2b · Verdict: weights vs data
_weights = "WEIGHTS OK" if tr.is_done("pretrained.warmup") else "WEIGHTS UNVERIFIED"
if MOUNT_GAPS:
    print(f"{_weights}, DATA MISSING -> attach {MOUNT_GAPS}")
    print("   Everything except the image data is ready. Attach the datasets above,")
    print("   re-run 1a, then continue with 3a.")
else:
    print(f"{_weights}, DATA OK -> continue with 3a")
"""),
    md("""
## 3 · Classification corpus manifest
One row per image: `path, dataset, label, dataset_label, orig_split`. The shipped
train/val/test folders are **ignored** here — splits are rebuilt leakage-safely in
notebook 02.
"""),
    code("""
# @title 3a · Discover every classification image
from src.config import DatasetMissingError
from src.data.discovery import discover_classification

# first cell that needs pixels: this is where a missing mount becomes fatal
if MOUNT_GAPS:
    raise DatasetMissingError(
        f"{len(MOUNT_GAPS)} required dataset(s) still not mounted: {MOUNT_GAPS}\\n"
        "  Kaggle right sidebar -> Add Input -> attach them, re-run 1a, then this cell.\\n"
        "  Sections 0 and 2 are already green, so the environment and the pretrained\\n"
        "  weights are fine; only the image data is absent.")

key = "data.classification_manifest"
manifest_path = cfg.paths.data_curated / "all_classification.csv"
if tr.is_done(key) and manifest_path.exists():
    df_cls = pd.read_csv(manifest_path)
    print("resumed:", len(df_cls), "images")
else:
    with stage(tr, key) as st:
        if not st.skip:
            df_cls = discover_classification(cfg)
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            df_cls.to_csv(manifest_path, index=False)
            st.out(manifest_path)
            st.metric(n_images=int(len(df_cls)), n_datasets=int(df_cls.dataset.nunique()),
                      n_classes=int(df_cls.dataset_label.nunique()))
            print(f"{len(df_cls)} images, {df_cls.dataset_label.nunique()} classes")
            st.out(cfg.paths.tables / "class_counts.csv")
            df_cls.groupby(["dataset", "label"]).size().rename("n_images").reset_index() \\
                  .to_csv(cfg.paths.tables / "class_counts.csv", index=False)
display(df_cls.groupby("dataset").agg(images=("path", "size"),
                                      classes=("dataset_label", "nunique")))
"""),
    md("""
## 4 · Detection corpus → single class `fish`
Every mounted detection dataset is converted to one class (`fish`), whatever its source
label space was. Images are symlinked, labels rewritten, and the collapse is recorded so
the metrics can state exactly what the detector was trained on.
"""),
    code("""
# @title 4a · Build the single-class detection set
import json
from src.data.discovery import build_single_class_yolo, discover_detection

key = "data.detection_single_class"
det_dir = cfg.paths.data_interim / "detection_single_class"
if tr.is_done(key) and (det_dir / "data.yaml").exists():
    det_stats = tr.get(key).metrics
    print("resumed:", det_stats)
else:
    with stage(tr, key) as st:
        if not st.skip:
            survey = discover_detection(cfg)
            det_stats = build_single_class_yolo(cfg, det_dir)
            src_json = cfg.paths.tables / "detection_sources.json"
            src_json.write_text(json.dumps({"survey": survey, "converted": det_stats},
                                           indent=2))
            st.out(det_dir / "data.yaml", src_json)
            st.metric(images=det_stats["images"], boxes=det_stats["boxes"])
            print(det_stats)
"""),
    md("## 5 · Leakage-safe detection splits (grouped by near-duplicate frames)"),
    code("""
# @title 5a · Split frames 70/15/15 by pHash group, then write the YOLO data yaml
from src.data.split import build_detection_splits

key = "data.detection_splits"
if tr.is_done(key):
    split_summary = tr.get(key).metrics
    print("resumed:", split_summary)
else:
    with stage(tr, key) as st:
        if not st.skip:
            imgs = sorted((det_dir / "images").glob("*"))
            split_summary = build_detection_splits(imgs, cfg.paths.data_splits / "detection",
                                                   seed=cfg.seed)
            yaml_path = det_dir / "data.yaml"
            yaml_path.write_text(
                f"path: {det_dir}\\n"
                f"train: {cfg.paths.data_splits}/detection/train_images.txt\\n"
                f"val: {cfg.paths.data_splits}/detection/val_images.txt\\n"
                f"test: {cfg.paths.data_splits}/detection/test_images.txt\\n"
                f"names:\\n  0: fish\\nnc: 1\\n")
            st.out(yaml_path, cfg.paths.data_splits / "detection" / "DETECTION_SPLIT_LOCK.json")
            st.metric(**{k: v["n"] for k, v in split_summary["files"].items()})
            print({k: v["n"] for k, v in split_summary["files"].items()})
"""),
    md("""
## 6 · Integrity snapshot
Counts are measured from disk (never copied from a paper or dataset card). This block also
flags unreadable files, which would otherwise fail later in odd ways.
"""),
    code("""
# @title 6a · Verify decodability of a random sample + size statistics
import random
from PIL import Image
import numpy as np

key = "data.integrity"
if tr.is_done(key):
    print(tr.get(key).metrics)
else:
    with stage(tr, key) as st:
        if not st.skip:
            sample = random.Random(cfg.seed).sample(df_cls.path.tolist(),
                                                    min(3000, len(df_cls)))
            bad, sizes = [], []
            for p in sample:
                try:
                    with Image.open(p) as im:
                        sizes.append(im.size)
                except Exception as exc:
                    bad.append((p, str(exc)[:80]))
            ws = np.array([s[0] for s in sizes]); hs = np.array([s[1] for s in sizes])
            stats = {"sampled": len(sample), "unreadable": len(bad),
                     "median_w": int(np.median(ws)), "median_h": int(np.median(hs)),
                     "min_short_side": int(min(ws.min(), hs.min()))}
            print(stats)
            if bad:
                print("first unreadable:", bad[:3])
            st.metric(**stats)
"""),
    md("## 7 · Provenance record + handoff"),
    code("""
# @title 7a · Write DATASETS_SNAPSHOT.json and print the run state
import json, hashlib, datetime
snapshot = {
    "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
    "config": str(cfg.path),
    "config_sha256": hashlib.sha256(Path(cfg.path).read_bytes()).hexdigest(),
    "git_commit": sha,
    "mounts": mounts.to_dict(orient="records"),
    "classification": {"images": int(len(df_cls)),
                       "classes": int(df_cls.dataset_label.nunique()),
                       "per_dataset": df_cls.dataset.value_counts().to_dict()},
    "detection": det_stats,
    "detection_splits": split_summary,
    "pretrained_policy": cfg.raw["pretrained"]["require"],
}
out = cfg.paths.tables / "DATASETS_SNAPSHOT.json"
out.write_text(json.dumps(snapshot, indent=2, default=str))
print("wrote", out)
print(tr.summary())
"""),
    md("""
### Next
`02_taxonomy_audit.ipynb` — taxonomy master, label map, duplicate removal, label-noise
detection, review pack. Nothing is trained before that audit passes.
"""),
]


# =========================================================================== #
# 02 - taxonomy + audit
# =========================================================================== #
nb02 = setup_cells("02_taxonomy_audit", require_gpu=True) + [
    md("""
## 1 · Load the classification manifest from notebook 01"""),
    code("""
# @title 1a · Manifest (rebuild from disk if notebook 01 was skipped)
import pandas as pd
from src.data.discovery import discover_classification

manifest_path = cfg.paths.data_curated / "all_classification.csv"
if manifest_path.exists():
    df_cls = pd.read_csv(manifest_path)
else:
    df_cls = discover_classification(cfg)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df_cls.to_csv(manifest_path, index=False)
print(len(df_cls), "images |", df_cls.dataset_label.nunique(), "classes")
display(df_cls.dataset.value_counts().to_frame("images"))
"""),
    md("""
## 2 · `taxonomy_master.csv` + `label_map.csv`
Two different jobs, deliberately kept apart:

* **taxonomy_master.csv** — the species/family reference (scientific names + hierarchy).
  Built from a mounted taxonomy source (a FishBase/WoRMS export if you attach one). If no
  taxonomy source is mounted, the file is written with the columns present and
  `source="not_available"` rows for every dataset label, and the report says so — it is
  never filled with invented names.
* **label_map.csv** — dataset label → accepted name, with an explicit `confidence` and
  `method`. Only `high`-confidence mappings are *applied*; everything else is reported and
  quarantined.

Note: a resolved scientific name is **metadata**. It rewrites the training label only when
the audit evidence says the folder contains a different fish than it claims.
"""),
    code("""
# @title 2a · Build taxonomy_master.csv from a mounted taxonomy export (optional input)
import re
TAXONOMY_DIR = cfg.paths.data_taxonomy
TAXONOMY_DIR.mkdir(parents=True, exist_ok=True)
tax_path = TAXONOMY_DIR / "taxonomy_master.csv"

key = "taxonomy.master"
if tr.is_done(key) and tax_path.exists():
    tax = pd.read_csv(tax_path)
    print("resumed:", len(tax), "rows")
else:
    with stage(tr, key) as st:
        if not st.skip:
            sources = {}
            # FishDex-style export: fishdex_sharded/genera/<family>/<genus>.csv
            for mount in cfg.paths.input_root.glob("*/fishdex_sharded"):
                rows = []
                for f in sorted(mount.glob("genera/*/*.csv")):
                    import csv as _csv
                    with open(f, newline="") as fh:
                        for r in _csv.DictReader(fh):
                            sci = (r.get("ScientificName") or "").strip()
                            rows.append({"accepted_name": sci,
                                         "common_names": r.get("CommonName"),
                                         "genus": sci.split()[0] if sci else "",
                                         "family": f.parent.name.capitalize(),
                                         "source": f"FishDex export ({f.parent.name})",
                                         "spec_code": r.get("SpecCode"),
                                         "habitat": (r.get("Environments") or "").strip("[]").replace("'", "")})
                if rows:
                    sources["fishdex"] = rows
            if sources:
                tax = pd.DataFrame(sources["fishdex"]).dropna(subset=["accepted_name"])
                tax = tax.drop_duplicates("accepted_name")
            else:
                cols = ["species_id", "accepted_name", "authority", "family", "genus",
                        "common_names", "local_names", "habitat", "states", "iucn_status",
                        "source", "synonyms", "spec_code"]
                tax = pd.DataFrame(columns=cols)
                print("No taxonomy export mounted -> writing an empty, correctly shaped "
                      "taxonomy_master.csv. Attach FishBase/WoRMS export as a Kaggle "
                      "dataset to fill it.")
            tax.to_csv(tax_path, index=False)
            st.out(tax_path)
            st.metric(rows=int(len(tax)))
            print(len(tax), "taxonomy rows")
"""),
    code("""
# @title 2b · Label map: every dataset folder -> accepted name, with confidence
import pandas as pd

# Explicit, reviewed mappings. `confidence` decides whether a row is applied.
# Bengali/market names are resolved to the scientific name of the species that is
# normally sold under that name; ambiguous names are marked `low` on purpose.
MANUAL_MAP = [
    # Bengali market names (BDIndigenousFish2019 proxy for AS-FishFresh)
    ("Koi",      "Anabas testudineus",     "high",   "standard Bengali name for climbing perch"),
    ("Sol",      "Channa striata",         "high",   "standard Bengali name for striped snakehead"),
    ("Sing",     "Heteropneustes fossilis","medium", "'Singhi' stinging catfish - genus-level match only"),
    ("Sorputi",  "Systomus sarana",        "medium", "'Sarputi' olive barb - regional name overlap"),
    ("Foli",     "Notopterus notopterus",  "medium", "'Pholi' bronze featherback"),
    ("Tengra",   "Mystus vittatus",        "low",    "'Tengra' covers several Mystus species"),
    ("Taki",     "Channa punctata",        "low",    "'Taki' also used for other Channa"),
    ("Byen",     None,                     "unresolved", "no verifiable match"),
    # Indian Seafood 8-folder audit (spec 4.2) - applied when the dataset is mounted
    ("Pomfret",            "Pampus argenteus",     "high",   "spelling fix: Pampus argeneus -> P. argenteus"),
    ("Mackerel",           "Rastrelliger kanagurta","medium", "Indian 'Bangda'; requires visual confirmation"),
    ("Black Snapper",      "Oreochromis niloticus","low",    "local name 'Tilapia' contradicts Apsilus dentatus"),
    ("Indian Carp",        "Labeo rohita",         "low",    "local name 'Rohu' contradicts Cyprinus carpio"),
    ("Prawn",              "Penaeoidea_prawn",     "high",   "Dendrobranchiata is a suborder, not a species"),
    ("Pink Perch",         "Nemipterus japonicus", "low",    "Indian 'Rani' is normally N. japonicus"),
    ("Indian flathead",    "Lates calcarifer",     "medium", "keep the scientific name, fix the English name"),
    ("Black Pomfret",      "Parastromus niger",    "high",   "printed name is correct"),
    # Kaggle 9-folder seafood copy (folder names are English common names)
    ("black_sea_sprat",    "Sprattus sprattus",       "medium", "common name; may be Clupeonella in the source imagery"),
    ("gilt_head_bream",    "Sparus aurata",           "high",   "unambiguous common name"),
    ("hourse_mackerel",    "Trachurus trachurus",     "high",   "unambiguous common name"),
    ("red_mullet",         "Mullus barbatus",         "high",   "unambiguous common name"),
    ("red_sea_bream",      "Pagrus major",            "high",   "unambiguous common name"),
    ("sea_bass",           "Dicentrarchus labrax",    "high",   "unambiguous common name"),
    ("shrimp",             "Penaeoidea_prawn",        "high",   "unresolved genus, resolved to prawn node"),
    ("striped_red_mullet", "Mullus surmuletus",       "high",   "unambiguous common name"),
    ("trout",              "Oncorhynchus mykiss",     "medium", "could be Salmo trutta or O. mykiss"),
    ("animal fish",        None,                      "unresolved", "not a species folder"),
    ("animal fish bass",   "Dicentrarchus labrax",    "low",    "ambiguous, tiny folder"),
]

key = "taxonomy.label_map"
lm_path = TAXONOMY_DIR / "label_map.csv"
if tr.is_done(key) and lm_path.exists():
    label_map = pd.read_csv(lm_path)
    print("resumed:", len(label_map), "rows")
else:
    with stage(tr, key) as st:
        if not st.skip:
            manual = pd.DataFrame(MANUAL_MAP, columns=["folder", "accepted_name", "confidence", "notes"])
            manual["key"] = [match_label(f, [str(x) for x in manual.folder]).key
                             or str(f).strip().lower().replace(" ", "_") for f in manual.folder]
            by_key = manual.drop_duplicates("key").set_index("key")
            rows = []
            for ds, folder in (df_cls.groupby("dataset")["label"].unique().items()):
                for f in sorted(folder):
                    # mirrors spell folders differently ("fish sea_food black_sea_sprat",
                    # "Black Sea Sprat", "gilt-head bream") -> normalise, then match,
                    # and record HOW it matched so the reviewer sees the confidence
                    m = match_label(f, list(by_key.index))
                    if m.ok and m.key in by_key.index:
                        r = by_key.loc[m.key]
                        conf = r.confidence
                        if m.method != "exact" and conf == "high":
                            conf = "medium"        # fuzzy match cannot stay high
                        rows.append({"dataset": ds, "original_label": f,
                                     "accepted_name": r.accepted_name if pd.notna(r.accepted_name) else "",
                                     "confidence": conf,
                                     "method": f"manual_reviewed:{m.method}",
                                     "notes": r.notes,
                                     "applied": conf == "high"})
                    elif re.fullmatch(r"[A-Z][a-z]+ [a-z]+", str(f)) or "_" in str(f):
                        cand = str(f).replace("_", " ")
                        rows.append({"dataset": ds, "original_label": f,
                                     "accepted_name": cand, "confidence": "medium",
                                     "method": "already_binomial",
                                     "notes": "folder already looks like a binomial; pending taxonomy validation",
                                     "applied": False})
                    else:
                        rows.append({"dataset": ds, "original_label": f, "accepted_name": "",
                                     "confidence": "unresolved", "method": "unmatched",
                                     "notes": "no reviewed mapping", "applied": False})
            label_map = pd.DataFrame(rows)
            label_map.to_csv(lm_path, index=False)
            st.out(lm_path)
            st.metric(rows=int(len(label_map)),
                      applied=int(label_map.applied.sum()))
            print(label_map.confidence.value_counts().to_dict())
display(label_map.head(20))
"""),
    md("""
## 3 · Embeddings (DINOv2, pretrained)
Computed **once** and cached — used for duplicate detection, for the split grouping and for
the out-of-fold label-noise pass. Resumes if the cache exists and the row count matches.
"""),
    code("""
# @title 3a · Extract / load DINOv2 embeddings
import numpy as np
from src.data.embed import embed_paths

key = "audit.embeddings"
emb_path = cfg.paths.data_interim / "embeddings.npy"
idx_path = cfg.paths.data_interim / "embedding_index.json"
if emb_path.exists() and idx_path.exists():
    import json
    idx = json.loads(idx_path.read_text())
    if idx["rows"] == df_cls.path.tolist():
        emb = np.load(emb_path)
        print("loaded cached embeddings", emb.shape)
    else:
        emb = None
else:
    emb = None

if emb is None:
    with stage(tr, key) as st:
        if not st.skip:
            emb = embed_paths(df_cls.path.tolist(),
                              arch=cfg.raw["pretrained"]["embeddings"]["name"],
                              batch_size=64)
            import json
            np.save(emb_path, emb.astype(np.float32))
            idx_path.write_text(json.dumps({"rows": df_cls.path.tolist(),
                                            "arch": cfg.raw["pretrained"]["embeddings"]["name"],
                                            "dim": int(emb.shape[1])}))
            st.out(emb_path, idx_path)
            st.metric(n=int(len(emb)), dim=int(emb.shape[1]))
            print("embedded", emb.shape)
"""),
    md("""
## 4 · Duplicates (spec 4.3.1)
Exact (sha256) + pHash + pretrained-embedding cosine. Photos that are near-duplicates are
grouped and later kept on one side of the split.
"""),
    code("""
# @title 4a · Duplicate clusters
import hashlib, numpy as np, pandas as pd
from src.data.embed import cosine_duplicate_pairs, union_find

key = "audit.duplicates"
dup_path = cfg.paths.tables / "duplicates.csv"
if tr.is_done(key) and dup_path.exists():
    dups = pd.read_csv(dup_path)
    print("resumed:", len(dups), "duplicate rows")
else:
    with stage(tr, key) as st:
        if not st.skip:
            thr = float(cfg.raw["pretrained"]["embeddings"]["cosine_duplicate_threshold"])
            edges = [(i, j, s, "embedding")
                     for i, j, s in cosine_duplicate_pairs(emb, threshold=thr)]
            sha = df_cls.path.map(lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
            for _, grp in df_cls.assign(sha=sha).groupby("sha").indices.items():
                for k in range(1, len(grp)):
                    edges.append((int(grp[0]), int(grp[k]), 1.0, "exact"))
            groups = union_find(len(df_cls), [(i, j, s) for i, j, s, _ in edges])
            df_cls = df_cls.assign(group=groups)
            dups = pd.DataFrame([{"path": df_cls.path.iloc[i], "group": int(groups[i]),
                                  "method": m, "score": s} for i, j, s, m in edges])
            if len(dups) == 0:
                dups = pd.DataFrame(columns=["path", "group", "method", "score"])
            dups.to_csv(dup_path, index=False)
            st.out(dup_path)
            st.metric(n_edges=int(len(dups)), n_groups=int(len(set(groups))))
            print(len(dups), "duplicate edges,", len(set(groups)), "photo groups")
if "group" not in df_cls:
    df_cls = df_cls.assign(group=0)
"""),
    md("## 5 · Image quality (spec 4.3.4)"),
    code("""
# @title 5a · Laplacian blur + size filter (never deletes anything)
from src.audit.pipeline import quality_report

key = "audit.quality"
q_path = cfg.paths.tables / "image_quality.csv"
if tr.is_done(key) and q_path.exists():
    quality = pd.read_csv(q_path)
    print("resumed:", len(quality))
else:
    with stage(tr, key) as st:
        if not st.skip:
            qcfg = cfg.raw["audit"]["quality"]
            quality = quality_report(df_cls.path.tolist(),
                                     laplacian_threshold=float(qcfg["laplacian_var_threshold"]),
                                     min_short_side=int(qcfg["min_short_side"]))
            quality.to_csv(q_path, index=False)
            st.out(q_path)
            st.metric(unreadable=int((~quality.ok).sum()), blurry=int(quality.blurry.sum()),
                      too_small=int(quality.too_small.sum()))
            print(quality[["blurry", "too_small"]].sum().to_dict())
"""),
    md("""
## 6 · Label-noise detection (cleanlab on out-of-fold probabilities)
The OOF probabilities come from a logistic probe on the frozen DINOv2 features — fast, and
it never touches the test split.
"""),
    code("""
# @title 6a · Out-of-fold probabilities + cleanlab
import numpy as np
from src.audit.pipeline import oof_pred_probs, find_label_issues

key = "audit.label_noise"
sus_path = cfg.paths.tables / "suspect_labels.csv"
labels_arr = df_cls.dataset_label.values
if tr.is_done(key) and sus_path.exists():
    suspects = pd.read_csv(sus_path)
    print("resumed:", len(suspects), "suspects")
else:
    with stage(tr, key) as st:
        if not st.skip:
            probs = oof_pred_probs(emb, labels_arr, folds=5, seed=cfg.seed)
            suspects = find_label_issues(probs, labels_arr)
            if len(suspects):
                suspects["path"] = suspects["index"].map(lambda i: df_cls.path.iloc[int(i)])
            suspects.to_csv(sus_path, index=False)
            np.save(cfg.paths.data_interim / "oof_probs.npy", probs)
            st.out(sus_path)
            st.metric(n_suspects=int(len(suspects)))
            print("cleanlab flagged", len(suspects))
"""),
    md("""
## 7 · BioCLIP zero-shot cross-check (spec 4.3.3)
Required checkpoint: if it cannot be downloaded the cell raises — it does not quietly
degrade the audit.
"""),
    code("""
# @title 7a · Zero-shot agreement with the folder label
import numpy as np, pandas as pd
from src.audit.pipeline import bioclip_zero_shot

key = "audit.bioclip"
candidates = sorted(df_cls.dataset_label.unique())
bioclip_status = {"available": False, "reason": "not run"}
if tr.is_done(key):
    bioclip_status = tr.get(key).metrics
    print("resumed:", bioclip_status)
else:
    with stage(tr, key) as st:
        if not st.skip:
            scores = bioclip_zero_shot(df_cls.path.tolist(), candidates, cfg, batch_size=64)
            np.save(cfg.paths.data_interim / "bioclip_scores.npy", scores)
            pred = [candidates[int(i)] for i in scores.argmax(1)]
            agree = float(np.mean([p == t for p, t in zip(pred, df_cls.dataset_label)]))
            bioclip_status = {"available": True, "n_images": int(len(scores)),
                              "agreement_rate": agree,
                              "n_disagreements": int(len(scores) * (1 - agree))}
            pd.DataFrame({"path": df_cls.path, "given": df_cls.dataset_label,
                          "bioclip_pred": pred,
                          "bioclip_conf": scores.max(1)}).to_csv(
                cfg.paths.tables / "bioclip_zeroshot.csv", index=False)
            st.metric(**bioclip_status)
            print(bioclip_status)
"""),
    md("""
## 8 · Relabel decisions: apply only on full agreement
A relabel is applied **only** when cleanlab flags the image *and* every available
substitute opinion (kNN on DINOv2 features, BioCLIP) names the same new class *and* the
confidence clears the threshold. Everything else is quarantined for you to review.
"""),
    code("""
# @title 8a · Decisions + quarantine + review pack
import numpy as np, pandas as pd
from src.audit.pipeline import decide_relabels, knn_third_opinion, write_quarantine, write_review_pack

key = "audit.decisions"
dec_path = cfg.paths.tables / "relabel_decisions.csv"
if tr.is_done(key) and dec_path.exists():
    decisions = pd.read_csv(dec_path)
    print("resumed:", len(decisions))
else:
    with stage(tr, key) as st:
        if not st.skip:
            if len(suspects) == 0:
                decisions = pd.DataFrame(columns=["path", "given_label", "decision", "new_label"])
            else:
                knn = knn_third_opinion(emb, labels_arr, k=15)
                bio_scores = (np.load(cfg.paths.data_interim / "bioclip_scores.npy")
                              if (cfg.paths.data_interim / "bioclip_scores.npy").exists()
                              else None)
                decisions = decide_relabels(
                df_cls.reset_index(drop=True), suspects, knn, bio_scores, candidates,
                min_confidence=float(cfg.raw["audit"]["label_noise"]
                                     ["min_confidence_to_apply_relabel"]))
            decisions.to_csv(dec_path, index=False)
            st.out(dec_path)
            st.metric(relabels=int((decisions.decision == "relabel").sum()),
                      quarantined=int((decisions.decision == "quarantine").sum()))
            print(decisions.decision.value_counts().to_dict() if len(decisions) else "none")
"""),
    code("""
# @title 8b · Quarantine + human review pack (contact sheets)
q = write_quarantine(decisions, quality, cfg.paths.quarantine,
                     unresolved=label_map[~label_map.applied][
                         ["original_label", "confidence"]].rename(
                         columns={"original_label": "path"}).to_dict("records"))
review = write_review_pack(
    df_cls, decisions, cfg.paths.review,
    images_per_class=int(cfg.raw["audit"]["review_pack"]["images_per_class"]),
    thumb=int(cfg.raw["audit"]["review_pack"]["thumb_px"]),
    cols=int(cfg.raw["audit"]["review_pack"]["grid_cols"]))
print(q); print(review)
"""),
    md("""
## 9 · Approved training manifest + `AUDIT_REPORT.md`
`approved_labels.csv` is the **only** file notebook 04 reads for labels: the audit output,
minus quarantined paths, with the applied relabels and the taxonomy metadata attached.
"""),
    code("""
# @title 9a · Approved labels
import pandas as pd
approved = df_cls.copy()
if len(decisions):
    rel = decisions[decisions.decision == "relabel"].set_index("path")["new_label"]
    approved["dataset_label"] = approved["path"].map(rel).fillna(approved["dataset_label"])
qpaths = set(pd.read_csv(cfg.paths.quarantine / "quarantine_index.csv").path.astype(str))
approved["quarantined"] = approved.path.astype(str).isin(qpaths)
approved = approved[~approved.quarantined]
approved = approved.merge(label_map[["dataset", "original_label", "accepted_name",
                                     "confidence", "applied"]],
                          left_on=["dataset", "label"],
                          right_on=["dataset", "original_label"], how="left")
approved.drop(columns=["original_label"], inplace=True)
approved.to_csv(cfg.paths.data_curated / "approved_labels.csv", index=False)
print(len(approved), "approved rows |", approved.dataset_label.nunique(), "classes")
display(approved.groupby("dataset").size().to_frame("approved_images"))
"""),
    code("""
# @title 9b · AUDIT_REPORT.md
import datetime
from collections import Counter
from src.audit.report import write_audit_report

before = Counter(df_cls.dataset_label)
after = Counter(approved.dataset_label)
report = write_audit_report(
    cfg.paths.outputs / "AUDIT_REPORT.md",
    config_snapshot={"config_path": str(cfg.path), "seed": cfg.seed,
                     "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
                     "approved_manifest": str(cfg.paths.data_curated / "approved_labels.csv")},
    before_counts=dict(before), after_counts=dict(after),
    duplicates=(dups if "dups" in dir() else None),
    suspicious=suspects, decisions=decisions, quarantine=q, quality=quality,
    label_map=label_map, bioclip_status=bioclip_status,
    seafood_mounted=any("seafood" in d for d in df_cls.dataset.unique()),
    extra={"n_train_rows": int(len(approved)),
           "not_computed": [
               "Indian Seafood 8-folder visual verdicts (dataset not mounted)",
               "Fig.1 vs text count discrepancy 9,007 vs 8,488 (images unavailable)",
               "coverage vs the FishBase India checklist (checklist source unavailable)",
           ]})
print(report.read_text()[:2000])
"""),
    md("### Next\n`03_detector.ipynb` — detector training, with this audit already in place."),
]


# =========================================================================== #
# 03 - detector
# =========================================================================== #
nb03 = setup_cells("03_detector", require_gpu=True) + [
    md("""
## 1 · Single-class detection dataset
Rebuilt here if notebook 01 did not run. Every source label space is collapsed to `fish`.
"""),
    code("""
# @title 1a · Dataset + splits
from src.data.split import build_detection_splits
from src.data.discovery import build_single_class_yolo

det_dir = cfg.paths.data_interim / "detection_single_class"
if not (det_dir / "images").exists():
    print(build_single_class_yolo(cfg, det_dir))
split_dir = cfg.paths.data_splits / "detection"
if not (split_dir / "train_images.txt").exists():
    print(build_detection_splits(sorted((det_dir / "images").glob("*")), split_dir,
                                  seed=cfg.seed))
data_yaml = det_dir / "data.yaml"
print(data_yaml.read_text())
"""),
    md("""
## 2 · Train the detector zoo
YOLO11 (n/s/m), YOLOv12-s and RT-DETR-l, all from **pretrained checkpoints**, single class.
Each model is its own resumable stage: an interrupted run continues from `last.pt`
(Ultralytics `resume=True`) instead of restarting.
"""),
    code("""
# @title 2a · Train (or resume) every configured detector
import json, shutil
from pathlib import Path
from src.utils.pretrained import ultralytics_model, PretrainedDownloadError

results = {}
for name in cfg.raw["detector"]["models"]:
    weights = cfg.raw["pretrained"]["detectors"][name]
    key = f"detector.train.{name}"
    run_dir = Path(cfg.paths.runs) / f"det_{name}"
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if tr.is_done(key):
        print(f"{name}: already trained ->", tr.get(key).metrics)
        results[name] = tr.get(key).metrics
        continue
    with stage(tr, key) as st:
        if st.skip:
            continue
        model = ultralytics_model(weights)          # raises if weights are unavailable
        resume = bool(last.exists() and cfg.raw["resume"]["enabled"])
        print(f"--- {name}: {'RESUMING from ' + str(last) if resume else 'fresh start'}")
        train_kwargs = dict(
            data=str(data_yaml), epochs=cfg.raw["detector"]["epochs"],
            imgsz=cfg.raw["detector"]["imgsz"], batch=cfg.raw["detector"]["batch"],
            patience=cfg.raw["detector"]["patience"], project=str(cfg.paths.runs),
            name=f"det_{name}", exist_ok=True, seed=cfg.seed, amp=cfg.raw.get("amp", True),
            plots=True, cos_lr=True, close_mosaic=10)
        aug = cfg.raw["detector"]["augment"]
        train_kwargs.update({k: aug[k] for k in ("mosaic", "hsv_h", "hsv_s", "hsv_v", "fliplr")
                             if k in aug})
        if resume:
            train_kwargs["resume"] = True
        model.train(**train_kwargs)
        assert best.exists(), f"{name}: training finished but no best.pt"
        dest = Path(cfg.paths.checkpoints) / f"det_{name}_best.pt"
        shutil.copy2(best, dest)
        metrics = {"checkpoint": str(dest), "weights_origin": weights}
        results[name] = metrics
        st.out(dest, run_dir / "results.csv")
        st.checkpoint(dest)
        st.metric(**{f"best_pt": 1})
        print(f"{name} done ->", dest)
json.dumps(results, indent=2)
"""),
    md("""
### Optional: extra underwater augmentations
Spec 6 asks for motion blur, low light and turbidity. Ultralytics handles mosaics/HSV;
these three are added as a dataset-level pre-pass into a shadow copy of the training
images (kept separate so the evaluation set is never altered).
"""),
    code("""
# @title 2b · Turbidity / low-light / motion-blur augment pass (train split only)
import cv2, numpy as np, random
from pathlib import Path

key = "detector.augment_pass"
aug_dir = cfg.paths.data_interim / "detection_augmented"
if tr.is_done(key):
    print("resumed:", tr.get(key).metrics)
else:
    with stage(tr, key) as st:
        if not st.skip:
            rng = random.Random(cfg.seed)
            train_list = [Path(p) for p in
                          (split_dir / "train_images.txt").read_text().split("\\n") if p]
            aug_root = aug_dir / "images"; aug_root.mkdir(parents=True, exist_ok=True)
            (aug_dir / "labels").mkdir(parents=True, exist_ok=True)
            made = 0
            for p in train_list:
                img = cv2.imread(str(p))
                if img is None:
                    continue
                if rng.random() < 0.5:
                    h, w = img.shape[:2]
                    out = img.astype(np.float32)
                    if rng.random() < 0.4:      # turbidity haze
                        out = out * 0.75 + np.array([120, 140, 135], dtype=np.float32) * 0.25
                    if rng.random() < 0.4:      # low light
                        out = (out / 255.0) ** 1.8 * 255.0
                    if rng.random() < 0.3:      # motion blur
                        k = rng.choice([5, 9, 13])
                        kernel = np.zeros((k, k)); kernel[k // 2, :] = 1.0 / k
                        out = cv2.filter2D(out, -1, kernel)
                    out = np.clip(out, 0, 255).astype(np.uint8)
                    cv2.imwrite(str(aug_root / p.name), out)
                    lab = det_dir / "labels" / f"{p.stem}.txt"
                    if lab.exists():
                        (aug_dir / "labels" / f"{p.stem}.txt").write_bytes(lab.read_bytes())
                    made += 1
            st.metric(augmented_images=made)
            print(f"augmented {made} training images -> {aug_dir}")
"""),
    md("""
## 3 · Evaluate on the held-out detection split (own COCO-style evaluator)
Precision / recall / F1 at the best-F1 confidence, mAP@50, mAP@50-95, PR curve and
F1-confidence curve — the same evaluator for YOLO11, YOLOv12 and RT-DETR, so the
comparison is apples-to-apples.
"""),
    code("""
# @title 3a · Score every trained detector on the val split
import numpy as np, pandas as pd
from pathlib import Path
from src.utils.pretrained import ultralytics_model
from src.eval.detection_metrics import evaluate_detection, save_detection_metrics, plot_pr_and_f1

def load_gt(img_paths, labels_dir):
    out = []
    for p in img_paths:
        lab = Path(labels_dir) / f"{Path(p).stem}.txt"
        boxes = []
        if lab.exists():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    _, xc, yc, bw, bh = parts[:5]
                    boxes.append([float(xc), float(yc), float(bw), float(bh)])
        out.append(np.array(boxes, dtype=float).reshape(-1, 4))
    return out

def to_xyxy(norm_boxes, w, h):
    if len(norm_boxes) == 0:
        return np.zeros((0, 4))
    b = norm_boxes.copy()
    b[:, 0] = (b[:, 0] - b[:, 2] / 2) * w
    b[:, 1] = (b[:, 1] - b[:, 3] / 2) * h
    b[:, 2] = (b[:, 0] + b[:, 2] * w); b[:, 3] = (b[:, 1] + b[:, 3] * h)
    return b

val_imgs = [Path(p) for p in (split_dir / "val_images.txt").read_text().split("\\n") if p]
gt_norm = load_gt(val_imgs, det_dir / "labels")

all_metrics = {}
for name in cfg.raw["detector"]["models"]:
    ckpt = Path(cfg.paths.checkpoints) / f"det_{name}_best.pt"
    if not ckpt.exists():
        print(f"{name}: no checkpoint, skipped"); continue
    key = f"detector.val.{name}"
    if tr.is_done(key):
        all_metrics[name] = tr.get(key).metrics
        print(f"{name}: resumed", all_metrics[name]); continue
    with stage(tr, key) as st:
        if st.skip:
            continue
        model = ultralytics_model(str(ckpt))
        preds, gts = [], []
        for p, gt in zip(val_imgs, gt_norm):
            import cv2
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            r = model.predict(str(p), conf=0.001, iou=0.6, verbose=False)[0]
            preds.append({"boxes": r.boxes.xyxy.cpu().numpy() if r.boxes is not None
                          else np.zeros((0, 4)),
                          "scores": r.boxes.conf.cpu().numpy() if r.boxes is not None
                          else np.zeros(0)})
            gts.append({"boxes": to_xyxy(gt, w, h)})
        m = evaluate_detection(preds, gts)
        save_detection_metrics(m, cfg.paths.metrics, f"detection_{name}",
                               meta={"dataset": sorted(det_dir.glob("data.yaml"))[0].parent.name,
                                     "split": "val", "n_images": len(val_imgs),
                                     "weights": cfg.raw["pretrained"]["detectors"][name],
                                     "seed": cfg.seed})
        all_metrics[name] = {k: v for k, v in m.items() if isinstance(v, (int, float))}
        st.metric(**all_metrics[name])
        print(name, all_metrics[name])
display(pd.DataFrame(all_metrics).T)
"""),
    md("## 4 · Comparison table + curves"),
    code("""
# @title 4a · Write the model-comparison table and the PR/F1 figure
import json, pandas as pd
comp = pd.DataFrame(all_metrics).T
comp.index.name = "model"
comp.to_csv(cfg.paths.metrics / "detection_model_comparison.csv")
if len(all_metrics):
    curves = [json.loads((cfg.paths.metrics / f"detection_{n}.json").read_text())["metrics"]
              for n in all_metrics]
    plot_pr_and_f1(curves, list(all_metrics),
                   cfg.paths.figures / "detection_pr_f1.png")
display(comp[[c for c in comp.columns if c in
              ("precision_at_best_f1", "recall_at_best_f1", "f1_at_best_f1",
               "map50", "map50_95")]])
"""),
    md("""
## 5 · Export detector crops for the classifier (cross-domain ablation input)
"""),
    code("""
# @title 5a · Crop classification images with the best detector
import cv2, numpy as np, pandas as pd
from pathlib import Path
from src.utils.pretrained import ultralytics_model

key = "detector.export_crops"
crop_dir = cfg.paths.data_interim / "detector_crops"
if tr.is_done(key):
    print("resumed:", tr.get(key).metrics)
else:
    with stage(tr, key) as st:
        if not st.skip:
            best_name = max(all_metrics, key=lambda n: all_metrics[n].get("map50", 0))
            model = ultralytics_model(str(Path(cfg.paths.checkpoints) / f"det_{best_name}_best.pt"))
            approved = pd.read_csv(cfg.paths.data_curated / "approved_labels.csv")
            crop_dir.mkdir(parents=True, exist_ok=True)
            rows = []
            for _, r in approved.groupby("dataset_label").head(300).iterrows():
                res = model.predict(r.path, conf=0.25, verbose=False)
                if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                    continue
                b = res[0].boxes.xyxy.cpu().numpy()[0]
                rows.append({"path": r.path, "dataset_label": r.dataset_label,
                             "x1": float(b[0]), "y1": float(b[1]),
                             "x2": float(b[2]), "y2": float(b[3]),
                             "score": float(res[0].boxes.conf.cpu().numpy()[0])})
            pd.DataFrame(rows).to_csv(crop_dir / "crops.csv", index=False)
            st.out(crop_dir / "crops.csv")
            st.metric(crops=int(len(rows)), detector=best_name)
            print(f"{len(rows)} crops from {best_name}")
"""),
    md("""
### Next
`04_classifier.ipynb` — species classifier on GT and on detector crops.
The detector **test** split is evaluated once, in notebook 04's final comparison run, not here.
"""),
]


# =========================================================================== #
# 04 - classifier
# =========================================================================== #
nb04 = setup_cells("04_classifier", require_gpu=True) + [
    md("""
## 1 · Audit-approved labels + stratified heads
"""),
    code("""
# @title 1a · Load approved labels and build taxonomy heads
import numpy as np, pandas as pd
from src.models.classifier import heads_from_approved, family_index_of_species
from src.data.discovery import discover_classification

approved_path = cfg.paths.data_curated / "approved_labels.csv"
if not approved_path.exists():
    raise FileNotFoundError("run 02_taxonomy_audit.ipynb first - the classifier only "
                            "trains on audit-approved labels")
df = pd.read_csv(approved_path)
print(len(df), "approved images |", df.dataset_label.nunique(), "classes")

# Three-level head from the audit output. A genus is only used when the audit
# established a binomial; otherwise the dataset label is its own node, so the
# hierarchy never invents taxonomy.
heads = heads_from_approved(df)
classes = heads.species
print(len(heads.species), "species |", len(heads.genera), "genera |", len(heads.families), "families")
"""),
    md("""
## 2 · Leakage-safe splits (frozen, hashed)
Copied from `02`'s grouping: duplicate photos never straddle train/test, and classes with
fewer than `rare_species_min_images` go to a rare-species manifest used for few-shot
reporting instead of training.
"""),
    code("""
# @title 2a · Build (or load) the frozen split manifests
from src.data.split import build_splits, verify_lock
splits_dir = cfg.paths.data_splits / "classification"
if (splits_dir / "SPLIT_LOCK.json").exists():
    lock = verify_lock(splits_dir)
    print("split lock verified:", {k: v["n"] for k, v in lock["files"].items()})
else:
    key = "classifier.splits"
    with stage(tr, key) as st:
        if not st.skip:
            summary = build_splits(cfg, df, splits_dir, embed_batch=64)
            st.out(splits_dir / "SPLIT_LOCK.json")
            st.metric(**{f"n_{k}": v["n"] for k, v in summary["files"].items()})
            print(summary["split_sizes"], "| rare classes:", len(summary["rare_classes"]))
train_df = pd.read_csv(splits_dir / "train.csv")
val_df = pd.read_csv(splits_dir / "val.csv")
test_df = pd.read_csv(splits_dir / "test.csv")
print(len(train_df), len(val_df), len(test_df))
"""),
    md("## 3 · Datasets and transforms"),
    code("""
# @title 3a · DataLoaders (train-only augmentation incl. turbidity/low-light proxies)
from src.data.dataset import ManifestDataset, build_transforms, make_loader

size = int(cfg.raw["classifier"]["input_size"])
class_to_idx = {c: i for i, c in enumerate(heads.species)}
loaders = {}
for split, loader_df in (("train", train_df), ("val", val_df), ("test", test_df)):
    ds = ManifestDataset(loader_df, class_to_idx,
                         build_transforms(size, train=(split == "train")))
    loaders[split] = make_loader(ds, cfg.raw["classifier"]["batch"], train=(split == "train"),
                                 workers=4, seed=cfg.seed)
    print(f"{split:5s} {len(ds):5d} images {len(loaders[split]):3d} batches")
"""),
    md("""
## 4 · Train the classifier zoo
ResNet50, EfficientNetV2-S, ConvNeXt-T/S, ViT-B/16, DINOv2-S — all from pretrained
checkpoints, two-stage (head → full with layer-wise LR decay), hierarchical loss
(species + genus + family), class-balanced CE, label smoothing, Mixup/CutMix.
Every model is a resumable stage.
"""),
    code("""
# @title 4a · Train / resume all configured architectures
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from src.models.classifier import build_classifier
from src.train.classifier_train import fit, predict_proba

def val_eval(model, loader, device):
    logits, y = predict_proba(model, loader, device)
    pred = logits.argmax(1)
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    return {"val_acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y, pred))}

trained = {}
for name in cfg.raw["classifier"]["models"]:
    arch = cfg.raw["pretrained"]["classifiers"][name]
    key = f"classifier.train.{name}"
    ckpt = Path(cfg.paths.checkpoints)
    best = ckpt / f"cls_{name}_best.pt"
    if best.exists() and tr.is_done(key):
        trained[name] = tr.get(key).metrics
        print(f"{name}: done ->", trained[name]); continue
    with stage(tr, key) as st:
        if st.skip:
            continue
        model = build_classifier(arch, heads, image_size=size)   # pretrained or raises
        res = fit(cfg, model, loaders, heads, ckpt, run_name=f"cls_{name}",
                  tracker=tr, stage_key=key, eval_fn=val_eval)
        trained[name] = res
        st.out(res["checkpoint"])
        st.checkpoint(res["checkpoint"])
        st.metric(**{k: v for k, v in res.items() if isinstance(v, (int, float))})
        print(f"{name}: best val macro-F1 {res['best_macro_f1']:.4f}")
"""),
    md("""
## 5 · Full metrics on validation for every model
Accuracy, top-5, macro/weighted P/R/F1, balanced accuracy, ECE, per-class report,
confusion matrices (raw + row-normalised), family-level matrix, top-20 confused pairs.
"""),
    code("""
# @title 5a · Validation metrics + confusion matrices
import json, numpy as np, pandas as pd
from src.eval.metrics import classification_metrics, plot_confusion, plot_per_class_f1, save_metrics
from src.models.classifier import build_classifier
from src.train.classifier_train import predict_proba
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
val_metrics = {}
for name in trained:
    ckpt = Path(cfg.paths.checkpoints) / f"cls_{name}_best.pt"
    if not ckpt.exists():
        continue
    model = build_classifier(cfg.raw["pretrained"]["classifiers"][name], heads, image_size=size)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"]); model.to(device)
    logits, y = predict_proba(model, loaders["val"], device)
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    m = classification_metrics(y, probs, heads.species,
                               family_of_class=family_index_of_species(heads),
                               family_names=heads.families)
    val_metrics[name] = m
    save_metrics(m, cfg.paths.metrics, f"classifier_{name}_val",
                 meta={"split": "val", "arch": cfg.raw["pretrained"]["classifiers"][name],
                       "n_images": int(len(val_df)), "seed": cfg.seed,
                       "pretrained": True})
    cm = np.array(m["confusion_matrix"])
    plot_confusion(cm, heads.species, cfg.paths.figures / f"cm_{name}_val.png",
                   normalize=False, title=f"{name} - validation")
    plot_confusion(cm, heads.species, cfg.paths.figures / f"cm_{name}_val_norm.png",
                   normalize=True, title=f"{name} - validation (row-normalised)")
    plot_per_class_f1(m, cfg.paths.figures / f"per_class_f1_{name}.png", title=name)
    print(f"{name}: acc={m['accuracy']:.4f} f1_macro={m['f1_macro']:.4f} ece={m['ece']:.3f}")
pd.DataFrame({k: {kk: vv for kk, vv in v.items() if isinstance(vv, (int, float))}
              for k, v in val_metrics.items()}).T.to_csv(
    cfg.paths.metrics / "classifier_model_comparison_val.csv")
"""),
    md("""
## 6 · Open-set threshold (`unknown` class)
Tuned on validation only: energy of in-distribution images vs. images whose labels were
quarantined. Also reported as AUROC.
"""),
    code("""
# @title 6a · Energy threshold + AUROC
import numpy as np, torch, json
from src.models.classifier import tune_open_set_threshold
from src.eval.metrics import open_set_auroc
from src.train.classifier_train import predict_proba

best_model_name = max(val_metrics, key=lambda n: val_metrics[n]["f1_macro"])
model = build_classifier(cfg.raw["pretrained"]["classifiers"][best_model_name], heads, image_size=size)
model.load_state_dict(torch.load(Path(cfg.paths.checkpoints) / f"cls_{best_model_name}_best.pt",
                                 map_location=device)["model"])
model.to(device).eval()

known_logits, _ = predict_proba(model, loaders["val"], device)
known_energy = torch.logsumexp(torch.tensor(known_logits), dim=1).numpy()

qidx = pd.read_csv(cfg.paths.quarantine / "quarantine_index.csv")
ood_paths = [p for p in qidx.path.astype(str) if Path(p).exists()][:800]
if ood_paths:
    from src.data.dataset import ManifestDataset
    ood_df = pd.DataFrame({"path": ood_paths, "dataset_label": heads.species[0]})
    ood_loader = make_loader(ManifestDataset(ood_df, {heads.species[0]: 0},
                                             build_transforms(size, train=False)),
                             batch_size=cfg.raw["classifier"]["batch"], train=False, workers=2)
    ood_logits, _ = predict_proba(model, ood_loader, device)
    ood_energy = torch.logsumexp(torch.tensor(ood_logits), dim=1).numpy()
else:
    ood_energy = np.array([])

thr = tune_open_set_threshold(known_energy, ood_energy,
                              target_tpr=cfg.raw["classifier"]["open_set"]["target_tpr"])
thr["open_set_auroc"] = (open_set_auroc(known_energy, ood_energy) if len(ood_energy) else
                         "not_computable: no quarantined images available")
thr["model"] = best_model_name
(cfg.paths.models / "open_set_threshold.json").write_text(json.dumps(thr, indent=2))
print(json.dumps(thr, indent=2))
"""),
    md("""
## 7 · Ablation: with vs without the label audit
The same architecture and split, trained briefly (fewer epochs, config-driven) on the
**uncleaned** label set, to quantify what the audit bought us.
"""),
    code("""
# @title 7a · Ablation run (audit-cleaned vs raw labels)
from sklearn.metrics import f1_score

def ablation_eval(model, loader, device):
    logits, y = predict_proba(model, loader, device)
    return {"macro_f1": float(f1_score(y, logits.argmax(1), average="macro",
                                       zero_division=0))}

RUN_ABLATION = True
if not RUN_ABLATION:
    print("skipped (RUN_ABLATION=False)")
else:
    key = "classifier.ablation_no_audit"
    with stage(tr, key) as st:
        if not st.skip:
            raw = discover_classification(cfg)
            raw_split = build_splits(cfg, raw, cfg.paths.data_splits / "ablation_raw",
                                     embed_batch=64)
            raw_train = pd.read_csv(cfg.paths.data_splits / "ablation_raw" / "train.csv")
            raw_val = pd.read_csv(cfg.paths.data_splits / "ablation_raw" / "val.csv")
            raw_classes = sorted(raw.dataset_label.unique())
            raw_heads = TaxonomyHeads.from_table(raw_classes, raw_classes, raw_classes)
            rl = {}
            for split, d in (("train", raw_train), ("val", raw_val)):
                ds = ManifestDataset(d, {c: i for i, c in enumerate(raw_classes)},
                                     build_transforms(size, train=(split == "train")))
                rl[split] = make_loader(ds, cfg.raw["classifier"]["batch"],
                                        train=(split == "train"), workers=4, seed=cfg.seed)
            model = build_classifier(cfg.raw["pretrained"]["classifiers"][best_model_name],
                                     raw_heads, image_size=size)
            res = fit(cfg, model, rl, raw_heads, Path(cfg.paths.checkpoints),
                      run_name="cls_ablation_no_audit", tracker=tr, stage_key=key,
                      epochs_stage1=2, epochs_stage2=6,
                      eval_fn=ablation_eval)
            st.metric(**{k: v for k, v in res.items() if isinstance(v, (int, float))})
            print("ablation:", res["best_macro_f1"])
"""),
    md("""
## 8 · FINAL: the test split, once
This is the only place the test split is read. The split lock is verified first; if the
manifests changed after freezing, this cell refuses to run.
"""),
    code("""
# @title 8a · Test evaluation (single shot, guarded)
from src.data.split import verify_lock
verify_lock(splits_dir)
key = "classifier.final_test"
if tr.is_done(key):
    print("test was already evaluated in this run:", tr.get(key).metrics)
else:
    with stage(tr, key) as st:
        if not st.skip:
            model = build_classifier(cfg.raw["pretrained"]["classifiers"][best_model_name],
                                     heads, image_size=size)
            model.load_state_dict(torch.load(
                Path(cfg.paths.checkpoints) / f"cls_{best_model_name}_best.pt",
                map_location=device)["model"]); model.to(device)
            logits, y = predict_proba(model, loaders["test"], device)
            probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
            m = classification_metrics(y, probs, heads.species)
            m["balanced_accuracy"] = m["balanced_accuracy"]
            save_metrics(m, cfg.paths.metrics, "classifier_final_test",
                         meta={"split": "test", "n_images": int(len(test_df)),
                               "arch": best_model_name, "seed": cfg.seed,
                               "note": "evaluated once; see SPLIT_LOCK.json"})
            plot_confusion(np.array(m["confusion_matrix"]), heads.species,
                           cfg.paths.figures / "cm_final_test_norm.png", normalize=True,
                           title="final test (row-normalised)")
            st.metric(accuracy=m["accuracy"], f1_macro=m["f1_macro"],
                      f1_weighted=m["f1_weighted"], ece=m["ece"])
            st.out(cfg.paths.metrics / "classifier_final_test.json")
            print({k: m[k] for k in ("accuracy", "top5_accuracy", "f1_macro",
                                    "f1_weighted", "balanced_accuracy", "ece")})
"""),
    md("""
## 9 · Cross-domain drop (market images → pond crops)
Reported only when detector crops exist; otherwise the cell says `not_computable` rather
than inventing a number.
"""),
    code("""
# @title 9a · Cross-domain evaluation
crop_csv = cfg.paths.data_interim / "detector_crops" / "crops.csv"
if not crop_csv.exists():
    print("not_computable: no detector crops (run 03_detector.ipynb step 5)")
else:
    from src.data.dataset import CropDataset
    cdf = pd.read_csv(crop_csv)
    mask = cdf.dataset_label.isin(heads.species) & cdf.score.gt(0.3)
    cdf = cdf[mask]
    if len(cdf) < 20:
        print(f"not_computable: only {len(cdf)} usable crops")
    else:
        ds = CropDataset(cdf, class_to_idx, build_transforms(size, train=False))
        dl = make_loader(ds, cfg.raw["classifier"]["batch"], train=False, workers=2)
        model = build_classifier(cfg.raw["pretrained"]["classifiers"][best_model_name],
                                 heads, image_size=size)
        model.load_state_dict(torch.load(Path(cfg.paths.checkpoints) /
                                         f"cls_{best_model_name}_best.pt",
                                         map_location=device)["model"]); model.to(device)
        logits, y = predict_proba(model, dl, device)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        cm = classification_metrics(y, probs, heads.species)
        save_metrics(cm, cfg.paths.metrics, "cross_domain_market_to_pond",
                     meta={"n_images": int(len(cdf)), "source": "detector crops",
                           "in_domain": "market/classification images"})
        drop = val_metrics[best_model_name]["accuracy"] - cm["accuracy"]
        print(f"in-domain (val) acc={val_metrics[best_model_name]['accuracy']:.4f} | "
              f"cross-domain acc={cm['accuracy']:.4f} | drop={drop:+.4f}")
        json.dump({"cross_domain_drop": float(drop)},
                  open(cfg.paths.metrics / "cross_domain_drop.json", "w"), indent=2)
"""),
    md("### Next\n`05_tracking_counting.ipynb` — ByteTrack/BoT-SORT, species voting, counts."),
]


# =========================================================================== #
# 05 - tracking + counting
# =========================================================================== #
nb05 = setup_cells("05_tracking_counting", require_gpu=True) + [
    md("""
## 1 · Checkpoints
Tracking needs the detector (notebook 03) and the classifier (notebook 04).
"""),
    code("""
# @title 1a · Verify checkpoints exist (fail early, don't invent a number)
from pathlib import Path
import torch, pandas as pd
from src.models.classifier import heads_from_approved, build_classifier

det_ckpts = sorted(Path(cfg.paths.checkpoints).glob("det_*_best.pt"))
cls_ckpts = sorted(Path(cfg.paths.checkpoints).glob("cls_*_best.pt"))
if not det_ckpts:
    raise FileNotFoundError("no detector checkpoint: run 03_detector.ipynb")
if not cls_ckpts:
    raise FileNotFoundError("no classifier checkpoint: run 04_classifier.ipynb")
print("detectors:", [p.name for p in det_ckpts])
print("classifiers:", [p.name for p in cls_ckpts])

approved = pd.read_csv(cfg.paths.data_curated / "approved_labels.csv")
heads = heads_from_approved(approved)
best_cls = cls_ckpts[0]
cls_run_name = best_cls.stem[len("cls_"):-len("_best")]
cls_arch = cfg.raw["pretrained"]["classifiers"][cls_run_name]
print(len(heads.species), "species |", best_cls.name, "->", cls_arch)
"""),
    md("""
## 2 · Tracking
One track id = one fish. ByteTrack is primary; BoT-SORT is run on the same clips for
comparison. Tracks shorter than `min_track_frames` are dropped **before** counting.
"""),
    code("""
# @title 2a · Choose clips + run both trackers
from src.track.tracker import track_video
from src.utils.pretrained import ultralytics_model

videos_dir = cfg.dataset_path("videos")
clips = []
if videos_dir:
    clips = sorted([p for p in videos_dir.rglob("*")
                    if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}])
if not clips:
    print("No video dataset mounted. Attach clips as a Kaggle dataset named 'fish-videos' "
          "to run tracking+counting. Counts below would be fabricated, so we stop here.")
else:
    det_model = ultralytics_model(str(det_ckpts[0]))
    tracks_all = {}
    for clip in clips[:4]:
        for tracker in [cfg.raw["tracking"]["tracker"], *cfg.raw["tracking"]["comparisons"]]:
            key = f"track.{clip.stem}.{tracker}"
            with stage(tr, key) as st:
                if st.skip:
                    continue
                df_t = track_video(det_model, clip, cfg, tracker=tracker)
                out = cfg.paths.tables / f"tracks_{clip.stem}_{tracker.replace('.yaml','')}.csv"
                df_t.to_csv(out, index=False)
                tracks_all[(clip.stem, tracker)] = df_t
                st.out(out)
                st.metric(n_detections=int(len(df_t)),
                          n_tracks=int(df_t.track_id.nunique()))
                print(clip.name, tracker, len(df_t), "rows")
"""),
    md("## 3 · Species per track (confidence-weighted vote) + counts"),
    code("""
# @title 3a · Classify track crops and count
import numpy as np, torch, cv2
from pathlib import Path
from src.track.tracker import classify_track_crops, count_species
from src.models.classifier import build_classifier

if not clips:
    print("not_computable: no clips mounted")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clf = build_classifier(cls_arch, heads,
                           image_size=int(cfg.raw["classifier"]["input_size"]))
    clf.load_state_dict(torch.load(best_cls, map_location=device)["model"])
    clf.to(device).eval()

    captures = {}
    def frame_lookup(clip_name, idx):
        clip = next(c for c in clips if c.name == clip_name)
        cap = captures.get(clip_name)
        if cap is None:
            cap = cv2.VideoCapture(str(clip)); captures[clip_name] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read frame {idx} of {clip_name}")
        return frame

    counts_rows = []
    for (clip_name, tracker), df_t in tracks_all.items():
        species = classify_track_crops(df_t, frame_lookup, clf, cfg)
        res = count_species(df_t, species, heads.species,
                            min_track_frames=int(cfg.raw["tracking"]["min_track_frames"]))
        out = cfg.paths.tables / f"counts_{clip_name}_{tracker.replace('.yaml','')}.csv"
        res["counts"].to_csv(out, index=False)
        counts_rows.append({"clip": clip_name, "tracker": tracker, **res["summary"],
                            "total_fish": int(res["summary"]["n_tracks_surviving"])})
        print(clip_name, tracker, res["summary"]["by_species"])
    counts_df = pd.DataFrame(counts_rows)
    counts_df.to_csv(cfg.paths.metrics / "counting_summary.csv", index=False)
    display(counts_df)
"""),
    md("""
## 4 · Counting accuracy vs manual ground truth (MAE / RMSE / MAPE)
Provide `ground_truth_counts.csv` with columns `clip, species, count` in the mounted data;
without it the metrics are reported as `not_computable`.
"""),
    code("""
# @title 4a · Counting metrics + MOTA/HOTA when tracking GT exists
import json
from src.track.tracker import counting_metrics, mot_metrics

gt_path = None
if videos_dir:
    cands = list(videos_dir.rglob("ground_truth_counts.csv"))
    gt_path = cands[0] if cands else None
if not clips:
    print("not_computable: no clips")
elif gt_path is None:
    print("not_computable: no ground_truth_counts.csv in the mounted video dataset")
else:
    truth = pd.read_csv(gt_path)
    per_tracker = {}
    for tracker in counts_df.tracker.unique():
        sub = counts_df[counts_df.tracker == tracker]
        pred = pd.read_csv(cfg.paths.tables /
                           f"counts_{sub.clip.iloc[0]}_{tracker.replace('.yaml','')}.csv")
        per_tracker[tracker] = counting_metrics(pred, truth)
    (cfg.paths.metrics / "counting_metrics.json").write_text(json.dumps(per_tracker, indent=2))
    print(json.dumps(per_tracker, indent=2)[:1500])
    gt_tracks = list(videos_dir.rglob("ground_truth_tracks.csv"))
    if gt_tracks:
        gt = pd.read_csv(gt_tracks[0])
        (cfg.paths.metrics / "mot_metrics.json").write_text(
            json.dumps(mot_metrics(list(tracks_all.values())[0], gt), indent=2))
        print("MOT metrics written")
"""),
    md("### Next\n`06_density_map.ipynb` — density with uncertainty, then the maps."),
]


# =========================================================================== #
# 06 - density + mapping
# =========================================================================== #
nb06 = setup_cells("06_density_map", require_gpu=False) + [
    md("""
## 1 · Counts + field metadata
Density is **estimated**, not "detections ÷ area". `N_hat = unique_tracks / p_detect`, with
`p_detect` taken from a calibration file. Without calibration the notebook reports a
**relative density index** and says so.
"""),
    code("""
# @title 1a · Load counts and site metadata
import json, pandas as pd, numpy as np
from pathlib import Path

counts_path = cfg.paths.metrics / "counting_summary.csv"
counts = pd.read_csv(counts_path) if counts_path.exists() else pd.DataFrame()
field_dir = cfg.dataset_path("jh_fish_field")
site_csv = None
if field_dir:
    c = list(Path(field_dir).rglob("field_schema.csv"))
    site_csv = c[0] if c else None
if site_csv:
    field = pd.read_csv(site_csv)
    print("field rows:", len(field))
else:
    field = pd.DataFrame(columns=["id", "date", "time", "lat", "lon", "water_body",
                                  "depth_m", "camera_coverage_m2", "species", "count"])
    print("No jh_fish_field dataset mounted -> density needs either counting_summary.csv "
          "or field_schema.csv; both are absent in this run.")
display(counts.head() if len(counts) else "no counts")
"""),
    md("## 2 · IoT sensor join"),
    code("""
# @title 2a · Join sensor logs to detections
from src.iot.sensors import load_sensor_csv, load_mqtt_log, join_by_time_gps, presence_model

sensor_frames = []
if field_dir:
    for p in Path(field_dir).rglob("*"):
        if p.suffix.lower() == ".csv" and "sensor" in p.name.lower():
            sensor_frames.append(load_sensor_csv(p))
        elif p.suffix.lower() in {".jsonl", ".log", ".txt"} and "mqtt" in p.name.lower():
            sensor_frames.append(load_mqtt_log(p))
sensors = pd.concat(sensor_frames, ignore_index=True) if sensor_frames else pd.DataFrame()
print("sensor rows:", len(sensors))
joined = join_by_time_gps(field, sensors) if (len(field) and len(sensors)) else field
"""),
    md("## 3 · Density with uncertainty"),
    code("""
# @title 3a · Estimated densities (+ bootstrap 95% CIs)
from src.density.estimate import estimate, per_species_density, p_detect_from_calibration

cal = p_detect_from_calibration(cfg.raw["density"]["calibration_file"])
print("p_detect calibration:", cal if cal else "none -> relative index only")
results = []
if len(counts):
    area = float(cfg.raw["density"]["camera_coverage_m2"])
    depth = float(cfg.raw["density"]["mean_depth_m"])
    for tracker, g in counts.groupby("tracker"):
        tot = int(g.total_fish.sum())
        est = estimate(tot, area, depth_m=depth, p_detect=cal,
                       bootstrap_iters=int(cfg.raw["density"]["bootstrap_iters"]),
                       ci=float(cfg.raw["density"]["ci"]),
                       per_clip_tracks=g.total_fish.values)
        results.append({"tracker": tracker, **est})
elif len(field):
    area = float(field.camera_coverage_m2.dropna().median() or 25.0)
    est = estimate(int(field["count"].sum()), area, p_detect=cal)
    results.append({"tracker": "field", **est})
else:
    print("not_computable: no counts and no field data")
dens = pd.DataFrame(results)
if len(dens):
    dens.to_csv(cfg.paths.metrics / "density_estimates.csv", index=False)
    display(dens)
"""),
    md("## 4 · Maps (GeoJSON / PNG / interactive HTML)"),
    code("""
# @title 4a · Spatial outputs
from src.mapping.maps import export_all

if len(field) and {"lat", "lon"} <= set(field.columns):
    maps = export_all(field, cfg.paths.figures / "maps", cfg)
    print(json.dumps({k: str(v) for k, v in maps.items()}, indent=2)[:1200])
    (cfg.paths.metrics / "map_outputs.json").write_text(json.dumps(maps, indent=2, default=str))
else:
    print("not_computable: no georeferenced detections (needs lat/lon per clip or image)")
"""),
    md("## 5 · Species presence vs water parameters"),
    code("""
# @title 5a · Gradient boosting baseline (+ optional DNN)
if len(joined) >= 30 and "species" in joined.columns:
    res = presence_model(joined, target="species", dnn=True)
    (cfg.paths.metrics / "presence_model.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2)[:1200])
else:
    print(f"not_computable: {len(joined)} joined rows (need >= 30 with species labels)")
"""),
    md("""
## 6 · Deliverables checklist
"""),
    code("""
# @title 6a · What exists in /kaggle/working for this run
from src.utils.state import validate_state
import subprocess
print(tr.summary())
print()
print("state health:", validate_state(cfg.paths.logs))
print(subprocess.run(["du", "-sh", str(cfg.paths.outputs)], capture_output=True,
                     text=True).stdout)
for folder in ("metrics", "figures", "tables", "models", "review", "quarantine"):
    p = cfg.paths.outputs / folder
    files = sorted(x.name for x in p.glob("*")) if p.exists() else []
    print(f"{folder:11s} {len(files):3d} files: {files[:8]}")
"""),
    code("""
# @title 6b · Manifest of produced artefacts (for the write-up)
import json, datetime, hashlib
from pathlib import Path
artefacts = []
for f in sorted(cfg.paths.outputs.rglob("*")):
    if f.is_file() and f.suffix in {".json", ".csv", ".png", ".md", ".html", ".geojson", ".pt"}:
        artefacts.append({"path": str(f.relative_to(cfg.paths.working_root)),
                          "bytes": f.stat().st_size,
                          "sha256": hashlib.sha256(f.read_bytes()).hexdigest()[:16]})
manifest = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "git_commit": sha, "config": str(cfg.path),
            "n_artefacts": len(artefacts), "artefacts": artefacts,
            "caveats": [
                "pretrained weights required; from-scratch training is disabled by policy",
                "detector numbering depends on which detection dataset was mounted",
                "density is relative unless a p_detect calibration file was provided",
            ]}
out = cfg.paths.outputs / "RUN_MANIFEST.json"
out.write_text(json.dumps(manifest, indent=2))
print("wrote", out, "with", len(artefacts), "artefacts")
"""),
]


# =========================================================================== #
def build(nb: dict, path: Path) -> None:
    nb = dict(nb)
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5
    nb.setdefault("metadata", {})
    nb["metadata"].update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
        "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True,
                   "language": "python", "sourceType": "notebook"},
    })
    for i, cell in enumerate(nb["cells"]):
        cell.setdefault("id", f"c{i:03d}")
        cell.setdefault("metadata", {})
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"{path.name:32s} {len(nb['cells']):3d} cells ({n_code} code)")


def main() -> None:
    for name, cells in [
        ("01_data_download.ipynb", nb01),
        ("02_taxonomy_audit.ipynb", nb02),
        ("03_detector.ipynb", nb03),
        ("04_classifier.ipynb", nb04),
        ("05_tracking_counting.ipynb", nb05),
        ("06_density_map.ipynb", nb06),
    ]:
        build({"cells": cells}, OUT / name)


if __name__ == "__main__":
    main()
