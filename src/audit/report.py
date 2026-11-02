"""Writes outputs/AUDIT_REPORT.md (spec 4.2 + 4.3 + deliverable 1).

Every number comes from a computed artefact passed in; the report refuses to
print a value it was not given, and prints "not computed" instead.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("geofish")

# spec 4.2 - decision table. Applied only where `applied` says so.
SEAFOOD_AUDIT = [
    dict(folder="Pomfret", printed="Pampus argeneus", action="rename",
         target="Pampus argenteus", rule="nomenclatural fix",
         confidence="n/a", applied="n/a (dataset not mounted)"),
    dict(folder="Mackerel", printed="Scomber scombrus", action="verify+relabel",
         target="Rastrelliger kanagurta (if Indian mackerel)",
         rule="vertical bars + spotted belly vs plain metallic flanks",
         confidence="high required", applied="n/a (dataset not mounted)"),
    dict(folder="Black Snapper", printed="Apsilus dentatus", action="verify+relabel",
         target="Oreochromis niloticus / O. mossambicus",
         rule="cichlid shape, vertical bars, dorsal spination",
         confidence="high required", applied="n/a (dataset not mounted)"),
    dict(folder="Indian Carp", printed="Cyprinus carpio", action="verify+relabel",
         target="Labeo rohita (if rohu)", rule="barbels, dorsal-fin base, colour",
         confidence="high required", applied="n/a (dataset not mounted)"),
    dict(folder="Prawn", printed="Dendrobranchiata", action="rank fix",
         target="Penaeoidea_prawn (species unresolved)",
         rule="suborder -> unresolved node; resolve per image only if carapace visible",
         confidence="high required", applied="n/a (dataset not mounted)"),
    dict(folder="Pink Perch", printed="Zalembius rosaceus", action="verify+relabel",
         target="Nemipterus japonicus", rule="filamentous upper caudal lobe, pink body",
         confidence="high required", applied="n/a (dataset not mounted)"),
    dict(folder="Indian flathead", printed="Lates calcarifer", action="verify",
         target="keep Lates calcarifer only if images show barramundi",
         rule="silver body, large mouth, lateral line",
         confidence="high required", applied="n/a (dataset not mounted)"),
    dict(folder="Black Pomfret", printed="Parastromateus niger", action="audit look-alikes",
         target="Parastromateus niger (split page-level flags separately)",
         rule="red fins suggest freshwater pacu look-alikes",
         confidence="high required", applied="n/a (dataset not mounted)"),
]


def write_audit_report(out_path: Path, *, config_snapshot: dict[str, Any],
                       before_counts: dict[str, int], after_counts: dict[str, int],
                       duplicates: pd.DataFrame | None,
                       suspicious: pd.DataFrame | None,
                       decisions: pd.DataFrame | None,
                       quarantine: dict[str, Any],
                       quality: pd.DataFrame | None,
                       label_map: pd.DataFrame | None,
                       bioclip_status: dict[str, Any],
                       seafood_mounted: bool,
                       extra: dict[str, Any] | None = None) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    A = L.append
    A("# AUDIT_REPORT.md — label audit results\n")
    A(f"Config: `{config_snapshot.get('config_path')}` · seed **{config_snapshot.get('seed')}** "
      f"· generated {config_snapshot.get('generated_at')}\n")

    # ---------------- 1. before / after ---------------- #
    A("\n## 1. Class counts: before vs after the audit\n")
    A("| dataset_label | before | after | delta |\n|---|---|---|---|")
    all_keys = sorted(set(before_counts) | set(after_counts))
    for k in all_keys:
        b, a = before_counts.get(k, 0), after_counts.get(k, 0)
        A(f"| `{k}` | {b} | {a} | {a - b:+d} |")
    A(f"\n**totals:** before {sum(before_counts.values())}, after {sum(after_counts.values())}\n")

    # ---------------- 2. duplicates ---------------- #
    A("\n## 2. Duplicates (spec 4.3.1)\n")
    if duplicates is None or len(duplicates) == 0:
        A("No duplicate clusters found (or the stage did not run).")
    else:
        n_groups = duplicates["group"].nunique() if "group" in duplicates else "n/a"
        extra_imgs = len(duplicates) - (n_groups if isinstance(n_groups, int) else 0)
        A(f"- duplicate clusters: **{n_groups}**")
        A(f"- images involved: **{len(duplicates)}** "
          f"(≈{extra_imgs} removable copies)")
        if {"method"} <= set(duplicates.columns):
            A(f"- detection methods: {dict(Counter(duplicates['method']))}")
        A("\nMethod: sha256 exact match + pHash Hamming ≤ 6 confirmed by "
          "HSV-histogram ≥ 0.97 + frozen DINOv2 embedding cosine ≥ threshold "
          "(see configs/kaggle.yaml). Duplicates are pooled across the source "
          "train/val/test folders and never split across sides.")

    # ---------------- 3. label noise ---------------- #
    A("\n## 3. Label-noise suspects (spec 4.3.2 / 4.3.3)\n")
    if suspicious is None or len(suspicious) == 0:
        A("cleanlab flagged no label issues.")
    else:
        A(f"- cleanlab flagged **{len(suspicious)}** images "
          f"({len(suspicious) / max(sum(before_counts.values()), 1) * 100:.2f}% of the pool).")
        A(f"- top predicted confusions: "
          f"{Counter(zip(suspicious['given_label'], suspicious['predicted_label'])).most_common(5)}")
    A("\nZero-shot cross-check (BioCLIP):")
    if bioclip_status.get("available"):
        A(f"- ran on {bioclip_status.get('n_images')} images, "
          f"{bioclip_status.get('n_disagreements')} disagreements with the folder label.")
    else:
        A(f"- **not run** — {bioclip_status.get('reason', 'model unavailable')}.")

    # ---------------- 4. applied vs quarantined ---------------- #
    A("\n## 4. Relabels applied vs quarantined (spec 4.3.5)\n")
    if decisions is None or len(decisions) == 0:
        A("No relabel candidates.")
    else:
        n_rel = int((decisions.decision == "relabel").sum())
        n_q = int((decisions.decision == "quarantine").sum())
        A(f"- applied automatically: **{n_rel}** (only where cleanlab + every available "
          f"substitute opinion agreed and confidence ≥ threshold)")
        A(f"- quarantined for human review: **{n_q}** → `outputs/quarantine/` + "
          f"`outputs/review/`")
        if n_rel:
            A("\n| path | given | new | evidence |")
            A("|---|---|---|---|")
            for _, r in decisions[decisions.decision == "relabel"].head(50).iterrows():
                A(f"| `{Path(r['path']).name}` | {r['given_label']} | {r['new_label']} | "
                  f"cleanlab+knn, self_conf={r['self_confidence']:.3f} |")
    A(f"\nQuarantine index: {quarantine.get('n', 'n/a')} rows "
      f"({quarantine.get('by_reason', {})}) → `{quarantine.get('path', 'n/a')}`")

    # ---------------- 5. quality ---------------- #
    A("\n## 5. Image quality filter (spec 4.3.4)\n")
    if quality is None or len(quality) == 0:
        A("Quality stage did not run.")
    else:
        A(f"- unreadable: **{int((~quality.ok).sum())}**")
        A(f"- blurry (Laplacian variance < configured threshold): **{int(quality.blurry.sum())}**")
        A(f"- too small (short side < configured minimum): **{int(quality.too_small.sum())}**")
        A(f"- median size: {int(quality.width.median())}×{int(quality.height.median())} px")
        A("- nothing is deleted: flagged images are excluded and listed in the "
          "quarantine index.")

    # ---------------- 6. label map ---------------- #
    A("\n## 6. label_map.csv summary\n")
    if label_map is None or len(label_map) == 0:
        A("label_map.csv was not produced.")
    else:
        A(f"- rows: **{len(label_map)}**")
        if "confidence" in label_map:
            A(f"- by confidence: {dict(Counter(label_map['confidence']))}")
        if "applied" in label_map:
            A(f"- applied: {dict(Counter(label_map['applied'].astype(str)))}")
        A("\n| dataset | original_label | accepted_name | confidence | applied | method |")
        A("|---|---|---|---|---|---|")
        for _, r in label_map.head(60).iterrows():
            A(f"| `{r.get('dataset')}` | {r.get('original_label')} | "
              f"`{r.get('accepted_name')}` | {r.get('confidence')} | "
              f"{r.get('applied')} | {r.get('method')} |")

    # ---------------- 7. section 4.2 ---------------- #
    A("\n## 7. Spec §4.2 — Indian Seafood 8-folder audit\n")
    if not seafood_mounted:
        A("**The Indian Seafood dataset was not mounted in this run.** No folder-level "
          "verdict below is an image-derived claim; this is the decision table that will "
          "execute the moment the data is attached.\n")
    A("| folder | printed label | action | target | rule | confidence | applied |")
    A("|---|---|---|---|---|---|---|")
    for row in SEAFOOD_AUDIT:
        A(f"| {row['folder']} | *{row['printed']}* | {row['action']} | {row['target']} | "
          f"{row['rule']} | {row['confidence']} | {row['applied']} |")
    A("\n**Fig. 1 vs text discrepancy** (1,579+1,248+1,328+752+1,136+519+974+1,471 = 9,007 "
      "vs stated 8,488): **unverified** — the images were not available, so actual "
      "per-folder counts could not be measured. Both numbers are quoted as statements "
      "from the paper, not as measurements.")

    # ---------------- 8. not computed ---------------- #
    A("\n## 8. Explicitly NOT computed\n")
    for item in (extra or {}).get("not_computed", []):
        A(f"- {item}")

    A("\n## 9. Downstream consequence\n")
    A(f"- training manifest: `{config_snapshot.get('approved_manifest', 'n/a')}`")
    A(f"- rows entering training: **{(extra or {}).get('n_train_rows', 'n/a')}**")
    A("- the classifier trains only on labels that survived this audit; quarantined "
      "images keep their original label but are excluded and listed for review.")

    out_path.write_text("\n".join(L) + "\n")
    log.info("wrote %s", out_path)
    return out_path
