"""Dataset registry: what the spec asks for, where it officially lives, what we can
actually reach from this sandbox, and what each item is allowed to be used for.

Ground rules encoded here:
  * `official` URLs are always attempted first and their result recorded.
  * `mirrors` are only ever *proxies*. When a mirror is not the same dataset as the
    official one it is flagged `substitute_for` + `identical_to_official=False`,
    and it can never be reported as the official dataset.
  * No sizes or class counts are typed by hand into this file; the inventory step
    (scripts/02_inventory_raw.py) measures them from the files on disk.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# Official sources (spec section 3)
# --------------------------------------------------------------------------- #
OFFICIAL: list[dict[str, Any]] = [
    {
        "key": "indian_seafood_fish",
        "name": "Indian Seafood Fish Dataset (Mendeley mxf2c45yb5, v1)",
        "spec_use": "Classification, 8 Indian market species",
        "citation": "Paygude et al., Data in Brief 58 (2025) 111209; DOI 10.17632/mxf2c45yb5.1",
        "official_urls": [
            "https://data.mendeley.com/public-api/datasets/mxf2c45yb5/files?folder_id=root&version=1",
            "https://data.mendeley.com/datasets/mxf2c45yb5/1",
        ],
        "access": "open (Mendeley Data API / web form)",
        "licence": "to be verified from the Mendeley landing page (not reachable here)",
        "known_issues": "Documented label/name mismatches - spec section 4.2.",
        "critical": True,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
    {
        "key": "as_fishfresh",
        "name": "AS-FishFresh (Mendeley sdhp9bmrz5, v2)",
        "spec_use": "Classification, 48 Assam freshwater species",
        "citation": "DOI 10.17632/sdhp9bmrz5.2",
        "official_urls": [
            "https://data.mendeley.com/public-api/datasets/sdhp9bmrz5/files?folder_id=root&version=2",
            "https://data.mendeley.com/datasets/sdhp9bmrz5/2",
        ],
        "access": "open (Mendeley Data API / web form)",
        "licence": "to be verified from the Mendeley landing page (not reachable here)",
        "known_issues": "Augmented copies must be train-only; originals only for val/test.",
        "critical": True,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
    {
        "key": "depondfi",
        "name": "DePondFi (pond fish detection, South India)",
        "spec_use": "Primary detector training data (single class `fish`)",
        "citation": "DePondFi'23/'24 challenge, NCVPRIPG; https://github.com/DePondFi23/DePondFi",
        "official_urls": [
            "https://github.com/DePondFi23/DePondFi",
            "https://api.github.com/repos/DePondFi23/DePondFi/git/trees/main?recursive=1",
        ],
        "access": "GitHub repo (code/docs only); the annotated image archive itself is hosted "
                  "outside GitHub and its host is not reachable from this sandbox",
        "licence": "no LICENSE file in the repo (checked via GitHub API); images are released "
                   "for the challenge - licence of the image archive cannot be verified here",
        "known_issues": "Repo main branch contains only PDF/PPTX/DOCX documentation "
                        "(DMACS_SAI_2.pdf, POND_VISION_3.pdf, DePondFi2023-main.zip, SAHAJEEVIS_3.docx).",
        "critical": True,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
    {
        "key": "qut_fish_468",
        "name": "Kaggle 'Fish Species Image Data' (QUT FISH, 468 species)",
        "spec_use": "Transfer learning / long-tail classification",
        "citation": "Kaggle sripaadsrinivasan/fish-species-image-data; Anantharajah et al., WACV 2014",
        "official_urls": [
            "https://www.kaggle.com/datasets/sripaadsrinivasan/fish-species-image-data",
            "https://www.kaggle.com/api/v1/datasets/download/sripaadsrinivasan/fish-species-image-data",
        ],
        "access": "Kaggle API (requires KAGGLE_USERNAME/KAGGLE_KEY) + host reachability",
        "licence": "stated inside the dataset (LicenceAndAuthors.pdf); not reachable here",
        "known_issues": "Not India-specific (QUT, Australia).",
        "critical": False,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
    {
        "key": "kaggle_large_scale_fish",
        "name": "Kaggle 'A Large Scale Fish Dataset' (9 seafood classes)",
        "spec_use": "Extra classification",
        "citation": "Kaggle crowww/a-large-scale-fish-dataset; Ulucan, Karakurt & Ulucan (2020)",
        "official_urls": [
            "https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset",
            "https://www.kaggle.com/api/v1/datasets/download/crowww/a-large-scale-fish-dataset",
        ],
        "access": "Kaggle API (requires credentials) + host reachability",
        "licence": "CC0-1.0 as stated on the Kaggle dataset page (to be re-verified)",
        "known_issues": "Check for heavy augmentation before use.",
        "critical": False,
        "content_available": True,  # content (not just the host) obtainable in this environment
    },
    {
        "key": "inat_gbif_india",
        "name": "iNaturalist + GBIF, India-filtered, research-grade, CC-licensed",
        "spec_use": "Long-tail Indian species",
        "citation": "api.inaturalist.org/v1, api.gbif.org/v1",
        "official_urls": [
            "https://api.inaturalist.org/v1/observations",
            "https://api.gbif.org/v1/occurrence/search",
        ],
        "access": "open APIs (rate-limited)",
        "licence": "per-record CC licences (kept as metadata)",
        "known_issues": "Must keep licence + observer metadata; no NC/ND images.",
        "critical": True,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
    {
        "key": "dried_fish",
        "name": "Dried fish dataset (Paygude et al., Data in Brief 55 (2024) 110563)",
        "spec_use": "Optional, separate domain",
        "citation": "DOI 10.1016/j.dib.2024.110563",
        "official_urls": [],  # Mendeley dataset id not verifiable from here - see DECISIONS.md
        "access": "Mendeley Data (identifier not resolvable offline; article page unreachable)",
        "licence": "unknown (article unreachable)",
        "known_issues": "Optional item; skipped when the identifier cannot be verified. "
                        "We do not guess a dataset id.",
        "critical": False,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
    {
        "key": "jh_fish",
        "name": "Own field data (JH-Fish: Ranchi ponds, Kanke Dam, Getalsud, Subarnarekha, Damodar)",
        "spec_use": "Final real-world test",
        "citation": "collected by the project owner",
        "official_urls": [],
        "access": "not collected yet - no files exist in this repo",
        "licence": "n/a",
        "known_issues": "Schema defined in spec section 9 (src/iot/). All field-test "
                        "numbers are reported as 'not available' until data exists.",
        "critical": False,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
    # ---- taxonomic authorities (spec 4.1) ----
    {
        "key": "fishbase_india",
        "name": "FishBase country checklist (India) + WoRMS / Eschmeyer validation",
        "spec_use": "Taxonomy master (section 4.1)",
        "citation": "fishbase.se, marinespecies.org, researcharchive.calacademy.org",
        "official_urls": [
            "https://www.fishbase.se/",
            "https://www.marinespecies.org/rest/",
            "https://researcharchive.calacademy.org/research/ichthyology/catalog/fishcatmain.asp",
        ],
        "access": "web/API",
        "licence": "FishBase: CC BY-NC 4.0 (non-commercial); WoRMS: CC BY 4.0",
        "known_issues": "All three hosts unreachable from this sandbox.",
        "critical": True,
        "content_available": False,  # content (not just the host) obtainable in this environment
    },
]

# --------------------------------------------------------------------------- #
# Reachable proxies / mirrors (verified reachable through codeload.github.com)
# --------------------------------------------------------------------------- #
MIRRORS: list[dict[str, Any]] = [
    {
        "key": "kaggle_large_scale_fish",
        "repo": "tar-ang-2004/Fish_Classification",
        "identical_to_official": True,
        "substitute_for": None,
        "provenance": "Third-party GitHub copy of Kaggle 'A Large Scale Fish Dataset' "
                      "(crowww/a-large-scale-fish-dataset, Ulucan et al. 2020). Folder names "
                      "match the Kaggle class list (Black Sea Sprat ... Trout).",
        "licence": "repo declares MIT (LICENSE); upstream Kaggle page states CC0-1.0 "
                   "- treat as CC0-1.0 with MIT repo packaging",
        "notes": "Ships the uploader's own random train/val/test split. That split is NOT used: "
                 "per spec section 5 a leakage-safe split is rebuilt from scratch.",
        "expected_layout": "Dataset/{train,val,test}/<class folder>/<image>.jpg",
        "usable_for": ["classification"],
        # only this subtree is the dataset; the repo also carries an app + charts screenshots
        "image_root": "Dataset",
        "label_root": None,
    },
    {
        "key": "fish_detection_proxy",
        "repo": "1wenjinjie/yolov10-fish-detection",
        "identical_to_official": False,
        "substitute_for": "depondfi",
        "provenance": "YOLO-format fish detection dataset bundled with a YOLOv10 fish-detection "
                      "project. NOT DePondFi: different source imagery, different annotation set.",
        "licence": "AGPL-3.0 (repo LICENSE) - derivative works must respect AGPL",
        "notes": "Only stand-in available for detector training; every detector number must be "
                 "reported as 'trained on <fish_detection_proxy>, not DePondFi'.",
        "expected_layout": "data/images/*.jpg + data/labels/*.txt (+ data/Annotations/*)",
        "usable_for": ["detection"],
        "image_root": "data/images",
        "label_root": "data/labels",
    },
    {
        "key": "bd_indigenous_fish",
        "repo": "knowaminul/BDIndigenousFish2019",
        "identical_to_official": False,
        "substitute_for": "as_fishfresh",
        "provenance": "BDIndigenousFish2019: 8 indigenous South-Asian freshwater species "
                      "(Bengali common-name folders, one zip per class). Proxy for the "
                      "48-species AS-FishFresh set, which is unreachable.",
        "licence": "no LICENSE file in repo (checked); upstream dataset published for research use "
                   "- redistribution terms unverified, treat as research-only",
        "notes": "Folder labels are Bengali common names, not scientific names: every label goes "
                 "through label_map.csv with an explicit confidence, and low-confidence mappings "
                 "go to quarantine + review pack.",
        "expected_layout": "<Class>.zip at repo root (8 zips)",
        "usable_for": ["classification"],
        "image_root": ".",
        "label_root": None,
    },
    {
        "key": "taxonomy_fishdex",
        "repo": "ScottMorris123/FishDex",
        "identical_to_official": False,
        "substitute_for": "fishbase_india",
        "provenance": "Sharded dump of FishBase taxa: one CSV per genus with "
                      "ScientificName, SpecCode (FishBase), CommonName, Environments, Subspecies, "
                      "plus per-family index.json carrying Family + Order.",
        "licence": "no LICENSE file in repo; underlying FishBase data is CC BY-NC 4.0 "
                   "(non-commercial, attribution to FishBase)",
        "notes": "Gives accepted names + family/genus/environment. Does NOT carry country "
                 "distribution, IUCN status, or synonyms, so the India checklist itself "
                 "cannot be reproduced offline. See DECISIONS.md D-04.",
        "expected_layout": "fishdex_sharded/families/*/index.json, fishdex_sharded/genera/*/*.csv",
        "usable_for": ["taxonomy"],
        "image_root": None,
        "label_root": None,
    },
]

# Datasets the spec wants that have no reachable copy at all (recorded, not invented)
UNAVAILABLE: list[dict[str, str]] = [
    {"key": "indian_seafood_fish", "reason": "Mendeley Data host unreachable (TLS closed); no "
                                             "GitHub copy found via repo+code search."},
    {"key": "as_fishfresh", "reason": "Mendeley Data host unreachable; no GitHub copy found."},
    {"key": "depondfi", "reason": "Image archive not in the GitHub repo and its host is "
                                  "unreachable; repo ships documentation only."},
    {"key": "qut_fish_468", "reason": "Kaggle host unreachable and no credentials; no GitHub "
                                      "mirror found."},
    {"key": "kaggle_large_scale_fish", "reason": "Kaggle host unreachable; used the verified "
                                                 "GitHub copy instead (see MIRRORS)."},
    {"key": "inat_gbif_india", "reason": "api.inaturalist.org and api.gbif.org unreachable."},
    {"key": "dried_fish", "reason": "Mendeley identifier not verifiable offline; not guessed."},
    {"key": "jh_fish", "reason": "Field data has not been collected."},
    {"key": "fishbase_india", "reason": "FishBase / WoRMS / Eschmeyer hosts unreachable; "
                                        "FishDex (FishBase-derived) used for names+family, "
                                        "no India checklist available."},
]

# Official hosts probed for the network report
PROBE_HOSTS: list[str] = [
    "https://data.mendeley.com/",
    "https://www.kaggle.com/",
    "https://api.inaturalist.org/v1/observations?per_page=1",
    "https://api.gbif.org/v1/occurrence/search?limit=1",
    "https://www.fishbase.se/",
    "https://www.marinespecies.org/rest/",
    "https://researcharchive.calacademy.org/",
    "https://github.com/",
    "https://api.github.com/",
    "https://codeload.github.com/",
    "https://pypi.org/",
    "https://huggingface.co/",
]
