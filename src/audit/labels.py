"""Label normalisation + matching against the reviewed label map.

Public dataset copies name their class folders inconsistently:
`fish sea_food black_sea_sprat`, `Black Sea Sprat`, `black-sea-sprat`. All three are
the same class, so the audit normalises before matching and *records how* it matched.

Important (DECISIONS D-17): resolving a folder to a scientific name is metadata only -
it never rewrites the training label. So a wrong match here degrades reporting, not
the classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# prefixes that dataset mirrors bolt onto folder names
_STRIP_PREFIXES = ("fish sea_food ", "fish sea food ", "fish seafood ", "sea_food ",
                   "seafood ", "fish ", "animal ")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_label(raw: str, strip_prefixes: bool = True) -> str:
    """`"Fish Sea_Food Black Sea Sprat"` -> `"black_sea_sprat"`.

    `strip_prefixes=False` keeps mirror prefixes, so `"animal fish"` stays
    `"animal_fish"` and can still be matched exactly instead of collapsing to `"fish"`.
    """
    s = str(raw).strip().lower().replace("-", " ")
    if strip_prefixes:
        changed = True
        while changed:                   # prefixes can stack ("animal fish bass")
            changed = False
            for prefix in _STRIP_PREFIXES:
                rest = s[len(prefix):]
                if s.startswith(prefix) and rest.strip():
                    s = rest
                    changed = True
    return _NON_ALNUM.sub("_", s).strip("_")


@dataclass
class LabelMatch:
    key: str | None            # matched manual-map key (normalised)
    method: str                # exact | substring | ambiguous | unmatched
    score: float               # 1.0 exact, 0.8 unique substring, 0.0 none

    @property
    def ok(self) -> bool:
        return self.key is not None


def match_label(raw: str, manual_keys: list[str]) -> LabelMatch:
    """Match a dataset folder name to a reviewed manual-map key.

    Order: exact (prefix-stripped) -> exact (untrimmed) -> unique substring ->
    closest substring -> ambiguous. Exact always beats fuzzy, so `"animal fish"`
    resolves to `animal_fish`, not to `animal_fish_bass`.
    """
    keys = [k for k in manual_keys if k]
    for n in (normalize_label(raw), normalize_label(raw, strip_prefixes=False)):
        if not n:
            continue
        if n in keys:
            return LabelMatch(n, "exact", 1.0)
    n = normalize_label(raw)
    if len(n) < 3:
        return LabelMatch(None, "unmatched", 0.0)
    subs = [k for k in keys if k in n or n in k]
    if not subs:
        return LabelMatch(None, "unmatched", 0.0)
    # closest length wins; equidistant candidates mean the name is genuinely ambiguous
    subs.sort(key=lambda k: abs(len(k) - len(n)))
    if len(subs) > 1 and abs(len(subs[0]) - len(n)) == abs(len(subs[1]) - len(n)):
        return LabelMatch(subs[0], "ambiguous", 0.4)
    return LabelMatch(subs[0], "substring", 0.8)


def resolve_labels(folders: list[str], manual_keys: list[str]) -> dict[str, LabelMatch]:
    return {f: match_label(f, manual_keys) for f in folders}
