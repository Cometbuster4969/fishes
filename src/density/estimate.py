"""Density estimation with uncertainty (spec section 8).

We never report "detections / area" as a density. The estimator is

    N_hat  = (unique tracks) / p_detect
    D_A    = N_hat / A                    [fish m^-2]
    D_V    = N_hat / (A * H)              [fish m^-3]
    D_i    = N_i_hat / A                  per species

`p_detect` must come from calibration data (repeat passes / mark-resight /
field-of-view coverage). When it does not exist, the module refuses to invent one
and returns a relative density index with an explicit flag.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("geofish")


class NoCalibrationError(RuntimeError):
    """Raised when absolute density is requested without a p_detect estimate."""


def bootstrap_ci(values: np.ndarray, iters: int = 2000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return (float("nan"), float("nan"))
    stats = [rng.choice(values, size=values.size, replace=True).mean()
             for _ in range(iters)]
    lo, hi = np.quantile(stats, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def p_detect_from_calibration(path: str | Path | None) -> dict[str, Any] | None:
    """Read a calibration table: clip, known_fish, detected_fish (or a single p)."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("calibration file %s does not exist", p)
        return None
    import pandas as pd
    df = pd.read_csv(p)
    if "p_detect" in df.columns:
        est = float(df["p_detect"].mean())
        vals = df["p_detect"].astype(float).values
    elif {"known_fish", "detected_fish"} <= set(df.columns):
        vals = (df["detected_fish"] / df["known_fish"]).astype(float).values
        est = float(np.nanmean(vals))
    else:
        raise NoCalibrationError(
            f"{p} needs either a 'p_detect' column or 'known_fish'+'detected_fish'")
    lo, hi = bootstrap_ci(vals) if len(vals) > 1 else (np.nan, np.nan)
    return {"p_detect": est, "ci_low": lo, "ci_high": hi, "n_sites": int(len(vals)),
            "source": str(p), "method": "field calibration"}


def estimate(n_tracks: int, area_m2: float, depth_m: float | None = None,
             p_detect: dict[str, Any] | None = None, bootstrap_iters: int = 2000,
             ci: float = 0.95, seed: int = 42,
             per_clip_tracks: np.ndarray | None = None) -> dict[str, Any]:
    """Absolute density when p_detect exists, otherwise a relative index."""
    if area_m2 <= 0:
        raise ValueError("area_m2 must be > 0")

    if p_detect is None:
        rel = n_tracks / area_m2
        ci_lo, ci_hi = (bootstrap_ci(per_clip_tracks / area_m2, bootstrap_iters, ci, seed)
                        if per_clip_tracks is not None and len(per_clip_tracks) > 1
                        else (float("nan"), float("nan")))
        return {
            "type": "relative_density_index",
            "warning": ("No p_detect calibration available: this is a RELATIVE index "
                        "(counts per m^2 of camera footprint), comparable between clips "
                        "of the same setup only. It is NOT an absolute density."),
            "relative_index_per_m2": float(rel),
            "ci_low": ci_lo, "ci_high": ci_hi,
            "n_tracks": int(n_tracks), "area_m2": float(area_m2),
        }

    p = float(p_detect["p_detect"])
    if not (0 < p <= 1):
        raise ValueError(f"p_detect must be in (0,1], got {p}")
    n_hat = n_tracks / p
    d_a = n_hat / area_m2
    out: dict[str, Any] = {
        "type": "absolute_density",
        "n_tracks": int(n_tracks), "p_detect": p,
        "p_detect_ci": [p_detect.get("ci_low"), p_detect.get("ci_high")],
        "p_detect_source": p_detect.get("source"),
        "p_detect_method": p_detect.get("method"),
        "n_hat": float(n_hat),
        "D_A_per_m2": float(d_a),
        "D_A_per_100m2": float(d_a * 100),
    }
    if depth_m:
        out["D_V_per_m3"] = float(n_hat / (area_m2 * depth_m))
        out["depth_m"] = float(depth_m)

    # uncertainty: bootstrap over clips AND over the p_detect estimate
    if per_clip_tracks is not None and len(per_clip_tracks) > 1:
        rng = np.random.default_rng(seed)
        p_lo = p_detect.get("ci_low") or p
        p_hi = p_detect.get("ci_high") or p
        p_lo = p_lo if np.isfinite(p_lo) else p
        p_hi = p_hi if np.isfinite(p_hi) else p
        samples = []
        for _ in range(bootstrap_iters):
            n_b = rng.choice(per_clip_tracks, size=len(per_clip_tracks), replace=True).sum()
            p_b = rng.uniform(min(p_lo, p), max(p_hi, p)) if p_hi > p_lo else p
            samples.append((n_b / p_b) / area_m2)
        lo, hi = np.quantile(samples, [(1 - ci) / 2, 1 - (1 - ci) / 2])
        out["ci_low"] = float(lo)
        out["ci_high"] = float(hi)
        out["bootstrap_iters"] = int(bootstrap_iters)
        out["ci"] = ci
    else:
        out["ci_low"] = out["ci_high"] = "not_computable: fewer than 2 independent clips"
    return out


def per_species_density(counts_by_species: dict[str, int], area_m2: float,
                        p_detect: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = []
    for sp, n in counts_by_species.items():
        est = estimate(n, area_m2, p_detect=p_detect)
        key = "D_A_per_m2" if est["type"] == "absolute_density" else "relative_index_per_m2"
        rows.append({"species": sp, "count": n, key: est[key],
                     "ci_low": est.get("ci_low"), "ci_high": est.get("ci_high"),
                     "type": est["type"]})
    return rows


def compare_against_truth(estimate_row: dict[str, Any], true_density_per_m2: float
                          ) -> dict[str, Any]:
    d = estimate_row.get("D_A_per_m2")
    if d is None:
        return {"error": "not_computable: relative index only"}
    err = float(d) - true_density_per_m2
    return {
        "estimated": float(d), "true": float(true_density_per_m2),
        "abs_error": abs(err), "rel_error_pct": (abs(err) / true_density_per_m2 * 100
                                                 if true_density_per_m2 else float("nan")),
        "covered_by_ci": bool(estimate_row["ci_low"] <= true_density_per_m2
                              <= estimate_row["ci_high"])
        if isinstance(estimate_row.get("ci_low"), float) else "not_computable",
    }
