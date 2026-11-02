"""JH-Fish field schema + IoT sensor ingestion (spec section 9).

* `FIELD_SCHEMA` is the one row per clip/image contract.
* `load_sensor_csv` / `load_mqtt_log` read logger output (CSV, JSON lines or the
  common "key=value" MQTT log dump).
* `join_by_time_gps` attaches sensor readings to detections (nearest timestamp
  within tolerance, same water body when known).
* `presence_model` fits a gradient-boosting baseline (then an optional DNN) of
  species presence vs water parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("geofish")

FIELD_SCHEMA = [
    "id", "date", "time", "lat", "lon", "water_body", "water_body_type", "depth_m",
    "temp_C", "pH", "DO_mgL", "turbidity_NTU", "EC", "TDS", "camera_coverage_m2",
    "species", "count", "bbox", "confidence",
]

SENSOR_COLUMNS = ["timestamp", "temp_C", "pH", "DO_mgL", "turbidity_NTU", "EC", "TDS",
                  "depth_m", "lat", "lon"]


def empty_field_frame() -> pd.DataFrame:
    """Schema-correct empty frame - used so downstream cells never crash on no data."""
    return pd.DataFrame(columns=FIELD_SCHEMA)


def validate_field_frame(df: pd.DataFrame) -> dict[str, Any]:
    missing = [c for c in FIELD_SCHEMA if c not in df.columns]
    return {"ok": not missing, "missing": missing, "rows": len(df)}


def load_sensor_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "timestamp" not in df.columns:
        for cand in ("time", "datetime", "ts"):
            if cand in df.columns:
                df = df.rename(columns={cand: "timestamp"})
                break
    if "timestamp" not in df.columns:
        raise KeyError(f"{path} has no timestamp column (looked for timestamp/time/datetime/ts)")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    log.info("sensor csv %s: %d rows, %s..%s", path, len(df),
             df.timestamp.min(), df.timestamp.max())
    return df


def load_mqtt_log(path: str | Path) -> pd.DataFrame:
    """JSON-lines, JSON array, or `key=value,key=value` text dump."""
    p = Path(path)
    text = p.read_text(errors="ignore").strip()
    rows: list[dict[str, Any]] = []
    if text.startswith("["):
        rows = json.loads(text)
    elif text.startswith("{"):
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        for line in text.splitlines():
            if not line.strip():
                continue
            rec: dict[str, Any] = {}
            head, _, payload = line.partition("{")
            parts = payload.rstrip("}").replace(":", "=").split(",")
            for part in parts:
                if "=" in part:
                    k, v = part.split("=", 1)
                    try:
                        rec[k.strip()] = float(v)
                    except ValueError:
                        rec[k.strip()] = v.strip()
            if rec:
                rows.append(rec)
    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns:
        raise KeyError(f"{p} log records have no timestamp field")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def join_by_time_gps(detections: pd.DataFrame, sensors: pd.DataFrame,
                     tolerance_s: int = 600, max_gps_km: float = 2.0) -> pd.DataFrame:
    """Nearest-in-time sensor reading per detection (optionally GPS-constrained)."""
    if detections.empty:
        return detections.assign(**{c: np.nan for c in SENSOR_COLUMNS if c in sensors})
    d = detections.copy()
    d["timestamp"] = pd.to_datetime(
        d["date"].astype(str) + " " + d.get("time", "00:00:00").astype(str), errors="coerce",
        utc=True)
    cols = [c for c in SENSOR_COLUMNS if c in sensors.columns and c != "timestamp"]
    out = []
    for _, row in d.iterrows():
        if pd.isna(row["timestamp"]):
            out.append({c: np.nan for c in cols})
            continue
        cand = sensors
        if {"lat", "lon"} <= set(sensors.columns) and not pd.isna(row.get("lat", np.nan)):
            dist = _haversine(sensors.lat.values, sensors.lon.values, row["lat"], row["lon"])
            cand = sensors[dist <= max_gps_km]
            if cand.empty:
                cand = sensors
        if cand.empty:
            out.append({c: np.nan for c in cols})
            continue
        dt = (cand["timestamp"] - row["timestamp"]).abs()
        if dt.min() > pd.Timedelta(seconds=tolerance_s):
            out.append({c: np.nan for c in cols})
            continue
        out.append(cand.loc[dt.idxmin(), cols].to_dict())
    merged = pd.concat([d.reset_index(drop=True), pd.DataFrame(out)], axis=1)
    n_joined = int(merged[cols[0]].notna().sum()) if cols else 0
    log.info("joined sensor data for %d/%d detections", n_joined, len(merged))
    return merged


def _haversine(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def presence_model(field: pd.DataFrame, target: str = "species",
                   features: list[str] | None = None, dnn: bool = False
                   ) -> dict[str, Any]:
    """Gradient-boosting (LightGBM) baseline of presence vs water parameters.

    Returns metrics + feature importances; a small torch MLP is fitted too when
    `dnn=True` so the two can be compared.
    """
    feats = features or [c for c in ("temp_C", "pH", "DO_mgL", "turbidity_NTU", "EC", "TDS",
                                     "depth_m") if c in field.columns]
    need = [target] + feats
    missing = [c for c in need if c not in field.columns]
    if missing:
        return {"error": f"not_computable: field data missing columns {missing}"}
    df = field.dropna(subset=need)
    if len(df) < 30:
        return {"error": f"not_computable: only {len(df)} complete rows (need >= 30)"}

    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    X = df[feats].astype(float).values
    y = df[target].astype(str).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42,
                                          stratify=y)
    import lightgbm as lgb

    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                               subsample=0.9, colsample_bytree=0.9, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    result: dict[str, Any] = {
        "target": target, "features": feats, "n_rows": int(len(df)),
        "n_classes": int(len(set(y))),
        "lightgbm": {"accuracy": float(accuracy_score(yte, pred)),
                     "f1_macro": float(f1_score(yte, pred, average="macro",
                                                zero_division=0))},
        "feature_importance": dict(sorted(zip(feats, model.feature_importances_.tolist()),
                                          key=lambda kv: -kv[1])),
    }
    if dnn:
        import torch
        import torch.nn as nn

        classes = sorted(set(ytr))
        cls_idx = {c: i for i, c in enumerate(classes)}
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        net = nn.Sequential(nn.Linear(len(feats), 64), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, len(classes)))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
        yt = torch.tensor([cls_idx[c] for c in ytr], dtype=torch.long)
        xv = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
        for _ in range(300):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(net(xt), yt)
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred_dnn = [classes[i] for i in net(xv).argmax(1).tolist()]
        result["dnn"] = {"accuracy": float(accuracy_score(yte, pred_dnn)),
                         "f1_macro": float(f1_score(yte, pred_dnn, average="macro",
                                                    zero_division=0))}
    return result
