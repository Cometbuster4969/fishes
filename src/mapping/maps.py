"""Spatial outputs (spec section 10): GeoJSON, static PNG maps and an interactive
Folium HTML, all from detection points weighted by count.

Input frame: one row per detection group with `lat`, `lon`, `count`, `species`,
`date`. Everything else (KDE bandwidth, grid size, CRS) comes from
configs/kaggle.yaml -> mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("geofish")


def to_geodataframe(df: pd.DataFrame, crs_metric: str = "EPSG:32645"):
    import geopandas as gpd
    from shapely.geometry import Point

    if not {"lat", "lon"} <= set(df.columns):
        raise KeyError("detections need 'lat' and 'lon' columns")
    gdf = gpd.GeoDataFrame(df.copy(),
                           geometry=[Point(xy) for xy in zip(df.lon, df.lat)],
                           crs="EPSG:4326")
    return gdf.to_crs(crs_metric)


def grid_cells(gdf_metric, grid_m: float = 250.0, weight_col: str = "count",
               id_cols: list[str] | None = None):
    """Square grid density in metric CRS; returns a GeoDataFrame with counts."""
    import geopandas as gpd

    x, y = gdf_metric.geometry.x, gdf_metric.geometry.y
    gx = np.floor(x / grid_m).astype(int)
    gy = np.floor(y / grid_m).astype(int)
    g = gdf_metric.assign(cell_x=gx, cell_y=gy)
    keys = ["cell_x", "cell_y"] + (id_cols or [])
    agg = {weight_col: "sum"}
    cells = g.groupby(keys, as_index=False).agg(agg)
    cells["n_detections"] = g.groupby(keys).size().values
    cells["cell_id"] = cells["cell_x"].astype(str) + "_" + cells["cell_y"].astype(str)
    if id_cols:
        cells["cell_id"] = cells["cell_id"] + "_" + cells[id_cols].astype(str).agg(
            "_".join, axis=1)
    cells["area_m2"] = grid_m * grid_m
    cells["density_per_m2"] = cells[weight_col] / cells["area_m2"]
    boxes = [
        gpd.GeoSeries.from_wkt([
            f"POLYGON(({cx * grid_m} {cy * grid_m}, {(cx + 1) * grid_m} {cy * grid_m}, "
            f"{(cx + 1) * grid_m} {(cy + 1) * grid_m}, {cx * grid_m} {(cy + 1) * grid_m}, "
            f"{cx * grid_m} {cy * grid_m}))"
        ]).iloc[0]
        for cx, cy in zip(cells.cell_x, cells.cell_y)
    ]
    return gpd.GeoDataFrame(cells, geometry=boxes, crs=gdf_metric.crs)


def kde_surface(gdf_metric, weight_col: str = "count", bandwidth_m: float = 500.0,
                resolution: float = 100.0):
    """Gaussian KDE of weighted points on a regular metric grid."""
    from scipy.stats import gaussian_kde

    x = gdf_metric.geometry.x.values
    y = gdf_metric.geometry.y.values
    w = gdf_metric[weight_col].values.astype(float)
    if len(x) < 3:
        raise RuntimeError("need at least 3 points for a KDE surface")
    kde = gaussian_kde(np.vstack([x, y]), weights=w, bw_method=bandwidth_m / np.std(x, ddof=1))
    xi = np.arange(x.min() - 2 * bandwidth_m, x.max() + 2 * bandwidth_m, resolution)
    yi = np.arange(y.min() - 2 * bandwidth_m, y.max() + 2 * bandwidth_m, resolution)
    xx, yy = np.meshgrid(xi, yi)
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    return xi, yi, zz


def export_all(detections: pd.DataFrame, out_dir: Path, cfg) -> dict[str, Any]:
    """Write GeoJSON, per-species heatmaps (PNG), grid CSV and an interactive HTML."""
    import geopandas as gpd

    m = cfg.raw["mapping"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if detections.empty:
        return {"error": "no detections with coordinates - nothing to map"}
    if not {"lat", "lon"} <= set(detections.columns):
        return {"error": "detections lack lat/lon; cannot produce a spatial output"}

    gdf = to_geodataframe(detections, m["crs_metric"])
    outputs: dict[str, Any] = {}

    # ---- points + grid, GeoJSON in WGS84 ----
    pts = gdf.to_crs("EPSG:4326")
    p = out_dir / "detections.geojson"
    pts.to_file(p, driver="GeoJSON")
    outputs["points_geojson"] = str(p)

    cells = grid_cells(gdf, grid_m=float(m["grid_m"]))
    p = out_dir / "grid_density.geojson"
    cells.to_crs("EPSG:4326").to_file(p, driver="GeoJSON")
    cells.drop(columns="geometry").to_csv(out_dir / "grid_density.csv", index=False)
    outputs["grid_geojson"] = str(p)

    # ---- static KDE figure per species + overall ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [("all", gdf)] + [(sp, g) for sp, g in gdf.groupby("species")
                               if len(g) >= 3]
    for name, g in groups:
        try:
            xi, yi, zz = kde_surface(g, bandwidth_m=float(m["kde_bandwidth_m"]))
        except Exception as exc:  # noqa: BLE001
            log.warning("KDE skipped for %s: %s", name, exc)
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        h = ax.imshow(zz, origin="lower", extent=[xi[0], xi[-1], yi[0], yi[-1]],
                      cmap="viridis")
        cells.plot(ax=ax, facecolor="none", edgecolor="white", linewidth=0.2, alpha=0.4)
        g.plot(ax=ax, color="red", markersize=8, alpha=0.6)
        fig.colorbar(h, ax=ax, label="KDE (weighted detections / m^2)")
        ax.set_title(f"{name} density surface (bandwidth {m['kde_bandwidth_m']} m)")
        ax.set_xlabel("easting (m)")
        ax.set_ylabel("northing (m)")
        safe = str(name).replace("/", "_").replace(" ", "_")
        fp = out_dir / f"heatmap_{safe}.png"
        fig.tight_layout()
        fig.savefig(fp, dpi=140)
        plt.close(fig)
        outputs.setdefault("heatmap_png", []).append(str(fp))

    # ---- interactive folium map ----
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    centre = [float(pts.lat.mean()), float(pts.lon.mean())]
    fmap = folium.Map(location=centre, zoom_start=12, tiles=m["basemap"])
    heat = [[r.lat, r.lon, float(getattr(r, "count", 1))] for r in detections.itertuples()]
    HeatMap(heat, radius=18, blur=14, min_opacity=0.3).add_to(fmap)
    cluster = MarkerCluster(name="detections").add_to(fmap)
    for r in detections.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=4, color="#c0392b", fill=True,
                            popup=f"{getattr(r, 'species', 'fish')} x{getattr(r, 'count', 1)}"
                                  f"<br>{getattr(r, 'date', '')}").add_to(cluster)
    folium.LayerControl().add_to(fmap)
    html = out_dir / "density_map.html"
    fmap.save(str(html))
    outputs["interactive_html"] = str(html)

    # ---- spatio-temporal facets ----
    if "date" in detections.columns and detections["date"].notna().any():
        d = detections.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["week"] = d["date"].dt.isocalendar().week
        weeks = sorted(d["week"].dropna().unique())
        if len(weeks) >= 2:
            ncol = min(4, len(weeks))
            nrow = int(np.ceil(len(weeks) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.6 * nrow),
                                     squeeze=False)
            gdf_w = gdf.assign(week=d["week"].values)
            for ax, wk in zip(axes.ravel(), weeks):
                sub = gdf_w[gdf_w.week == wk]
                if len(sub):
                    sub.plot(ax=ax, markersize=10 + 4 * np.log1p(sub["count"]),
                             color="#2a7ab9")
                ax.set_title(f"week {int(wk)} (n={len(sub)})", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in axes.ravel()[len(weeks):]:
                ax.axis("off")
            fp = out_dir / "weekly_facets.png"
            fig.tight_layout()
            fig.savefig(fp, dpi=130)
            plt.close(fig)
            outputs["weekly_facets"] = str(fp)

    log.info("map outputs: %s", {k: v for k, v in outputs.items() if isinstance(v, str)})
    return outputs
