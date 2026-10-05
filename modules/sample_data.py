"""
Synthetic DEMO data generator.

IMPORTANT: this module exists so the app is fully explorable before you have
real Sentinel-2 credentials, UAV flights or irrigation logs loaded. Every
dataset it returns is clearly synthetic and must never be presented to the
user as real observations - the UI is responsible for showing a persistent
"DEMO DATA" banner whenever these functions are the active data source.
(A previous project of ours learned this the hard way: a silent
hash/random-noise fallback can look plausible while being completely
spurious. Nothing here is silent - every call site must label it.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

from config import DEFAULT_CRS, FIELD_ID_COL

RNG = np.random.default_rng(42)


def generate_synthetic_fields(n_fields: int = 12, cols: int = 4) -> gpd.GeoDataFrame:
    """Simple grid of rectangular 'fields' so demo mode has geometry to map."""
    rows = int(np.ceil(n_fields / cols))
    cell = 400.0  # metres
    gap = 40.0
    records = []
    x0, y0 = 500000.0, 8100000.0  # arbitrary UTM-ish origin
    fid = 0
    for r in range(rows):
        for c in range(cols):
            if fid >= n_fields:
                break
            minx = x0 + c * (cell + gap)
            miny = y0 - r * (cell + gap)
            geom = box(minx, miny, minx + cell, miny + cell)
            area_ha = geom.area / 10_000.0
            records.append(
                {
                    FIELD_ID_COL: f"F{fid + 1:02d}",
                    "area_ha": round(area_ha, 2),
                    "variety": RNG.choice(["N19", "N41", "NCo376", "N25"]),
                    "ratoon": int(RNG.integers(0, 6)),
                    "geometry": geom,
                }
            )
            fid += 1
    gdf = gpd.GeoDataFrame(records, crs=DEFAULT_CRS)
    return gdf


def generate_synthetic_index_timeseries(
    fields_gdf: gpd.GeoDataFrame, weeks: int = 12, index_name: str = "NDVI"
) -> pd.DataFrame:
    """
    Long-format DataFrame: field_id, week, date, value.
    Two or three fields get an injected stress/anomaly dip in the final weeks
    so the anomaly-detection and dashboard pages have something real to find.
    """
    field_ids = fields_gdf[FIELD_ID_COL].tolist()
    anomalous = set(RNG.choice(field_ids, size=max(1, len(field_ids) // 5), replace=False))

    today = pd.Timestamp.today().normalize()
    dates = [today - pd.Timedelta(weeks=(weeks - 1 - w)) for w in range(weeks)]

    rows = []
    base_range = {"NDVI": (0.35, 0.82), "EVI": (0.2, 0.6), "SAVI": (0.25, 0.65), "NDRE": (0.1, 0.4)}
    lo, hi = base_range.get(index_name, (0.3, 0.8))

    for fid in field_ids:
        # smooth growth curve + noise
        growth = np.linspace(lo, hi, weeks) + RNG.normal(0, 0.02, weeks)
        if fid in anomalous:
            dip_start = int(weeks * 0.7)
            growth[dip_start:] -= np.linspace(0.05, 0.22, weeks - dip_start)
        growth = np.clip(growth, 0.02, 0.95)
        for w, (d, v) in enumerate(zip(dates, growth)):
            rows.append({FIELD_ID_COL: fid, "week": w + 1, "date": d, "index": index_name, "value": round(float(v), 4)})

    return pd.DataFrame(rows)


def generate_synthetic_yield(fields_gdf: gpd.GeoDataFrame, season: str = "2025/26") -> pd.DataFrame:
    rows = []
    for _, f in fields_gdf.iterrows():
        base = RNG.normal(95, 12)  # t/ha cane yield, typical irrigated estate range
        rows.append(
            {
                FIELD_ID_COL: f[FIELD_ID_COL],
                "season": season,
                "cane_yield_t_ha": round(max(30, base), 1),
                "area_ha": f["area_ha"],
            }
        )
    return pd.DataFrame(rows)


def generate_synthetic_irrigation_weather(
    fields_gdf: gpd.GeoDataFrame, weeks: int = 12, season: str = "2025/26"
) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    dates = [today - pd.Timedelta(weeks=(weeks - 1 - w)) for w in range(weeks)]
    rows = []
    for _, f in fields_gdf.iterrows():
        for d in dates:
            rows.append(
                {
                    FIELD_ID_COL: f[FIELD_ID_COL],
                    "season": season,
                    "date": d,
                    "irrigation_mm": round(max(0, RNG.normal(28, 8)), 1),
                    "rainfall_mm": round(max(0, RNG.normal(6, 5)), 1),
                    "eto_mm": round(max(1, RNG.normal(30, 4)), 1),
                }
            )
    return pd.DataFrame(rows)


def generate_full_demo_dataset(n_fields: int = 12, weeks: int = 12) -> dict:
    fields = generate_synthetic_fields(n_fields)
    indices = pd.concat(
        [generate_synthetic_index_timeseries(fields, weeks, idx) for idx in ["NDVI", "EVI", "SAVI", "NDRE"]],
        ignore_index=True,
    )
    yield_df = generate_synthetic_yield(fields)
    irrigation_df = generate_synthetic_irrigation_weather(fields, weeks)
    return {
        "fields": fields,
        "indices": indices,
        "yield": yield_df,
        "irrigation": irrigation_df,
        "is_synthetic": True,
    }
