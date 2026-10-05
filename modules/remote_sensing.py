"""
Vegetation index calculation, cloud/SCL masking and zonal statistics.

Works against either:
  (a) a locally uploaded multi-band GeoTIFF (the common path when you don't
      yet have live CDSE credentials wired up), or
  (b) raw band arrays returned by modules.sentinel_hub_client.

All index math is plain numpy so it is independent of where the bands came
from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.io import MemoryFile
from rasterstats import zonal_stats

from config import SCL_MASK_CODES, FIELD_ID_COL

EPS = 1e-6


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + EPS)


def evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + EPS)


def savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    return ((nir - red) / (nir + red + L + EPS)) * (1 + L)


def ndre(nir: np.ndarray, rededge: np.ndarray) -> np.ndarray:
    return (nir - rededge) / (nir + rededge + EPS)


def compute_all_indices(blue: np.ndarray, red: np.ndarray, rededge: np.ndarray, nir: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "NDVI": ndvi(nir, red),
        "EVI": evi(nir, red, blue),
        "SAVI": savi(nir, red),
        "NDRE": ndre(nir, rededge),
    }


def apply_scl_mask(index_array: np.ndarray, scl_array: np.ndarray) -> np.ndarray:
    """Set pixels flagged as cloud/shadow/nodata/snow (per config.SCL_MASK_CODES) to NaN."""
    out = index_array.astype("float32").copy()
    mask = np.isin(scl_array, SCL_MASK_CODES)
    out[mask] = np.nan
    return out


def read_geotiff_bands(file_bytes: bytes, band_order: list[str]) -> dict:
    """
    Read an uploaded multi-band GeoTIFF. `band_order` maps band index (1-based,
    matching the order in the file) to a name, e.g. ["blue","green","red","rededge","nir"].
    Returns dict with 'arrays' (name -> 2D array), 'transform', 'crs'.
    """
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            arrays = {}
            for i, name in enumerate(band_order, start=1):
                if i > src.count:
                    break
                arrays[name] = src.read(i).astype("float32")
            transform = src.transform
            crs = src.crs
    return {"arrays": arrays, "transform": transform, "crs": crs}


def zonal_mean_per_field(
    index_array: np.ndarray, transform, crs, fields_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Per-field mean/std/min/max of a single index raster, reprojecting fields to match the raster CRS."""
    fields_in_raster_crs = fields_gdf.to_crs(crs) if str(fields_gdf.crs) != str(crs) else fields_gdf
    stats = zonal_stats(
        fields_in_raster_crs,
        index_array,
        affine=transform,
        stats=["mean", "std", "min", "max"],
        nodata=np.nan,
    )
    df = pd.DataFrame(stats)
    df[FIELD_ID_COL] = fields_gdf[FIELD_ID_COL].values
    return df[[FIELD_ID_COL, "mean", "std", "min", "max"]]


def build_index_timeseries(
    scenes: list[dict], fields_gdf: gpd.GeoDataFrame, index_name: str
) -> pd.DataFrame:
    """
    scenes: list of {"date": <date>, "index_array": np.ndarray, "transform":..., "crs":...}
    already computed + SCL-masked for `index_name`.
    Returns long-format DataFrame: ID, date, index, value (=zonal mean).
    """
    rows = []
    for scene in scenes:
        zdf = zonal_mean_per_field(scene["index_array"], scene["transform"], scene["crs"], fields_gdf)
        for _, r in zdf.iterrows():
            rows.append(
                {
                    FIELD_ID_COL: r[FIELD_ID_COL],
                    "date": scene["date"],
                    "index": index_name,
                    "value": r["mean"],
                }
            )
    return pd.DataFrame(rows).sort_values([FIELD_ID_COL, "date"]).reset_index(drop=True)
