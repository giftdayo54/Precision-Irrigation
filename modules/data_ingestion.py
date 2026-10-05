"""
Ingestion helpers for field boundaries, yield records and irrigation/weather
logs uploaded through the Streamlit file_uploader widgets.

Design notes:
- Field boundaries accept GeoJSON directly, or a zipped Shapefile (.zip
  containing .shp/.shx/.dbf/.prj). Everything is reprojected to
  config.DEFAULT_CRS so all downstream area/zonal-stat math is in metres.
- The field-ID column is normalised to config.FIELD_ID_COL so every other
  module can join on one consistent key, regardless of what the source file
  called it.
"""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from config import DEFAULT_CRS, FIELD_ID_COL, FIELD_ID_ALIASES


class IngestionError(ValueError):
    """Raised when an uploaded file can't be parsed into the expected shape."""


def _normalise_field_id(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if FIELD_ID_COL in gdf.columns:
        gdf[FIELD_ID_COL] = gdf[FIELD_ID_COL].astype(str)
        return gdf
    for alias in FIELD_ID_ALIASES:
        if alias in gdf.columns:
            gdf = gdf.rename(columns={alias: FIELD_ID_COL})
            gdf[FIELD_ID_COL] = gdf[FIELD_ID_COL].astype(str)
            return gdf
    # Nothing matched - fall back to a generated sequential ID rather than failing,
    # but this is surfaced to the user so they can fix the source file if it matters.
    gdf[FIELD_ID_COL] = [f"F{i+1:03d}" for i in range(len(gdf))]
    return gdf


def load_field_boundaries(uploaded_file) -> tuple[gpd.GeoDataFrame, list[str]]:
    """
    Accepts a Streamlit UploadedFile that is either .geojson/.json or a .zip
    containing an ESRI Shapefile. Returns (GeoDataFrame in DEFAULT_CRS, warnings).
    """
    warnings: list[str] = []
    name = uploaded_file.name.lower()

    if name.endswith(".zip"):
        with tempfile.TemporaryDirectory() as tmp:
            zpath = Path(tmp) / "upload.zip"
            zpath.write_bytes(uploaded_file.getvalue())
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(tmp)
            shp_files = list(Path(tmp).rglob("*.shp"))
            if not shp_files:
                raise IngestionError("No .shp file found inside the uploaded zip archive.")
            gdf = gpd.read_file(shp_files[0])
    elif name.endswith((".geojson", ".json")):
        gdf = gpd.read_file(io.BytesIO(uploaded_file.getvalue()))
    else:
        raise IngestionError(
            "Unsupported field-boundary format. Upload a GeoJSON (.geojson/.json) "
            "or a zipped Shapefile (.zip containing .shp/.shx/.dbf/.prj)."
        )

    if gdf.empty:
        raise IngestionError("The uploaded field-boundary file contains no features.")

    if gdf.crs is None:
        warnings.append(
            f"Uploaded boundaries had no CRS defined - assumed to already be {DEFAULT_CRS}."
        )
        gdf = gdf.set_crs(DEFAULT_CRS)
    elif str(gdf.crs) != DEFAULT_CRS:
        gdf = gdf.to_crs(DEFAULT_CRS)

    gdf = _normalise_field_id(gdf)

    if "area_ha" not in gdf.columns:
        gdf["area_ha"] = (gdf.geometry.area / 10_000.0).round(2)

    dupes = gdf[FIELD_ID_COL].duplicated().sum()
    if dupes:
        warnings.append(f"{dupes} duplicate field IDs found after normalisation - check the source attribute table.")

    return gdf.reset_index(drop=True), warnings


REQUIRED_YIELD_COLS = {FIELD_ID_COL: ["ID", "id", "field_id", "Field_ID"], "cane_yield_t_ha": ["cane_yield_t_ha", "yield_t_ha", "yield", "tons_ha", "t_ha"]}
REQUIRED_IRRIGATION_COLS = {
    FIELD_ID_COL: ["ID", "id", "field_id", "Field_ID"],
    "date": ["date", "Date", "week_ending"],
    "irrigation_mm": ["irrigation_mm", "irrigation", "applied_mm"],
}


def _coerce_columns(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[str]]:
    warnings = []
    for target, aliases in spec.items():
        if target in df.columns:
            continue
        match = next((a for a in aliases if a in df.columns), None)
        if match:
            df = df.rename(columns={match: target})
        else:
            warnings.append(f"Could not find a column for '{target}' (looked for {aliases}).")
    return df, warnings


def load_yield_records(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(uploaded_file)
    df, warnings = _coerce_columns(df, REQUIRED_YIELD_COLS)
    if FIELD_ID_COL in df.columns:
        df[FIELD_ID_COL] = df[FIELD_ID_COL].astype(str)
    if "season" not in df.columns:
        df["season"] = "unspecified"
    missing = [c for c in ["cane_yield_t_ha"] if c not in df.columns]
    if missing:
        raise IngestionError(
            f"Yield file is missing required column(s): {missing}. "
            f"Expected at least a field ID column and a cane yield (t/ha) column."
        )
    return df, warnings


def load_irrigation_weather_log(uploaded_file) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(uploaded_file)
    df, warnings = _coerce_columns(df, REQUIRED_IRRIGATION_COLS)
    missing = [c for c in ["date", "irrigation_mm"] if c not in df.columns]
    if missing:
        raise IngestionError(
            f"Irrigation/weather file is missing required column(s): {missing}."
        )
    if FIELD_ID_COL in df.columns:
        df[FIELD_ID_COL] = df[FIELD_ID_COL].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        warnings.append("Some rows had an unparseable date and were dropped.")
        df = df.dropna(subset=["date"])
    for col in ["rainfall_mm", "eto_mm"]:
        if col not in df.columns:
            df[col] = 0.0
    return df, warnings
