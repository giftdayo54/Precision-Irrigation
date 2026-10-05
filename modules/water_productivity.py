"""
Water productivity (WP) assessment.

Physical WP = crop output (kg cane) / water applied+received (m3), expressed
here as t/ha per 100 mm for readability. When irrigation is metered well
enough that total water input is reliable, this is a direct measurement;
otherwise treat it as a remote-sensing-informed proxy and label it as such
in the UI (per objective 6 / the proposal's own distinction between direct
measurement and proxy estimation).
"""

from __future__ import annotations

import pandas as pd

from config import FIELD_ID_COL


def compute_water_productivity(features: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: ID, cane_yield_t_ha, area_ha, total_irrigation_mm,
    total_rainfall_mm (from ml_models.build_feature_table output).
    """
    df = features.copy()
    df["total_water_mm"] = df["total_irrigation_mm"].fillna(0) + df["total_rainfall_mm"].fillna(0)
    # t/ha yield per 100mm of total water applied+received - a simple, readable
    # physical water-productivity indicator; swap in ET-based or economic WP
    # once you have reliable evapotranspiration / price data.
    df["water_productivity_t_ha_per_100mm"] = (
        df["cane_yield_t_ha"] / (df["total_water_mm"].replace(0, pd.NA) / 100.0)
    )
    return df[[FIELD_ID_COL, "cane_yield_t_ha", "total_water_mm", "water_productivity_t_ha_per_100mm"]]


def classify_wp_performance(wp_df: pd.DataFrame) -> pd.DataFrame:
    """Tercile-based tiering (Low/Medium/High) - simple, robust for small field counts."""
    df = wp_df.dropna(subset=["water_productivity_t_ha_per_100mm"]).copy()
    if len(df) < 3:
        df["wp_tier"] = "Insufficient data"
        return df
    try:
        df["wp_tier"] = pd.qcut(
            df["water_productivity_t_ha_per_100mm"], q=3, labels=["Low", "Medium", "High"], duplicates="drop"
        )
    except ValueError:
        # Too many tied values for 3 clean bins (common with very small/synthetic
        # samples) - fall back to a simple median split rather than failing.
        median = df["water_productivity_t_ha_per_100mm"].median()
        df["wp_tier"] = df["water_productivity_t_ha_per_100mm"].apply(lambda v: "High" if v >= median else "Low")
    # Cast away from pandas Categorical (qcut's return dtype) to a plain string
    # column - callers merge/map/fillna this freely without tripping over
    # "new category not in categories" errors.
    df["wp_tier"] = df["wp_tier"].astype(str)
    return df
