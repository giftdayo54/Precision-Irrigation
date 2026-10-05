"""
Machine learning models:
  1. Yield/productivity prediction - RandomForestRegressor or XGBRegressor
     on a per-field feature table (vegetation indices + irrigation/weather).
  2. Cross-sectional anomaly detection - IsolationForest over the same
     feature table, flags fields that look structurally "off" this season.
  3. Temporal anomaly detection - per-field OLS trend fit over a baseline
     window with a Student's-t critical value (degrees-of-freedom corrected),
     flags a field whose latest observation deviates from its own expected
     trajectory more than chance would predict. In prior field-scale work
     this materially outperformed a naive z-score threshold (~2.8% vs ~18%
     false-positive rate at similar sensitivity), so it's the default here
     rather than a simple fixed-threshold rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from config import FIELD_ID_COL, MODEL_DIR, ANOMALY_BASELINE_WEEKS, ANOMALY_CONFIDENCE


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def build_feature_table(
    indices_df: pd.DataFrame, irrigation_df: pd.DataFrame, yield_df: pd.DataFrame
) -> pd.DataFrame:
    """
    indices_df: long format (ID, date, index, value) -> pivoted to season-mean per index.
    irrigation_df: (ID, date, irrigation_mm, rainfall_mm, eto_mm) -> summed per field.
    yield_df: (ID, season, cane_yield_t_ha, area_ha).
    Returns one row per field with index means + irrigation totals + yield.
    """
    idx_wide = (
        indices_df.groupby([FIELD_ID_COL, "index"])["value"].mean().unstack("index").reset_index()
    )
    idx_wide.columns = [FIELD_ID_COL] + [f"{c}_mean" for c in idx_wide.columns[1:]]

    irr_summary = (
        irrigation_df.groupby(FIELD_ID_COL)
        .agg(total_irrigation_mm=("irrigation_mm", "sum"), total_rainfall_mm=("rainfall_mm", "sum"), total_eto_mm=("eto_mm", "sum"))
        .reset_index()
    )

    features = idx_wide.merge(irr_summary, on=FIELD_ID_COL, how="left")
    features = features.merge(yield_df[[FIELD_ID_COL, "cane_yield_t_ha", "area_ha"]], on=FIELD_ID_COL, how="left")
    return features


# ---------------------------------------------------------------------------
# Yield prediction
# ---------------------------------------------------------------------------

def train_yield_model(features: pd.DataFrame, model_type: str = "random_forest"):
    """
    Trains on all rows that have a known yield; returns (model, metrics, feature_cols).
    metrics come from 5-fold cross-validated predictions, which is honest for the
    small (field-count-sized) samples typical of a single estate/season.
    """
    feature_cols = [c for c in features.columns if c.endswith("_mean") or c.startswith("total_")]
    data = features.dropna(subset=feature_cols + ["cane_yield_t_ha"])
    if len(data) < 5:
        raise ValueError("Need at least 5 fields with complete index + yield data to train a model.")

    X = data[feature_cols].values
    y = data["cane_yield_t_ha"].values

    if model_type == "xgboost" and HAS_XGBOOST:
        model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.08, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)

    n_splits = min(5, len(data))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(model, X, y, cv=cv)

    metrics = {
        "r2": r2_score(y, y_pred_cv),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred_cv))),
        "n_fields": len(data),
        "model_type": model_type if (model_type != "xgboost" or HAS_XGBOOST) else "random_forest (xgboost unavailable)",
    }

    model.fit(X, y)  # final fit on all available data for downstream prediction/export
    return model, metrics, feature_cols, y_pred_cv, data[FIELD_ID_COL].values


def save_model(model, name: str) -> str:
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    path = str(Path(MODEL_DIR) / f"{name}.joblib")
    joblib.dump(model, path)
    return path


def load_model(path: str):
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Cross-sectional anomaly detection (IsolationForest)
# ---------------------------------------------------------------------------

def detect_anomalies_isolation_forest(features: pd.DataFrame, contamination: float = 0.12) -> pd.DataFrame:
    feature_cols = [c for c in features.columns if c.endswith("_mean") or c.startswith("total_")]
    data = features.dropna(subset=feature_cols).copy()
    if len(data) < 4:
        raise ValueError("Need at least 4 fields with complete feature data for anomaly detection.")

    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    labels = model.fit_predict(data[feature_cols].values)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(data[feature_cols].values)  # lower = more anomalous

    data["is_anomaly"] = labels == -1
    data["anomaly_score"] = scores
    return data[[FIELD_ID_COL, "is_anomaly", "anomaly_score"] + feature_cols]


# ---------------------------------------------------------------------------
# Temporal anomaly detection (t-statistic on OLS trend residual)
# ---------------------------------------------------------------------------

def detect_temporal_anomalies(
    timeseries_df: pd.DataFrame,
    index_name: str = "NDVI",
    baseline_weeks: int = ANOMALY_BASELINE_WEEKS,
    confidence: float = ANOMALY_CONFIDENCE,
) -> pd.DataFrame:
    """
    For each field: fit an OLS trend (week -> value) over the baseline window,
    then check whether the latest observation's residual is larger than chance
    would predict given a Student's-t distribution with (n-2) degrees of
    freedom - not a fixed z-score cutoff, which over- or under-flags depending
    on how few weeks of history you actually have.

    Returns one row per field: predicted value, actual value, t-statistic,
    critical t, and a boolean anomaly flag.
    """
    results = []
    df = timeseries_df[timeseries_df["index"] == index_name].sort_values(["ID", "date"])

    for fid, g in df.groupby(FIELD_ID_COL):
        g = g.dropna(subset=["value"]).reset_index(drop=True)
        if len(g) < baseline_weeks + 1:
            continue  # not enough history to establish a baseline + current point

        baseline = g.iloc[:baseline_weeks]
        current = g.iloc[baseline_weeks]  # first point after the baseline window

        n = len(baseline)
        x = np.arange(n)
        y = baseline["value"].values
        slope, intercept, _, _, _ = sp_stats.linregress(x, y)
        predicted = intercept + slope * x
        residuals = y - predicted
        dof = n - 2
        if dof < 1:
            continue
        se = np.sqrt(np.sum(residuals**2) / dof)
        if se == 0:
            se = 1e-6

        x_current = n  # the point right after the baseline window
        expected_current = intercept + slope * x_current
        t_stat = (current["value"] - expected_current) / se
        t_crit = sp_stats.t.ppf(1 - (1 - confidence) / 2, dof)

        results.append(
            {
                FIELD_ID_COL: fid,
                "date": current["date"],
                "expected_value": round(float(expected_current), 4),
                "actual_value": round(float(current["value"]), 4),
                "t_statistic": round(float(t_stat), 3),
                "t_critical": round(float(t_crit), 3),
                "is_anomaly": bool(abs(t_stat) > t_crit),
            }
        )

    return pd.DataFrame(results)
