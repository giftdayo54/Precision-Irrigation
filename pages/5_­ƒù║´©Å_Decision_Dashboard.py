"""GIS-based decision-support dashboard: combined layers + prioritised field list."""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import ml_models, water_productivity, visualization
from config import FIELD_ID_COL

st.set_page_config(page_title="Decision Dashboard", page_icon="🗺️", layout="wide")
st.title("🗺️ Precision Irrigation Decision-Support Dashboard")

fields_gdf = st.session_state.get("fields")
indices_df = st.session_state.get("indices")
yield_df = st.session_state.get("yield_df")
irrigation_df = st.session_state.get("irrigation_df")

if fields_gdf is None or indices_df is None:
    st.info("Load field boundaries and vegetation indices first (home page / Satellite Monitoring page).")
    st.stop()

if st.session_state.get("is_synthetic"):
    st.warning("⚠️ DEMO DATA loaded - this dashboard is showing synthetic fields, not a real estate.", icon="⚠️")

# --- Assemble whatever layers are available, computing anything missing on the fly ---
latest_ndvi = (
    indices_df[indices_df["index"] == "NDVI"].sort_values("date").groupby(FIELD_ID_COL).last().reset_index()[[FIELD_ID_COL, "value"]]
    .rename(columns={"value": "latest_ndvi"})
)
view = fields_gdf.merge(latest_ndvi, on=FIELD_ID_COL, how="left")

can_build_features = yield_df is not None and irrigation_df is not None
if can_build_features:
    features = st.session_state.get("feature_table") or ml_models.build_feature_table(indices_df, irrigation_df, yield_df)
    st.session_state["feature_table"] = features

    try:
        iforest = ml_models.detect_anomalies_isolation_forest(features)
        view = view.merge(iforest[[FIELD_ID_COL, "is_anomaly", "anomaly_score"]], on=FIELD_ID_COL, how="left")
    except ValueError:
        view["is_anomaly"] = False

    wp_raw = water_productivity.compute_water_productivity(features)
    wp_df = water_productivity.classify_wp_performance(wp_raw)
    view = view.merge(wp_df[[FIELD_ID_COL, "wp_tier", "water_productivity_t_ha_per_100mm"]], on=FIELD_ID_COL, how="left")
else:
    view["is_anomaly"] = False
    view["wp_tier"] = None

temporal_result = st.session_state.get("anomaly_temporal")
if temporal_result is not None and not temporal_result.empty:
    view = view.merge(temporal_result[[FIELD_ID_COL, "is_anomaly"]].rename(columns={"is_anomaly": "temporal_anomaly"}), on=FIELD_ID_COL, how="left")
else:
    view["temporal_anomaly"] = False

view["is_anomaly"] = view["is_anomaly"].fillna(False)
view["temporal_anomaly"] = view["temporal_anomaly"].fillna(False)
view["needs_investigation"] = view["is_anomaly"] | view["temporal_anomaly"] | (view["wp_tier"] == "Low")

st.divider()
layer = st.selectbox("Map layer", ["Latest NDVI", "Anomaly flag (Isolation Forest)", "Water productivity tier", "Needs investigation"])

left, right = st.columns([2, 1])
with left:
    if layer == "Latest NDVI":
        m = visualization.field_choropleth(view, "latest_ndvi", "Latest NDVI")
    elif layer == "Anomaly flag (Isolation Forest)":
        m = visualization.anomaly_hotspot_map(view, "is_anomaly")
    elif layer == "Water productivity tier":
        tier_map = {"Low": 1, "Medium": 2, "High": 3}
        view["_tier_num"] = view["wp_tier"].map(tier_map)
        m = visualization.field_choropleth(view, "_tier_num", "WP tier (1=Low,3=High)", fill_color="RdYlGn")
    else:
        view["_needs"] = view["needs_investigation"].map({True: 1, False: 0})
        m = visualization.field_choropleth(view, "_needs", "Needs investigation", fill_color="YlOrRd")
    st_folium(m, height=460, use_container_width=True)

with right:
    st.subheader("Priority field list")
    priority_cols = [FIELD_ID_COL, "latest_ndvi", "is_anomaly", "temporal_anomaly", "wp_tier", "needs_investigation"]
    priority = view[priority_cols].sort_values("needs_investigation", ascending=False)
    st.dataframe(priority, use_container_width=True, height=420)

n_priority = int(view["needs_investigation"].sum())
st.metric("Fields flagged for field investigation", n_priority)

st.divider()
st.subheader("Export priority zones")
export_df = view.drop(columns=[c for c in ["_tier_num", "_needs"] if c in view.columns])
c1, c2 = st.columns(2)
with c1:
    csv = export_df.drop(columns="geometry").to_csv(index=False).encode("utf-8")
    st.download_button("Download priority list (CSV)", csv, file_name="priority_fields.csv", mime="text/csv")
with c2:
    geojson = export_df.to_crs("EPSG:4326").to_json()
    st.download_button("Download priority zones (GeoJSON)", geojson, file_name="priority_zones.geojson", mime="application/geo+json")

st.caption(
    "This combines the Isolation Forest cross-sectional flag, the temporal baseline t-statistic flag "
    "(if run on the ML page) and the Low water-productivity tier into one 'needs investigation' rule. "
    "Treat it as a triage list for agronomist follow-up, not an automated treatment decision."
)
