"""Water productivity assessment module."""

import streamlit as st
from streamlit_folium import st_folium

from modules import ml_models, water_productivity, visualization
from config import FIELD_ID_COL

st.set_page_config(page_title="Water Productivity", page_icon="💧", layout="wide")
st.title("💧 Sugarcane Water Productivity Assessment")

fields_gdf = st.session_state.get("fields")
indices_df = st.session_state.get("indices")
yield_df = st.session_state.get("yield_df")
irrigation_df = st.session_state.get("irrigation_df")

missing = [
    name for name, val in
    [("field boundaries", fields_gdf), ("vegetation indices", indices_df), ("yield records", yield_df), ("irrigation/weather", irrigation_df)]
    if val is None
]
if missing:
    st.info(f"Missing data for this page: {', '.join(missing)}. Load these on the home page first.")
    st.stop()

if st.session_state.get("is_synthetic"):
    st.warning("⚠️ DEMO DATA loaded - water-productivity figures below are synthetic.", icon="⚠️")

features = st.session_state.get("feature_table")
if features is None:
    features = ml_models.build_feature_table(indices_df, irrigation_df, yield_df)
    st.session_state["feature_table"] = features

wp_raw = water_productivity.compute_water_productivity(features)
wp_df = water_productivity.classify_wp_performance(wp_raw)
st.session_state["wp_results"] = wp_df

st.caption(
    "Physical water productivity here = cane yield (t/ha) per 100mm of total water received "
    "(irrigation + rainfall). Treat as a direct measurement only where irrigation is reliably "
    "metered; otherwise read it as a remote-sensing-informed proxy rather than a precise WP figure."
)

if wp_df["wp_tier"].eq("Insufficient data").all():
    st.info("Need at least 3 fields with complete yield + water data to rank water productivity.")
    st.dataframe(wp_raw, use_container_width=True)
    st.stop()

m1, m2, m3 = st.columns(3)
m1.metric("Fields assessed", len(wp_df))
m2.metric("Median WP (t/ha per 100mm)", f"{wp_df['water_productivity_t_ha_per_100mm'].median():.2f}")
m3.metric("Low-tier fields", int((wp_df["wp_tier"] == "Low").sum()))

st.plotly_chart(visualization.wp_bar_chart(wp_df), use_container_width=True)

st.divider()
left, right = st.columns([2, 1])

tier_map = {"Low": 1, "Medium": 2, "High": 3}
merged = fields_gdf.merge(wp_df[[FIELD_ID_COL, "wp_tier", "water_productivity_t_ha_per_100mm"]], on=FIELD_ID_COL, how="left")
merged["tier_numeric"] = merged["wp_tier"].map(tier_map)

with left:
    m = visualization.field_choropleth(merged, "tier_numeric", "WP tier (1=Low, 2=Medium, 3=High)", fill_color="RdYlGn")
    st_folium(m, height=420, use_container_width=True)
with right:
    st.dataframe(
        wp_df[[FIELD_ID_COL, "cane_yield_t_ha", "total_water_mm", "water_productivity_t_ha_per_100mm", "wp_tier"]]
        .sort_values("water_productivity_t_ha_per_100mm"),
        use_container_width=True,
        height=420,
    )

st.caption(
    "Low-tier fields are strong candidates for field investigation: check for under-irrigation, "
    "drainage/waterlogging, soil constraints or crop-stage effects before adjusting scheduling."
)
