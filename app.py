"""
Precision Irrigation & Sugarcane Water Productivity - home page.

Integrates satellite remote sensing, UAV geomatics and machine learning to
support spatial irrigation-performance and water-productivity assessment,
per the MSc research proposal this app operationalises.
"""

import streamlit as st
from streamlit_folium import st_folium

from modules import data_ingestion, sample_data
from modules.data_ingestion import IngestionError

st.set_page_config(page_title="Precision Irrigation | Sugarcane WP", page_icon="🌾", layout="wide")

DEFAULTS = {
    "fields": None,
    "indices": None,
    "yield_df": None,
    "irrigation_df": None,
    "is_synthetic": False,
    "feature_table": None,
    "yield_model": None,
    "yield_model_metrics": None,
    "anomaly_iforest": None,
    "anomaly_temporal": None,
    "wp_results": None,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

st.title("🌾 Precision Irrigation & Sugarcane Water Productivity")
st.caption(
    "Integrating satellite remote sensing, UAV geomatics and machine learning "
    "for precision irrigation and sugarcane water productivity assessment"
)

if st.session_state["is_synthetic"]:
    st.warning(
        "⚠️ **DEMO DATA** is currently loaded. Every number and map on every page is "
        "synthetic and for exploring the workflow only - it is not a real field "
        "observation. Load your own data below to replace it.",
        icon="⚠️",
    )

with st.expander("What this app does (research framework overview)", expanded=False):
    st.markdown(
        """
**Aim:** develop and evaluate an integrated Geomatics and machine-learning framework for
monitoring sugarcane crop condition, identifying spatial irrigation anomalies and assessing
water productivity using satellite remote sensing, UAV imagery and agricultural field data.

**Pages in this app:**
1. 📡 **Satellite Monitoring** - Sentinel-2 NDVI/EVI/SAVI/NDRE, cloud-masked, per field
2. 🚁 **UAV Analysis** - high-resolution UAV indices vs. satellite, complementary value
3. 🤖 **ML Yield & Anomaly** - Random Forest / XGBoost yield prediction, Isolation Forest +
   temporal t-statistic anomaly detection
4. 💧 **Water Productivity** - yield relative to water applied, per-field performance tiers
5. 🗺️ **Decision Dashboard** - combined map and prioritised list of fields for field investigation
        """
    )

st.divider()

st.subheader("1. Load your data")
mode = st.radio(
    "Data source", ["Try with demo data", "Upload my own data"], horizontal=True,
    help="Demo data lets you explore every page immediately with synthetic fields, "
         "indices, yield and irrigation records.",
)

if mode == "Try with demo data":
    c1, c2 = st.columns([1, 1])
    n_fields = c1.slider("Number of demo fields", 4, 24, 12)
    weeks = c2.slider("Weeks of history", 6, 20, 12)
    if st.button("Generate demo dataset", type="primary"):
        demo = sample_data.generate_full_demo_dataset(n_fields=n_fields, weeks=weeks)
        st.session_state["fields"] = demo["fields"]
        st.session_state["indices"] = demo["indices"]
        st.session_state["yield_df"] = demo["yield"]
        st.session_state["irrigation_df"] = demo["irrigation"]
        st.session_state["is_synthetic"] = True
        st.rerun()

else:
    st.session_state["is_synthetic"] = False
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Field boundaries**")
        st.caption("GeoJSON or zipped Shapefile")
        boundary_file = st.file_uploader("Upload field boundaries", type=["geojson", "json", "zip"], key="boundary_upl")
        if boundary_file is not None:
            try:
                gdf, warnings = data_ingestion.load_field_boundaries(boundary_file)
                st.session_state["fields"] = gdf
                for w in warnings:
                    st.info(w)
                st.success(f"Loaded {len(gdf)} fields.")
            except IngestionError as e:
                st.error(str(e))

    with col2:
        st.markdown("**Historical yield records**")
        st.caption("CSV with field ID + cane_yield_t_ha")
        yield_file = st.file_uploader("Upload yield CSV", type=["csv"], key="yield_upl")
        if yield_file is not None:
            try:
                ydf, warnings = data_ingestion.load_yield_records(yield_file)
                st.session_state["yield_df"] = ydf
                for w in warnings:
                    st.info(w)
                st.success(f"Loaded {len(ydf)} yield records.")
            except IngestionError as e:
                st.error(str(e))

    with col3:
        st.markdown("**Irrigation / weather log**")
        st.caption("CSV with field ID, date, irrigation_mm, [rainfall_mm, eto_mm]")
        irr_file = st.file_uploader("Upload irrigation/weather CSV", type=["csv"], key="irr_upl")
        if irr_file is not None:
            try:
                idf, warnings = data_ingestion.load_irrigation_weather_log(irr_file)
                st.session_state["irrigation_df"] = idf
                for w in warnings:
                    st.info(w)
                st.success(f"Loaded {len(idf)} irrigation/weather rows.")
            except IngestionError as e:
                st.error(str(e))

    st.caption(
        "Vegetation-index time series are loaded on the **📡 Satellite Monitoring** page "
        "(live CDSE fetch, uploaded GeoTIFFs, or demo)."
    )

st.divider()

fields_gdf = st.session_state["fields"]
if fields_gdf is not None:
    st.subheader("2. Field boundaries preview")
    left, right = st.columns([2, 1])
    with left:
        gdf_wgs84 = fields_gdf.to_crs("EPSG:4326")
        centroid = gdf_wgs84.geometry.unary_union.centroid
        import folium
        m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14)
        folium.GeoJson(gdf_wgs84.__geo_interface__, name="Fields").add_to(m)
        st_folium(m, height=420, use_container_width=True)
    with right:
        st.dataframe(fields_gdf.drop(columns="geometry"), use_container_width=True, height=420)

    tabs = st.tabs(["Yield records", "Irrigation / weather"])
    with tabs[0]:
        if st.session_state["yield_df"] is not None:
            st.dataframe(st.session_state["yield_df"], use_container_width=True)
        else:
            st.info("No yield records loaded yet.")
    with tabs[1]:
        if st.session_state["irrigation_df"] is not None:
            st.dataframe(st.session_state["irrigation_df"], use_container_width=True)
        else:
            st.info("No irrigation/weather records loaded yet.")
else:
    st.info("⬆️ Load field boundaries (demo or your own) to get started, then use the pages in the sidebar.")
