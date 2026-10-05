"""Satellite remote sensing module: Sentinel-2 vegetation indices per field."""

import datetime as dt

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import remote_sensing, sentinel_hub_client, visualization
from config import VEGETATION_INDICES, FIELD_ID_COL

st.set_page_config(page_title="Satellite Monitoring", page_icon="📡", layout="wide")
st.title("📡 Satellite Remote Sensing Monitoring")

fields_gdf = st.session_state.get("fields")
if fields_gdf is None:
    st.info("Load field boundaries on the home page first.")
    st.stop()

if st.session_state.get("is_synthetic"):
    st.warning("⚠️ DEMO DATA loaded - indices below are synthetic, not real Sentinel-2 observations.", icon="⚠️")

source = st.radio(
    "Vegetation index source",
    ["Use loaded data (demo or previously computed)", "Upload Sentinel-2 GeoTIFF bands", "Fetch live from CDSE"],
    horizontal=False,
)

indices_df = st.session_state.get("indices")

if source == "Upload Sentinel-2 GeoTIFF bands":
    st.markdown(
        "Upload a **multi-band GeoTIFF** clipped to your study area (e.g. exported from SNAP, QGIS, "
        "or a CDSE batch download), containing blue, red, red-edge and NIR bands."
    )
    band_order_str = st.text_input("Band order in the file (comma-separated)", value="blue,red,rededge,nir")
    scene_date = st.date_input("Acquisition date for this scene", value=dt.date.today())
    tif = st.file_uploader("Upload GeoTIFF", type=["tif", "tiff"])
    if tif is not None and st.button("Compute indices for this scene", type="primary"):
        band_order = [b.strip() for b in band_order_str.split(",")]
        result = remote_sensing.read_geotiff_bands(tif.getvalue(), band_order)
        arrays = result["arrays"]
        missing = [b for b in ["blue", "red", "rededge", "nir"] if b not in arrays]
        if missing:
            st.error(f"Missing expected band(s) in upload: {missing}. Check the band-order field above.")
        else:
            computed = remote_sensing.compute_all_indices(arrays["blue"], arrays["red"], arrays["rededge"], arrays["nir"])
            new_rows = []
            for idx_name, arr in computed.items():
                zdf = remote_sensing.zonal_mean_per_field(arr, result["transform"], result["crs"], fields_gdf)
                for _, r in zdf.iterrows():
                    new_rows.append({FIELD_ID_COL: r[FIELD_ID_COL], "date": pd.Timestamp(scene_date), "index": idx_name, "value": r["mean"]})
            new_df = pd.DataFrame(new_rows)
            indices_df = pd.concat([indices_df, new_df], ignore_index=True) if indices_df is not None else new_df
            st.session_state["indices"] = indices_df
            st.session_state["is_synthetic"] = False
            st.success(f"Computed {len(VEGETATION_INDICES)} indices for {scene_date} and merged into the dataset.")

elif source == "Fetch live from CDSE":
    if sentinel_hub_client.is_configured():
        st.info(
            "CDSE credentials found. Note: this scaffold's `fetch_scene()` performs the OAuth2 + "
            "Process API request but the multipart-TIFF response parsing is left as an integration "
            "point (`modules/sentinel_hub_client.py`) - it's far easier to get right against a live "
            "response than to guess blind. Wire that up, then this page's 'Fetch scene' button will "
            "populate the index time series exactly like the upload path above."
        )
    else:
        st.warning(
            "No CDSE credentials found in `.streamlit/secrets.toml` (see `secrets.toml.example`). "
            "Use 'Upload GeoTIFF bands' or demo data in the meantime.",
            icon="🔑",
        )

if indices_df is not None and not indices_df.empty:
    st.divider()
    st.subheader("Vegetation index map & time series")
    c1, c2 = st.columns([1, 1])
    chosen_index = c1.selectbox("Index", sorted(indices_df["index"].unique()))
    dates_available = sorted(indices_df[indices_df["index"] == chosen_index]["date"].unique())
    chosen_date = c2.selectbox("Date", dates_available, index=len(dates_available) - 1)

    snapshot = indices_df[(indices_df["index"] == chosen_index) & (indices_df["date"] == chosen_date)]
    merged = fields_gdf.merge(snapshot[[FIELD_ID_COL, "value"]], on=FIELD_ID_COL, how="left")

    left, right = st.columns([2, 1])
    with left:
        m = visualization.field_choropleth(merged, "value", f"{chosen_index} ({pd.Timestamp(chosen_date).date()})")
        st_folium(m, height=430, use_container_width=True)
    with right:
        field_for_ts = st.selectbox("Field time series", sorted(fields_gdf[FIELD_ID_COL].unique()))
        fig = visualization.index_timeseries_chart(indices_df, field_for_ts, chosen_index)
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Field-level snapshot values")
    st.dataframe(merged[[FIELD_ID_COL, "value"]].rename(columns={"value": chosen_index}), use_container_width=True)
else:
    st.info("No vegetation-index data yet - generate demo data on the home page, or upload/fetch a scene above.")
