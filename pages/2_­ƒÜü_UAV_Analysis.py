"""UAV geomatics module: high-resolution indices and their complementary value vs. satellite."""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from modules import remote_sensing, visualization
from config import FIELD_ID_COL

st.set_page_config(page_title="UAV Analysis", page_icon="🚁", layout="wide")
st.title("🚁 UAV Geomatics Analysis")
st.caption("Objective 2: evaluate the contribution of high-resolution UAV imagery for within-field variability.")

fields_gdf = st.session_state.get("fields")
if fields_gdf is None:
    st.info("Load field boundaries on the home page first.")
    st.stop()

st.markdown(
    "Upload a **UAV orthomosaic GeoTIFF** (already stitched/georeferenced, e.g. from Pix4D, "
    "DroneDeploy or WebODM) covering one or more fields."
)
band_order_str = st.text_input("Band order in the orthomosaic (comma-separated)", value="blue,red,rededge,nir")
flight_date = st.date_input("Flight date", value=dt.date.today())
uav_tif = st.file_uploader("Upload UAV orthomosaic", type=["tif", "tiff"])

if uav_tif is not None and st.button("Compute UAV indices", type="primary"):
    band_order = [b.strip() for b in band_order_str.split(",")]
    result = remote_sensing.read_geotiff_bands(uav_tif.getvalue(), band_order)
    arrays = result["arrays"]
    missing = [b for b in ["blue", "red", "rededge", "nir"] if b not in arrays]
    if missing:
        st.error(f"Missing expected band(s): {missing}. Check the band-order field above.")
    else:
        computed = remote_sensing.compute_all_indices(arrays["blue"], arrays["red"], arrays["rededge"], arrays["nir"])
        uav_rows = []
        for idx_name, arr in computed.items():
            zdf = remote_sensing.zonal_mean_per_field(arr, result["transform"], result["crs"], fields_gdf)
            for _, r in zdf.iterrows():
                uav_rows.append({FIELD_ID_COL: r[FIELD_ID_COL], "index": idx_name, "uav_value": r["mean"], "uav_std": r["std"]})
        st.session_state["uav_indices"] = pd.DataFrame(uav_rows)
        st.session_state["uav_flight_date"] = flight_date
        st.success("UAV indices computed.")

uav_df = st.session_state.get("uav_indices")
if uav_df is not None:
    st.divider()
    st.subheader("UAV vs. satellite comparison")
    chosen_index = st.selectbox("Index", sorted(uav_df["index"].unique()))
    uav_snapshot = uav_df[uav_df["index"] == chosen_index]

    sat_df = st.session_state.get("indices")
    merged = fields_gdf.merge(uav_snapshot[[FIELD_ID_COL, "uav_value", "uav_std"]], on=FIELD_ID_COL, how="left")

    left, right = st.columns([2, 1])
    with left:
        m = visualization.field_choropleth(merged, "uav_value", f"UAV {chosen_index}", fill_color="Greens")
        st_folium(m, height=430, use_container_width=True)

    with right:
        if sat_df is not None and not sat_df.empty:
            sat_latest = (
                sat_df[sat_df["index"] == chosen_index]
                .sort_values("date")
                .groupby(FIELD_ID_COL)
                .last()
                .reset_index()[[FIELD_ID_COL, "value"]]
                .rename(columns={"value": "satellite_value"})
            )
            compare = uav_snapshot[[FIELD_ID_COL, "uav_value"]].merge(sat_latest, on=FIELD_ID_COL, how="left")
            compare["difference"] = compare["uav_value"] - compare["satellite_value"]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=compare[FIELD_ID_COL], y=compare["satellite_value"], name="Satellite (Sentinel-2)", marker_color="#7CB342"))
            fig.add_trace(go.Bar(x=compare[FIELD_ID_COL], y=compare["uav_value"], name="UAV", marker_color="#1B5E20"))
            fig.update_layout(barmode="group", title=f"{chosen_index}: UAV vs. most recent satellite value", template="plotly_white", height=430)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Within-field UAV standard deviation shown below indicates finer-grained variability "
                "than a single satellite pixel/zonal mean can resolve."
            )
            st.dataframe(compare.merge(uav_snapshot[[FIELD_ID_COL, "uav_std"]], on=FIELD_ID_COL), use_container_width=True)
        else:
            st.info("No satellite index data loaded yet to compare against - see the Satellite Monitoring page.")
else:
    st.info("Upload a UAV orthomosaic above to compute high-resolution indices.")
