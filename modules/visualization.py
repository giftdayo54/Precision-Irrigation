"""
Map and chart builders shared across pages - folium for the spatial views,
plotly for time series, kept deliberately thin so page files stay readable.
"""

from __future__ import annotations

import folium
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

from config import FIELD_ID_COL


def field_choropleth(
    fields_gdf: gpd.GeoDataFrame, value_col: str, legend_name: str, fill_color: str = "RdYlGn"
) -> folium.Map:
    gdf_wgs84 = fields_gdf.to_crs("EPSG:4326")
    centroid = gdf_wgs84.geometry.unary_union.centroid
    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=14, tiles="OpenStreetMap")

    # folium's Choropleth binning chokes on NaN - fill with the column mean (or 0
    # if every value is missing) purely for the colour scale; raw NaNs are still
    # shown correctly in the per-field tooltips/markers below.
    choropleth_data = gdf_wgs84.copy()
    fill_value = choropleth_data[value_col].mean()
    if pd.isna(fill_value):
        fill_value = 0
    choropleth_data[value_col] = choropleth_data[value_col].fillna(fill_value)

    folium.Choropleth(
        geo_data=gdf_wgs84.__geo_interface__,
        data=choropleth_data,
        columns=[FIELD_ID_COL, value_col],
        key_on=f"feature.properties.{FIELD_ID_COL}",
        fill_color=fill_color,
        fill_opacity=0.75,
        line_opacity=0.6,
        legend_name=legend_name,
        nan_fill_color="lightgrey",
    ).add_to(m)

    for _, row in gdf_wgs84.iterrows():
        c = row.geometry.centroid
        val = row.get(value_col)
        label = f"{row[FIELD_ID_COL]}: {val:.3f}" if pd.notna(val) else f"{row[FIELD_ID_COL]}: no data"
        folium.Marker(
            [c.y, c.x],
            icon=folium.DivIcon(html=f'<div style="font-size:10pt;color:#222">{row[FIELD_ID_COL]}</div>'),
            tooltip=label,
        ).add_to(m)

    return m


def anomaly_hotspot_map(fields_gdf: gpd.GeoDataFrame, anomaly_col: str = "is_anomaly") -> folium.Map:
    gdf = fields_gdf.copy()
    gdf["_flag"] = gdf[anomaly_col].map({True: 1, False: 0})
    return field_choropleth(gdf, "_flag", "Anomaly flag (1 = flagged)", fill_color="YlOrRd")


def index_timeseries_chart(ts_df: pd.DataFrame, field_id: str, index_name: str = "NDVI") -> go.Figure:
    sub = ts_df[(ts_df[FIELD_ID_COL] == field_id) & (ts_df["index"] == index_name)].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub["date"], y=sub["value"], mode="lines+markers", name=f"{index_name} ({field_id})"))
    fig.update_layout(
        title=f"{index_name} time series - {field_id}",
        xaxis_title="Date",
        yaxis_title=index_name,
        template="plotly_white",
        height=350,
    )
    return fig


def yield_prediction_scatter(actual, predicted, field_ids) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual, y=predicted, mode="markers", text=field_ids,
            marker=dict(size=10, color="#2E7D32"), name="Fields",
        )
    )
    lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="grey"), name="1:1 line"))
    fig.update_layout(
        title="Predicted vs. actual cane yield (cross-validated)",
        xaxis_title="Actual yield (t/ha)",
        yaxis_title="Predicted yield (t/ha)",
        template="plotly_white",
        height=420,
    )
    return fig


def feature_importance_bar(model, feature_cols) -> go.Figure | None:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return None
    order = importances.argsort()[::-1]
    fig = go.Figure(go.Bar(x=[feature_cols[i] for i in order], y=importances[order], marker_color="#558B2F"))
    fig.update_layout(title="Feature importance", template="plotly_white", height=350)
    return fig


def wp_bar_chart(wp_df: pd.DataFrame) -> go.Figure:
    sub = wp_df.sort_values("water_productivity_t_ha_per_100mm", ascending=False)
    colors = sub["wp_tier"].map({"High": "#2E7D32", "Medium": "#F9A825", "Low": "#C62828"}).fillna("#9E9E9E")
    fig = go.Figure(
        go.Bar(x=sub[FIELD_ID_COL], y=sub["water_productivity_t_ha_per_100mm"], marker_color=colors)
    )
    fig.update_layout(
        title="Water productivity by field (t/ha per 100mm water)",
        xaxis_title="Field",
        yaxis_title="t/ha per 100mm",
        template="plotly_white",
        height=380,
    )
    return fig
