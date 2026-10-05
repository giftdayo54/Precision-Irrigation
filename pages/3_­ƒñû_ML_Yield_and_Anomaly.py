"""Machine learning module: yield/productivity prediction + anomaly detection."""

import streamlit as st
from streamlit_folium import st_folium

from modules import ml_models, visualization
from config import FIELD_ID_COL

st.set_page_config(page_title="ML Yield & Anomaly", page_icon="🤖", layout="wide")
st.title("🤖 ML Yield Prediction & Anomaly Detection")

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
    st.info(f"Missing data for this page: {', '.join(missing)}. Load these on the home page / Satellite Monitoring page first.")
    st.stop()

if st.session_state.get("is_synthetic"):
    st.warning("⚠️ DEMO DATA loaded - models below are fit on synthetic fields, not real observations.", icon="⚠️")

features = ml_models.build_feature_table(indices_df, irrigation_df, yield_df)
st.session_state["feature_table"] = features

with st.expander("Feature table used for modelling", expanded=False):
    st.dataframe(features, use_container_width=True)

st.divider()
st.header("Yield prediction")

c1, c2 = st.columns([1, 1])
model_choice = c1.selectbox("Model", ["random_forest", "xgboost"], format_func=lambda s: s.replace("_", " ").title())
train_clicked = c2.button("Train yield model", type="primary")

if train_clicked:
    try:
        model, metrics, feature_cols, y_pred_cv, field_ids_used = ml_models.train_yield_model(features, model_choice)
        st.session_state["yield_model"] = model
        st.session_state["yield_model_metrics"] = metrics
        st.session_state["yield_model_feature_cols"] = feature_cols
        st.session_state["yield_model_cv_preds"] = (y_pred_cv, field_ids_used)
    except ValueError as e:
        st.error(str(e))

if st.session_state.get("yield_model") is not None:
    metrics = st.session_state["yield_model_metrics"]
    m1, m2, m3 = st.columns(3)
    m1.metric("R² (cross-validated)", f"{metrics['r2']:.2f}")
    m2.metric("RMSE (t/ha)", f"{metrics['rmse']:.1f}")
    m3.metric("Fields used", metrics["n_fields"])

    y_pred_cv, field_ids_used = st.session_state["yield_model_cv_preds"]
    actual = features.set_index(FIELD_ID_COL).loc[field_ids_used, "cane_yield_t_ha"].values

    left, right = st.columns(2)
    with left:
        st.plotly_chart(visualization.yield_prediction_scatter(actual, y_pred_cv, field_ids_used), use_container_width=True)
    with right:
        fig = visualization.feature_importance_bar(st.session_state["yield_model"], st.session_state["yield_model_feature_cols"])
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    import io
    import joblib
    buf = io.BytesIO()
    joblib.dump(st.session_state["yield_model"], buf)
    st.download_button("Download trained model (.joblib)", buf.getvalue(), file_name=f"yield_model_{model_choice}.joblib")

st.divider()
st.header("Anomaly detection")

tab1, tab2 = st.tabs(["Cross-sectional (Isolation Forest)", "Temporal (baseline t-statistic)"])

with tab1:
    st.caption("Flags fields whose current-season feature profile looks structurally different from the rest of the estate.")
    contamination = st.slider("Expected anomaly fraction", 0.05, 0.35, 0.12, 0.01)
    if st.button("Run Isolation Forest"):
        try:
            result = ml_models.detect_anomalies_isolation_forest(features, contamination=contamination)
            st.session_state["anomaly_iforest"] = result
        except ValueError as e:
            st.error(str(e))

    if st.session_state.get("anomaly_iforest") is not None:
        result = st.session_state["anomaly_iforest"]
        n_flagged = int(result["is_anomaly"].sum())
        st.metric("Fields flagged", n_flagged)
        merged = fields_gdf.merge(result[[FIELD_ID_COL, "is_anomaly", "anomaly_score"]], on=FIELD_ID_COL, how="left")
        merged["is_anomaly"] = merged["is_anomaly"].fillna(False)
        left, right = st.columns([2, 1])
        with left:
            st_folium(visualization.anomaly_hotspot_map(merged), height=420, use_container_width=True)
        with right:
            st.dataframe(result.sort_values("anomaly_score")[[FIELD_ID_COL, "is_anomaly", "anomaly_score"]], use_container_width=True, height=420)

with tab2:
    st.caption(
        "Fits each field's own baseline trend and flags the current week only if it deviates more than "
        "a Student's-t critical value (degrees-of-freedom corrected) would predict by chance - more "
        "robust with short time series than a fixed z-score cutoff."
    )
    idx_options = sorted(indices_df["index"].unique())
    c1, c2, c3 = st.columns(3)
    idx_choice = c1.selectbox("Index", idx_options, index=idx_options.index("NDVI") if "NDVI" in idx_options else 0)
    baseline_weeks = c2.slider("Baseline weeks", 3, 10, 5)
    confidence = c3.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)

    if st.button("Run temporal anomaly detection"):
        result = ml_models.detect_temporal_anomalies(indices_df, idx_choice, baseline_weeks, confidence)
        st.session_state["anomaly_temporal"] = result

    if st.session_state.get("anomaly_temporal") is not None:
        result = st.session_state["anomaly_temporal"]
        if result.empty:
            st.info("Not enough history yet for any field (need baseline_weeks + 1 observations).")
        else:
            n_flagged = int(result["is_anomaly"].sum())
            st.metric("Fields flagged", n_flagged)
            st.dataframe(result.sort_values("t_statistic", key=abs, ascending=False), use_container_width=True)

            flagged_fields = result[result["is_anomaly"]][FIELD_ID_COL].tolist()
            if flagged_fields:
                pick = st.selectbox("Inspect time series for a flagged field", flagged_fields)
                st.plotly_chart(visualization.index_timeseries_chart(indices_df, pick, idx_choice), use_container_width=True)
