# AI-GENERATED (Claude via Antigravity)
"""Streamlit dashboard for Sri Lanka weekly district dengue forecasting (loads saved pickle, never retrains)."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
TOP5_FEATURES = [
    "cases_roll_mean_4",
    "cases_lag_0",
    "cases_roll_max_4",
    "cases_roll_mean_8",
    "cases_lag_1",
]

st.set_page_config(page_title="Sri Lanka Dengue Forecast", layout="wide")


@st.cache_resource
def load_model():
    with open(ROOT / "models" / "best_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_assets():
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    results = pd.read_csv(ROOT / "reports" / "results_final.csv")
    preds = pd.read_csv(ROOT / "models" / "test_predictions.csv")
    test_feat = pd.read_csv(ROOT / "app" / "data" / "test_features.csv")
    interp = (ROOT / "app" / "interpretation.md").read_text(encoding="utf-8")
    return meta, results, preds, test_feat, interp


def predict_cases(pipe, df_rows, feature_cols):
    return np.clip(np.expm1(pipe.predict(df_rows[feature_cols])), 0, None)


pipe = load_model()
meta, results, preds, test_feat, interp_md = load_assets()

st.title("Weekly Dengue Case Forecasting — Sri Lanka")
st.warning("Educational project. Not for medical or public-health decisions.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Overview",
    "2. Model comparison",
    "3. Why the winner won",
    "4. Try the model",
])

with tab1:
    st.subheader("Project Overview")
    st.write(
        "This university machine learning mini-project predicts **next week's dengue cases per district in Sri Lanka** "
        "using ensemble learning (bagging, boosting, and stacking/voting) on weekly district-level records from 2006 to 2025. "
        "The dataset is Kaggle's *Dengue_Research_2006-2026* (author & license: see README.md). "
        "Because weekly case counts are heavily right-skewed with 13% zero weeks and rare large outbreaks, "
        "models are trained on `log1p(target_next_cases)` and converted back to real case counts with `expm1` before scoring."
    )
    m = meta["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winning Model", meta["model_name"])
    c2.metric("Test RMSE (2022–2025)", f"{m['RMSE']:.2f} cases")
    c3.metric("Test MAE", f"{m['MAE']:.2f} cases")
    c4.metric("Test R²", f"{m['R2']:.3f}")

with tab2:
    st.subheader("Model Comparison (Test Period: 2022–2025)")
    st.dataframe(results.round(3), width="stretch", hide_index=True)

    chart_df = results.set_index("model")[["test_RMSE", "test_MAE"]].sort_values("test_RMSE")
    st.bar_chart(chart_df)

    st.info(
        "**Bagging vs. Boosting vs. Stacking:** "
        "**Bagging** (like Random Forest) trains many independent trees on random samples and averages them to cut variance/overfitting. "
        "**Boosting** (like XGBoost and CatBoost) trains trees one after another so each new tree fixes previous errors, cutting bias. "
        "**Stacking & Voting** combine different model families together—simple averaging helped default models, while tuned Random Forest, CatBoost, and XGBoost finished in a near tie."
    )

with tab3:
    st.subheader("Why the Winner Won")
    st.markdown(interp_md)
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(ROOT / "reports" / "figures" / "08_importance_by_group.png"), caption="Importance by Feature Group")
    with c2:
        st.image(str(ROOT / "reports" / "figures" / "08_shap_summary.png"), caption="SHAP Summary (2,000 test rows)")

with tab4:
    st.subheader("Part A: Inspect Test-Period Predictions")
    col_d, col_w = st.columns(2)
    district = col_d.selectbox("District", meta["districts"], index=meta["districts"].index("Colombo"))
    dist_preds = preds[preds["district"] == district].sort_values("date")
    week = col_w.selectbox("Forecast week (start.date)", dist_preds["date"].tolist(), index=len(dist_preds) - 1)

    row = dist_preds[dist_preds["date"] == week].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted next-week cases", f"{row['predicted']:.1f}")
    m2.metric("Actual next-week cases", f"{row['actual']:.0f}")
    m3.metric("Error (Actual - Predicted)", f"{row['actual'] - row['predicted']:+.1f}")

    st.line_chart(dist_preds.set_index("date")[["actual", "predicted"]])

    st.divider()
    st.subheader("Part B: What-If Sliders (Top 5 Most Important Features)")
    st.caption(f"Other features are fixed to {district}'s test-period median.")

    dist_feat = test_feat[test_feat["district"] == district]
    base_row = dist_feat[meta["features"]].median(numeric_only=True).to_dict()
    base_row["district"] = district

    slider_cols = st.columns(len(TOP5_FEATURES))
    for col_ui, feat in zip(slider_cols, TOP5_FEATURES):
        f_min = float(dist_feat[feat].min())
        f_max = float(max(dist_feat[feat].max(), f_min + 1.0))
        f_med = float(dist_feat[feat].median())
        base_row[feat] = col_ui.slider(feat, min_value=f_min, max_value=f_max, value=f_med, step=1.0)

    whatif_df = pd.DataFrame([base_row])[meta["features"]]
    whatif_pred = float(predict_cases(pipe, whatif_df, meta["features"])[0])
    st.metric(f"What-If Predicted Next-Week Cases ({district})", f"{whatif_pred:.1f}")
