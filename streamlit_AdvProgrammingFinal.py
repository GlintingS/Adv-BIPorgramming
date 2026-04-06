import logging

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from scr.data.make_dataset import (
    load_phase2_data,
    prepare_features,
    split_data,
    scale_features,
    FEATURE_COLS,
)
from scr.Model.train_models import (
    get_models,
    train_all_models,
    cross_validate_models,
)
from scr.Model.predict_models import (
    build_results_table,
    get_best_model,
    directional_accuracy,
    regime_error_analysis,
    monthly_error_analysis,
)
from scr.visuals.visualize import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_all_models_overlay,
    plot_monthly_error,
)

logger = logging.getLogger(__name__)

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title="Tesla Stock Price Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "Tesla Stock Price Prediction app built with Streamlit, "
            "ARIMA, GARCH, and LSTM models.  "
            "Algonquin College — Advanced BI Programming Final Project."
        ),
    },
)

st.title("📈 Tesla Stock Price Forecasting")
st.caption(
    "Predicting Tesla (TSLA) closing price using economic indicators "
    "and advanced time-series models."
)


# ── Secrets / API key helper ──────────────────────────────────────────
def _get_fred_key() -> str | None:
    """Read the FRED API key from Streamlit secrets (cloud) or return None
    to let data_download.py fall back to env/cache behavior."""
    try:
        return st.secrets["FRED_API_KEY"]
    except (FileNotFoundError, KeyError):
        return None


# ── Cached data loading ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(fred_key: str | None = None):
    df = load_phase2_data(fred_api_key=fred_key)
    model_df = prepare_features(df)
    X_train, X_test, y_train, y_test, train_df, test_df = split_data(model_df)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return (
        df,
        model_df,
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df,
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


@st.cache_resource(show_spinner=False)
def run_training(X_train, y_train, X_train_scaled, X_test, X_test_scaled, y_test):
    models = get_models()
    results = train_all_models(
        models, X_train, y_train, X_train_scaled, X_test, X_test_scaled, y_test
    )
    return models, results


@st.cache_data(show_spinner=False)
def run_cross_validation(_models, X_train, y_train, X_train_scaled):
    return cross_validate_models(_models, X_train, y_train, X_train_scaled)


# ── Load & Train with progress feedback ───────────────────────────────
try:
    with st.spinner("Downloading market data & engineering features …"):
        (
            df,
            model_df,
            X_train,
            X_test,
            y_train,
            y_test,
            train_df,
            test_df,
            X_train_scaled,
            X_test_scaled,
            scaler,
        ) = load_data(_get_fred_key())

    with st.spinner("Training models …"):
        models, results = run_training(
            X_train,
            y_train,
            X_train_scaled,
            X_test,
            X_test_scaled,
            y_test,
        )

except Exception as exc:
    logger.exception("Streamlit pipeline failed")
    st.error(f"**Data pipeline failed:** {exc}")
    st.info(
        "If running on Streamlit Cloud, make sure the FRED_API_KEY secret is set "
        "in **App settings → Secrets**."
    )
    st.stop()

results_table = build_results_table(results)
best_name, _, best_preds = get_best_model(results, models)
test_dates = test_df["Date"].values
dir_acc = directional_accuracy(y_test, best_preds)

# ── Sidebar ───────────────────────────────────────────────────────────
st.sidebar.header("Dataset Info")
st.sidebar.metric("Total Rows", f"{len(model_df):,}")
st.sidebar.metric("Features", len(FEATURE_COLS))
st.sidebar.metric("Train Rows", f"{len(train_df):,}")
st.sidebar.metric("Test Rows", f"{len(test_df):,}")
st.sidebar.markdown(
    f"**Date Range:** {df['Date'].min().date()} → {df['Date'].max().date()}"
)

if st.sidebar.button("🔄 Refresh Data", help="Clear cache and re-download live data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Select Model")
show_cv = st.sidebar.toggle(
    "Run cross-validation",
    value=False,
    help="Enable only when needed. This can be slow on Streamlit Cloud.",
)
selected_model = st.sidebar.selectbox(
    "Model", list(results.keys()), index=list(results.keys()).index(best_name)
)
sel_preds = results[selected_model]["Predictions"]

# ── Tab Layout ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Model Comparison",
        "Predictions",
        "Residual Analysis",
        "Volatility",
        "Error Analysis",
    ]
)

# ── Tab 1: Model Comparison ──────────────────────────────────────────
with tab1:
    st.subheader("Test-Set Performance (sorted by RMSE)")
    styled = (
        results_table.style.format(
            {
                "MAE": "{:.2f}",
                "RMSE": "{:.2f}",
                "R²": "{:.4f}",
                "MAPE (%)": "{:.2f}",
            }
        )
        .background_gradient(subset=["R²"], cmap="Greens")
        .background_gradient(subset=["RMSE"], cmap="Reds_r")
    )
    st.dataframe(styled, use_container_width=True)

    fig = plot_model_comparison(results_table, return_fig=True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Time-Series Cross-Validation (5-Fold)")
    if show_cv:
        with st.spinner("Running cross-validation ..."):
            cv_df = run_cross_validation(models, X_train, y_train, X_train_scaled)
        st.dataframe(
            cv_df.style.format({"CV R² Mean": "{:.4f}", "CV R² Std": "{:.4f}"}),
            use_container_width=True,
        )
    else:
        st.info("Cross-validation is disabled. Enable it from the sidebar when needed.")

# ── Tab 2: Predictions ───────────────────────────────────────────────
with tab2:
    st.subheader(f"Actual vs Predicted — {selected_model}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"${results[selected_model]['MAE']:.2f}")
    col2.metric("RMSE", f"${results[selected_model]['RMSE']:.2f}")
    col3.metric("R²", f"{results[selected_model]['R²']:.4f}")
    col4.metric("MAPE", f"{results[selected_model]['MAPE (%)']:.2f}%")

    fig = plot_actual_vs_predicted(
        test_dates, y_test, sel_preds, selected_model, return_fig=True
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("All Models Overlay")
    fig = plot_all_models_overlay(test_dates, y_test, results, return_fig=True)
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 3: Residual Analysis ─────────────────────────────────────────
with tab3:
    st.subheader(f"Residual Analysis — {selected_model}")
    fig = plot_residual_analysis(
        test_dates, y_test, sel_preds, selected_model, return_fig=True
    )
    st.pyplot(fig)
    plt.close(fig)

    residuals = np.asarray(y_test) - np.asarray(sel_preds)
    rcol1, rcol2, rcol3, rcol4 = st.columns(4)
    rcol1.metric("Mean Residual", f"${residuals.mean():.2f}")
    rcol2.metric("Std Residual", f"${residuals.std():.2f}")
    rcol3.metric("Max Over-predict", f"${residuals.min():.2f}")
    rcol4.metric("Max Under-predict", f"${residuals.max():.2f}")

# ── Tab 4: Volatility ─────────────────────────────────────────────────
with tab4:
    st.subheader("Volatility Regime Overview")
    regime_stats = regime_error_analysis(test_df, sel_preds, selected_model)
    st.dataframe(regime_stats, use_container_width=True)

    garch_model = models.get("GARCH")
    if garch_model is not None and hasattr(garch_model, "get_volatility_forecast"):
        vol = garch_model.get_volatility_forecast()
        if vol is not None:
            vol_df = pd.DataFrame(
                {
                    "Date": test_df["Date"].values[: len(vol)],
                    "Forecast_Volatility": vol,
                }
            )
            st.line_chart(vol_df.set_index("Date"))

# ── Tab 5: Error Analysis ────────────────────────────────────────────
with tab5:
    st.subheader("Monthly Prediction Accuracy")
    monthly_err = monthly_error_analysis(test_df, sel_preds)
    fig = plot_monthly_error(monthly_err, selected_model, return_fig=True)
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(monthly_err, use_container_width=True)

    st.subheader("Directional Accuracy")
    sel_dir_acc = directional_accuracy(y_test, sel_preds)
    st.metric("Direction Correct", f"{sel_dir_acc:.1f}%")

# ── Summary ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Summary")
st.markdown(
    f"""
- **Best Model:** {best_name} (R² = {results[best_name]['R²']:.4f}, MAPE = {results[best_name]['MAPE (%)']:.2f}%)
- **Directional Accuracy:** {dir_acc:.1f}%
- Model performance can shift over time, so periodic retraining is recommended.
- NASDAQ and S&P 500 are among the strongest predictors.
- Model accuracy degrades during high-volatility regimes.
"""
)

# ── Footer ────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Algonquin College — Advanced BI Programming Final Project · "
    "Data: Yahoo Finance & FRED · Built with Streamlit"
)
