"""Tests for scr.visuals.visualize module."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scr.visuals.visualize import (
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residual_analysis,
    plot_all_models_overlay,
    plot_monthly_error,
)


class TestPlotModelComparison:
    """Tests for plot_model_comparison()."""

    def test_returns_figure(self, trained_models):
        from scr.Model.predict_models import build_results_table

        _, results = trained_models
        table = build_results_table(results)
        fig = plot_model_comparison(table, return_fig=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_has_four_subplots(self, trained_models):
        from scr.Model.predict_models import build_results_table

        _, results = trained_models
        table = build_results_table(results)
        fig = plot_model_comparison(table, return_fig=True)
        assert len(fig.axes) == 4
        plt.close(fig)


class TestPlotActualVsPredicted:
    """Tests for plot_actual_vs_predicted()."""

    def test_returns_figure(self):
        dates = pd.date_range("2025-01-01", periods=50).values
        y_test = np.random.rand(50) * 100 + 200
        y_pred = y_test + np.random.randn(50) * 5
        fig = plot_actual_vs_predicted(
            dates, y_test, y_pred, "TestModel", return_fig=True
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotResidualAnalysis:
    """Tests for plot_residual_analysis()."""

    def test_returns_figure_with_three_axes(self):
        dates = pd.date_range("2025-01-01", periods=50).values
        y_test = np.random.rand(50) * 100 + 200
        y_pred = y_test + np.random.randn(50) * 5
        fig = plot_residual_analysis(
            dates, y_test, y_pred, "TestModel", return_fig=True
        )
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 3
        plt.close(fig)


class TestPlotAllModelsOverlay:
    """Tests for plot_all_models_overlay()."""

    def test_returns_figure(self, trained_models, split_data_fixture):
        _, results = trained_models
        _, _, _, y_test, _, test_df = split_data_fixture
        dates = test_df["Date"].values
        fig = plot_all_models_overlay(dates, y_test, results, return_fig=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMonthlyError:
    """Tests for plot_monthly_error()."""

    def test_returns_figure(self):
        idx = pd.period_range("2025-01", periods=6, freq="M")
        monthly_err = pd.DataFrame(
            {"MAE": [5, 6, 7, 8, 9, 10], "MAPE": [1, 2, 3, 4, 5, 6], "Count": [20] * 6},
            index=idx,
        )
        monthly_err.index.name = "YearMonth"
        fig = plot_monthly_error(monthly_err, "TestModel", return_fig=True)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
