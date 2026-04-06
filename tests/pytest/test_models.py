"""Tests for ARIMA/GARCH/LSTM model training and evaluation modules."""

import numpy as np
import pandas as pd

from scr.Model.train_models import (
    get_models,
    save_models,
    load_model,
    MODELS_DIR,
)
from scr.Model.predict_models import (
    evaluate_model,
    build_results_table,
    get_best_model,
    directional_accuracy,
    regime_error_analysis,
    monthly_error_analysis,
)
from scr.Model.hyper_tuning import get_param_grids, tune_model, _grid_size


class TestGetModels:
    def test_returns_required_time_series_models(self):
        models = get_models()
        assert set(models.keys()) == {"ARIMA", "GARCH", "LSTM"}

    def test_all_models_have_fit_predict(self):
        models = get_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit()"
            assert hasattr(model, "predict"), f"{name} missing predict()"


class TestTrainAllModels:
    def test_all_models_have_results(self, trained_models):
        models, results = trained_models
        assert set(results.keys()) == set(models.keys())

    def test_results_have_required_metrics(self, trained_models):
        _, results = trained_models
        required = {"MAE", "RMSE", "R²", "MAPE (%)", "Predictions"}
        for name, metrics in results.items():
            assert required.issubset(set(metrics.keys())), f"{name} missing metrics"

    def test_predictions_have_correct_length(self, trained_models, split_data_fixture):
        _, results = trained_models
        _, _, _, y_test, _, _ = split_data_fixture
        for name, metrics in results.items():
            assert len(metrics["Predictions"]) == len(y_test), f"{name} prediction mismatch"

    def test_metrics_are_finite(self, trained_models):
        _, results = trained_models
        for name, metrics in results.items():
            assert np.isfinite(metrics["MAE"]), f"{name} MAE invalid"
            assert np.isfinite(metrics["RMSE"]), f"{name} RMSE invalid"
            assert np.isfinite(metrics["R²"]), f"{name} R² invalid"


class TestSaveLoadModels:
    def test_save_creates_pkl_files(self, trained_models, tmp_path):
        models, _ = trained_models
        save_models(models, output_dir=tmp_path)
        pkl_files = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == len(models)

    def test_load_model_returns_fitted(self, trained_models, tmp_path):
        models, _ = trained_models
        save_models(models, output_dir=tmp_path)
        loaded = load_model("ARIMA", model_dir=tmp_path)
        assert hasattr(loaded, "predict")

    def test_model_files_exist_in_models_dir(self):
        pkl_files = list(MODELS_DIR.glob("*.pkl"))
        assert len(pkl_files) >= 3


class TestPredictModels:
    def test_evaluate_model_keys(self):
        y_true = np.array([100, 200, 300, 400], dtype=float)
        y_pred = np.array([102, 198, 301, 395], dtype=float)
        result = evaluate_model(y_true, y_pred)
        assert set(result.keys()) == {"MAE", "RMSE", "R²", "MAPE (%)"}

    def test_build_results_table_returns_dataframe(self, trained_models):
        _, results = trained_models
        table = build_results_table(results)
        assert isinstance(table, pd.DataFrame)
        assert "Predictions" not in table.columns

    def test_get_best_model(self, trained_models):
        models, results = trained_models
        name, model, preds = get_best_model(results, models)
        assert isinstance(name, str)
        assert hasattr(model, "predict")
        assert isinstance(preds, np.ndarray)

    def test_directional_accuracy_range(self, trained_models, split_data_fixture):
        models, results = trained_models
        _, _, _, y_test, _, _ = split_data_fixture
        _, _, preds = get_best_model(results, models)
        acc = directional_accuracy(y_test, preds)
        assert 0 <= acc <= 100

    def test_regime_and_monthly_analysis(self, trained_models, split_data_fixture):
        models, results = trained_models
        _, _, _, _, _, test_df = split_data_fixture
        name, _, preds = get_best_model(results, models)

        regime = regime_error_analysis(test_df, preds, name)
        monthly = monthly_error_analysis(test_df, preds)

        assert isinstance(regime, pd.DataFrame)
        assert "MAE" in regime.columns
        assert isinstance(monthly, pd.DataFrame)
        assert "MAE" in monthly.columns


class TestHyperTuning:
    def test_get_param_grids(self):
        grids = get_param_grids()
        assert set(grids.keys()) == {"ARIMA", "GARCH", "LSTM"}

    def test_grid_size(self):
        assert _grid_size({"a": [1, 2], "b": [3, 4, 5]}) == 6

    def test_tune_single_model(self, split_data_fixture):
        X_train, _, y_train, _, _, _ = split_data_fixture
        estimator, param_grid = get_param_grids()["ARIMA"]
        result = tune_model("ARIMA", estimator, param_grid, X_train, y_train)
        assert "best_estimator" in result
        assert "best_score" in result
        assert "best_params" in result
