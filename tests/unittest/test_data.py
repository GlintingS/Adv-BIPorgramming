"""Unit tests for data modules using the unittest framework."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scr.data.data_download import load_raw_datasets, merge_raw, EXTERNAL_DIR, RAW_DIR
from scr.data.make_dataset import (
    _engineer_features,
    prepare_features,
    split_data,
    scale_features,
    FEATURE_COLS,
    TARGET_COL,
)
from scr.Model.train_models import get_models, train_all_models
from scr.Model.predict_models import evaluate_model, build_results_table, get_best_model


class TestDataDownload(unittest.TestCase):
    """Tests for the data_download module."""

    @classmethod
    def setUpClass(cls):
        cls.datasets = load_raw_datasets()

    def test_dataset_count(self):
        self.assertEqual(len(self.datasets), 7)

    def test_tsla_has_close_column(self):
        self.assertIn("Close", self.datasets["TSLA"].columns)

    def test_fred_datasets_not_empty(self):
        for key in ["FEDFUNDS", "CPIAUCSL", "UNRATE"]:
            self.assertGreater(len(self.datasets[key]), 0)

    def test_external_files_exist(self):
        expected = [
            "tesla_stock_raw.csv",
            "sp500_raw.csv",
            "nasdaq_raw.csv",
            "vix_raw.csv",
            "fed_funds_rate_raw.csv",
            "cpi_raw.csv",
            "unemployment_rate_raw.csv",
        ]
        for f in expected:
            self.assertTrue((EXTERNAL_DIR / f).exists(), f"Missing: {f}")


class TestMergeRaw(unittest.TestCase):
    """Tests for merge_raw()."""

    @classmethod
    def setUpClass(cls):
        datasets = load_raw_datasets()
        cls.merged = merge_raw(datasets)

    def test_is_dataframe(self):
        self.assertIsInstance(self.merged, pd.DataFrame)

    def test_has_tsla_close(self):
        self.assertIn("TSLA_Close", self.merged.columns)

    def test_row_count(self):
        self.assertGreater(len(self.merged), 1000)

    def test_dates_sorted(self):
        dates = self.merged["Date"].values
        self.assertTrue((dates[1:] >= dates[:-1]).all())


class TestFeatureEngineering(unittest.TestCase):
    """Tests for _engineer_features()."""

    @classmethod
    def setUpClass(cls):
        datasets = load_raw_datasets()
        merged = merge_raw(datasets)
        cls.df = _engineer_features(merged)

    def test_daily_returns_added(self):
        self.assertIn("TSLA_Daily_Return", self.df.columns)

    def test_calendar_features_added(self):
        for col in ["Month", "Quarter", "Year"]:
            self.assertIn(col, self.df.columns)

    def test_highvol_regime_binary(self):
        unique = set(self.df["HighVol_Regime"].dropna().unique())
        self.assertTrue(unique.issubset({0, 1}))


class TestPrepareAndSplit(unittest.TestCase):
    """Tests for prepare_features() and split_data()."""

    @classmethod
    def setUpClass(cls):
        datasets = load_raw_datasets()
        merged = merge_raw(datasets)
        df = _engineer_features(merged)
        cls.model_df = prepare_features(df)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test, cls.train_df, cls.test_df = (
            split_data(cls.model_df)
        )

    def test_no_nulls(self):
        self.assertEqual(self.model_df.isnull().sum().sum(), 0)

    def test_all_features_present(self):
        for col in FEATURE_COLS:
            self.assertIn(col, self.model_df.columns)

    def test_no_data_leakage(self):
        self.assertLess(self.train_df["Date"].max(), self.test_df["Date"].min())

    def test_train_bigger_than_test(self):
        self.assertGreater(len(self.X_train), len(self.X_test))


class TestScaleFeatures(unittest.TestCase):
    """Tests for scale_features()."""

    @classmethod
    def setUpClass(cls):
        datasets = load_raw_datasets()
        merged = merge_raw(datasets)
        df = _engineer_features(merged)
        model_df = prepare_features(df)
        X_train, X_test, _, _, _, _ = split_data(model_df)
        cls.X_train_sc, cls.X_test_sc, cls.scaler = scale_features(X_train, X_test)

    def test_shapes_preserved(self):
        self.assertEqual(self.X_train_sc.shape[1], len(FEATURE_COLS))

    def test_train_mean_near_zero(self):
        means = np.mean(self.X_train_sc, axis=0)
        self.assertTrue(np.allclose(means, 0, atol=1e-10))


class TestEvaluateModel(unittest.TestCase):
    """Tests for evaluate_model()."""

    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = evaluate_model(y, y)
        self.assertEqual(result["MAE"], 0.0)
        self.assertEqual(result["R²"], 1.0)

    def test_returns_four_keys(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        result = evaluate_model(y_true, y_pred)
        self.assertEqual(set(result.keys()), {"MAE", "RMSE", "R²", "MAPE (%)"})


class TestTrainAndEvaluate(unittest.TestCase):
    """End-to-end training and evaluation test."""

    @classmethod
    def setUpClass(cls):
        datasets = load_raw_datasets()
        merged = merge_raw(datasets)
        df = _engineer_features(merged)
        model_df = prepare_features(df)
        X_train, X_test, y_train, y_test, _, _ = split_data(model_df)
        X_train_sc, X_test_sc, _ = scale_features(X_train, X_test)
        cls.models = get_models()
        cls.results = train_all_models(
            cls.models, X_train, y_train, X_train_sc, X_test, X_test_sc, y_test
        )

    def test_all_models_trained(self):
        self.assertEqual(set(self.results.keys()), set(self.models.keys()))

    def test_best_model_r2_above_threshold(self):
        name, _, _ = get_best_model(self.results, self.models)
        self.assertGreater(self.results[name]["R²"], 0.5)

    def test_results_table_sorted(self):
        table = build_results_table(self.results)
        rmse = table["RMSE"].values
        self.assertTrue((rmse[1:] >= rmse[:-1]).all())


if __name__ == "__main__":
    unittest.main()
