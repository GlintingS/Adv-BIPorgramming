"""Tests for scr.data.data_download and scr.data.make_dataset modules."""

import pandas as pd
import numpy as np
import pytest

from scr.data.data_download import (
    load_raw_datasets,
    merge_raw,
    EXTERNAL_DIR,
    RAW_DIR,
)
from scr.data.make_dataset import (
    _engineer_features,
    prepare_features,
    split_data,
    scale_features,
    FEATURE_COLS,
    TARGET_COL,
    PROCESSED_DIR,
)


# ── data_download tests ──────────────────────────────────────────────
class TestLoadRawDatasets:
    """Tests for load_raw_datasets()."""

    def test_returns_dict_with_seven_keys(self, raw_datasets):
        assert isinstance(raw_datasets, dict)
        assert len(raw_datasets) == 7

    def test_expected_keys_present(self, raw_datasets):
        expected = {"TSLA", "^GSPC", "^NDX", "^VIX", "FEDFUNDS", "CPIAUCSL", "UNRATE"}
        assert set(raw_datasets.keys()) == expected

    def test_each_dataset_is_dataframe(self, raw_datasets):
        for key, df in raw_datasets.items():
            assert isinstance(df, pd.DataFrame), f"{key} is not a DataFrame"

    def test_each_dataset_has_date_column(self, raw_datasets):
        for key, df in raw_datasets.items():
            assert "Date" in df.columns, f"{key} missing 'Date' column"

    def test_date_column_is_datetime(self, raw_datasets):
        for key, df in raw_datasets.items():
            assert pd.api.types.is_datetime64_any_dtype(
                df["Date"]
            ), f"{key} Date column is not datetime"

    def test_no_empty_datasets(self, raw_datasets):
        for key, df in raw_datasets.items():
            assert len(df) > 0, f"{key} dataset is empty"

    def test_external_csv_files_exist(self):
        expected_files = [
            "tesla_stock_raw.csv",
            "sp500_raw.csv",
            "nasdaq_raw.csv",
            "vix_raw.csv",
            "fed_funds_rate_raw.csv",
            "cpi_raw.csv",
            "unemployment_rate_raw.csv",
        ]
        for fname in expected_files:
            assert (EXTERNAL_DIR / fname).exists(), f"Missing {fname}"


class TestMergeRaw:
    """Tests for merge_raw()."""

    def test_returns_dataframe(self, merged_df):
        assert isinstance(merged_df, pd.DataFrame)

    def test_has_required_columns(self, merged_df):
        required = [
            "Date",
            "TSLA_Close",
            "SP500_Close",
            "NASDAQ_Close",
            "VIX_Close",
            "Fed_Funds_Rate",
            "CPI",
            "Unemployment_Rate",
        ]
        for col in required:
            assert col in merged_df.columns, f"Missing column: {col}"

    def test_no_null_in_key_columns(self, merged_df):
        for col in ["TSLA_Close", "SP500_Close", "NASDAQ_Close"]:
            assert merged_df[col].notna().all(), f"NaN found in {col}"

    def test_dates_are_sorted(self, merged_df):
        dates = merged_df["Date"].values
        assert (dates[1:] >= dates[:-1]).all(), "Dates are not sorted"

    def test_row_count_reasonable(self, merged_df):
        assert len(merged_df) > 1000, "Too few rows in merged dataset"

    def test_merged_csv_saved(self):
        assert (RAW_DIR / "tesla_merged_dataset.csv").exists()


# ── make_dataset tests ───────────────────────────────────────────────
class TestEngineerFeatures:
    """Tests for _engineer_features()."""

    def test_adds_daily_returns(self, engineered_df):
        for col in ["TSLA_Daily_Return", "SP500_Daily_Return", "NASDAQ_Daily_Return"]:
            assert col in engineered_df.columns

    def test_adds_calendar_features(self, engineered_df):
        for col in ["Month", "Quarter", "Year"]:
            assert col in engineered_df.columns

    def test_adds_lag_features(self, engineered_df):
        for col in ["TSLA_Return_Lag1", "TSLA_Return_Lag3"]:
            assert col in engineered_df.columns

    def test_adds_rolling_features(self, engineered_df):
        for col in ["TSLA_MA_7", "TSLA_MA_30", "TSLA_Vol_30"]:
            assert col in engineered_df.columns

    def test_adds_highvol_regime(self, engineered_df):
        assert "HighVol_Regime" in engineered_df.columns
        assert set(engineered_df["HighVol_Regime"].dropna().unique()).issubset({0, 1})

    def test_shape_increases(self, merged_df, engineered_df):
        assert engineered_df.shape[1] > merged_df.shape[1]


class TestPrepareFeatures:
    """Tests for prepare_features()."""

    def test_returns_dataframe(self, model_df):
        assert isinstance(model_df, pd.DataFrame)

    def test_no_null_values(self, model_df):
        assert model_df.isnull().sum().sum() == 0

    def test_has_all_feature_columns(self, model_df):
        for col in FEATURE_COLS:
            assert col in model_df.columns, f"Missing feature: {col}"

    def test_has_target_column(self, model_df):
        assert TARGET_COL in model_df.columns

    def test_has_date_column(self, model_df):
        assert "Date" in model_df.columns

    def test_processed_csv_saved(self):
        assert (PROCESSED_DIR / "tesla_processed_dataset.csv").exists()


class TestSplitData:
    """Tests for split_data()."""

    def test_returns_six_elements(self, split_data_fixture):
        assert len(split_data_fixture) == 6

    def test_train_larger_than_test(self, split_data_fixture):
        X_train, X_test, y_train, y_test, train_df, test_df = split_data_fixture
        assert len(X_train) > len(X_test)

    def test_no_data_leakage(self, split_data_fixture):
        X_train, X_test, y_train, y_test, train_df, test_df = split_data_fixture
        assert train_df["Date"].max() < test_df["Date"].min()

    def test_feature_count_matches(self, split_data_fixture):
        X_train, X_test, *_ = split_data_fixture
        assert X_train.shape[1] == len(FEATURE_COLS)
        assert X_test.shape[1] == len(FEATURE_COLS)

    def test_target_is_series(self, split_data_fixture):
        _, _, y_train, y_test, _, _ = split_data_fixture
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_approximately_80_20_split(self, split_data_fixture, model_df):
        X_train, X_test, *_ = split_data_fixture
        ratio = len(X_train) / (len(X_train) + len(X_test))
        assert 0.75 <= ratio <= 0.85


class TestScaleFeatures:
    """Tests for scale_features()."""

    def test_returns_three_elements(self, scaled_data):
        assert len(scaled_data) == 3

    def test_scaled_shape_matches_input(self, split_data_fixture, scaled_data):
        X_train, X_test, *_ = split_data_fixture
        X_train_scaled, X_test_scaled, _ = scaled_data
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

    def test_train_mean_near_zero(self, scaled_data):
        X_train_scaled, _, _ = scaled_data
        means = np.mean(X_train_scaled, axis=0)
        assert np.allclose(means, 0, atol=1e-10)

    def test_train_std_near_one(self, scaled_data):
        X_train_scaled, _, _ = scaled_data
        stds = np.std(X_train_scaled, axis=0)
        assert np.allclose(stds, 1, atol=0.1)
