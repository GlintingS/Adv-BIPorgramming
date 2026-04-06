"""Shared fixtures for all pytest tests.

Loads cached data from data/external/ so tests run fast without API calls.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def raw_datasets():
    """Load cached raw CSVs from data/external/."""
    from scr.data.data_download import load_raw_datasets

    return load_raw_datasets()


@pytest.fixture(scope="session")
def merged_df(raw_datasets):
    """Merged DataFrame from all raw datasets."""
    from scr.data.data_download import merge_raw

    return merge_raw(raw_datasets)


@pytest.fixture(scope="session")
def engineered_df(merged_df):
    """DataFrame after feature engineering."""
    from scr.data.make_dataset import _engineer_features

    return _engineer_features(merged_df)


@pytest.fixture(scope="session")
def model_df(engineered_df):
    """Cleaned DataFrame ready for modelling."""
    from scr.data.make_dataset import prepare_features

    return prepare_features(engineered_df)


@pytest.fixture(scope="session")
def split_data_fixture(model_df):
    """Train/test split tuple."""
    from scr.data.make_dataset import split_data

    return split_data(model_df)


@pytest.fixture(scope="session")
def scaled_data(split_data_fixture):
    """Scaled features tuple."""
    from scr.data.make_dataset import scale_features

    X_train, X_test, y_train, y_test, train_df, test_df = split_data_fixture
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, scaler


@pytest.fixture(scope="session")
def trained_models(split_data_fixture, scaled_data):
    """Dict of trained models and results."""
    from scr.Model.train_models import get_models, train_all_models

    X_train, X_test, y_train, y_test, train_df, test_df = split_data_fixture
    X_train_scaled, X_test_scaled, scaler = scaled_data
    models = get_models()
    results = train_all_models(
        models, X_train, y_train, X_train_scaled, X_test, X_test_scaled, y_test
    )
    return models, results
