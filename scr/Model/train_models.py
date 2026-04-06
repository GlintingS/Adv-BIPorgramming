import logging
import pickle
import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit

from scr.Model.time_series_models import ARIMAForecaster, GARCHForecaster, LSTMForecaster

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_models() -> dict[str, Any]:
    return {
        "ARIMA": ARIMAForecaster(order=(3, 1, 2)),
        "GARCH": GARCHForecaster(p=1, q=1, mean_lags=1),
        "LSTM": LSTMForecaster(
            seq_len=30,
            hidden_size=64,
            num_layers=2,
            dropout=0.1,
            lr=0.001,
            epochs=35,
            batch_size=32,
            random_state=42,
        ),
    }


def train_all_models(
    models: dict[str, Any],
    X_train,
    y_train,
    X_train_scaled,
    X_test,
    X_test_scaled,
    y_test,
) -> dict[str, dict[str, Any]]:
    del X_train_scaled, X_test_scaled
    results: dict[str, dict[str, Any]] = {}
    y_train_arr = np.asarray(y_train)
    y_test_arr = np.asarray(y_test)

    for name, model in models.items():
        model.fit(X_train, y_train_arr)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test_arr, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_arr, y_pred))
        r2 = r2_score(y_test_arr, y_pred)
        mape = mean_absolute_percentage_error(y_test_arr, y_pred) * 100

        results[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "MAPE (%)": mape,
            "Predictions": y_pred,
        }
        logger.info(
            "%s  MAE=%.2f  RMSE=%.2f  R²=%.4f  MAPE=%.2f%%", name, mae, rmse, r2, mape
        )

    return results


def save_models(models: dict[str, Any], output_dir: Path | str | None = None) -> None:
    output_dir = Path(output_dir) if output_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        safe_name = name.replace(" ", "_").lower()
        path = output_dir / f"{safe_name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info("Saved %s to %s", name, path)


def load_model(name: str, model_dir: Path | str | None = None) -> Any:
    model_dir = Path(model_dir) if model_dir else MODELS_DIR
    safe_name = name.replace(" ", "_").lower()
    path = model_dir / f"{safe_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def cross_validate_models(
    models: dict[str, Any],
    X_train,
    y_train,
    X_train_scaled,
    n_splits: int = 5,
) -> pd.DataFrame:
    del X_train_scaled
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_results = []

    X_df = X_train if isinstance(X_train, pd.DataFrame) else pd.DataFrame(X_train)
    y_series = y_train if isinstance(y_train, pd.Series) else pd.Series(y_train)

    for name, base_model in models.items():
        scores = []
        for tr_idx, va_idx in tscv.split(X_df):
            X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
            y_tr, y_va = y_series.iloc[tr_idx], y_series.iloc[va_idx]

            # Recreate model per split to avoid state leakage.
            model = copy.deepcopy(base_model)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            scores.append(r2_score(y_va, y_pred))

        scores = np.asarray(scores, dtype=float)
        cv_results.append(
            {"Model": name, "CV R² Mean": scores.mean(), "CV R² Std": scores.std()}
        )
        logger.info("%s  CV R² = %.4f ± %.4f", name, scores.mean(), scores.std())

    return pd.DataFrame(cv_results).sort_values("CV R² Mean", ascending=False)
