"""Hyperparameter tuning for ARIMA, GARCH, and LSTM models."""

import copy
import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from scr.Model.time_series_models import ARIMAForecaster, GARCHForecaster, LSTMForecaster

logger = logging.getLogger(__name__)


def get_param_grids() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return a dict of model name -> (estimator, param_grid)."""
    return {
        "ARIMA": (
            ARIMAForecaster(),
            {
                "order": [(2, 1, 1), (3, 1, 2), (5, 1, 2)],
            },
        ),
        "GARCH": (
            GARCHForecaster(),
            {
                "p": [1, 2],
                "q": [1, 2],
                "mean_lags": [1, 2],
            },
        ),
        "LSTM": (
            LSTMForecaster(),
            {
                "seq_len": [20, 30],
                "hidden_size": [32, 64],
                "num_layers": [1, 2],
                "dropout": [0.0, 0.1],
                "epochs": [20, 35],
            },
        ),
    }


def _iter_param_combinations(param_grid: dict[str, list[Any]]):
    keys = list(param_grid.keys())
    if not keys:
        yield {}
        return

    def _walk(i, current):
        if i == len(keys):
            yield dict(current)
            return
        key = keys[i]
        for v in param_grid[key]:
            current[key] = v
            yield from _walk(i + 1, current)

    yield from _walk(0, {})


def _validation_split(X_train, y_train, val_ratio: float = 0.2):
    split_idx = int(len(X_train) * (1 - val_ratio))
    split_idx = max(split_idx, 60)

    if isinstance(X_train, pd.DataFrame):
        X_tr = X_train.iloc[:split_idx]
        X_va = X_train.iloc[split_idx:]
    else:
        X_tr = X_train[:split_idx]
        X_va = X_train[split_idx:]

    if isinstance(y_train, pd.Series):
        y_tr = y_train.iloc[:split_idx]
        y_va = y_train.iloc[split_idx:]
    else:
        y_tr = y_train[:split_idx]
        y_va = y_train[split_idx:]

    return X_tr, X_va, y_tr, y_va


def tune_model(name: str, estimator, param_grid: dict[str, list[Any]], X_train, y_train):
    """Run a lightweight time-series validation search for one model."""
    X_tr, X_va, y_tr, y_va = _validation_split(X_train, y_train)

    best_score = -np.inf
    best_rmse = np.inf
    best_params = None
    best_model = None

    for params in _iter_param_combinations(param_grid):
        candidate = copy.deepcopy(estimator)
        for k, v in params.items():
            setattr(candidate, k, v)

        try:
            candidate.fit(X_tr, y_tr)
            preds = candidate.predict(X_va)
            score = r2_score(y_va, preds)
            rmse = np.sqrt(mean_squared_error(y_va, preds))
        except Exception as exc:
            logger.warning("%s params=%s failed: %s", name, params, exc)
            continue

        if score > best_score:
            best_score = score
            best_rmse = rmse
            best_params = params
            best_model = candidate

    if best_model is None:
        raise RuntimeError(f"No valid parameter combination found for {name}.")

    logger.info("%s best R²=%.4f RMSE=%.2f params=%s", name, best_score, best_rmse, best_params)
    return {
        "best_estimator": best_model,
        "best_score": best_score,
        "best_rmse": best_rmse,
        "best_params": best_params,
    }


def tune_all_models(
    X_train,
    y_train,
    X_train_scaled,
    n_splits: int = 5,
    scoring: str = "r2",
):
    """Tune all models and return fitted estimators + summary DataFrame.

    Arguments are intentionally kept compatible with the previous pipeline API.
    """
    del X_train_scaled, n_splits, scoring
    grids = get_param_grids()
    tuned_models = {}
    summary_rows = []

    for name, (estimator, param_grid) in grids.items():
        result = tune_model(name, estimator, param_grid, X_train, y_train)

        # Refit selected model on full training set before returning.
        best_model = result["best_estimator"]
        best_model.fit(X_train, y_train)
        tuned_models[name] = best_model

        summary_rows.append(
            {
                "Model": name,
                "Best CV R²": result["best_score"],
                "Best RMSE": result["best_rmse"],
                "Best Params": result["best_params"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("Best CV R²", ascending=False)
    logger.info("\n%s", summary_df.to_string(index=False))
    return tuned_models, summary_df


def _grid_size(param_grid):
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size
