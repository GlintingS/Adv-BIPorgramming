import logging
from dataclasses import dataclass

import numpy as np
from arch import arch_model
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _to_2d_array(X):
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


@dataclass
class ARIMAForecaster:
    order: tuple = (3, 1, 2)

    def __post_init__(self):
        self._model = None
        self._result = None

    def fit(self, X, y):
        exog = _to_2d_array(X)
        y_arr = np.asarray(y, dtype=float)

        self._model = SARIMAX(
            endog=y_arr,
            exog=exog,
            order=self.order,
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = self._model.fit(disp=False)
        return self

    def predict(self, X):
        if self._result is None:
            raise RuntimeError("ARIMAForecaster must be fit before predict().")
        exog_future = _to_2d_array(X)
        preds = self._result.forecast(steps=len(exog_future), exog=exog_future)
        return np.asarray(preds, dtype=float)


@dataclass
class GARCHForecaster:
    p: int = 1
    q: int = 1
    mean_lags: int = 1

    def __post_init__(self):
        self._result = None
        self._last_price = None
        self._last_variance = None

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        if len(y_arr) < 60:
            raise ValueError("GARCHForecaster needs at least 60 training observations.")

        log_returns = 100.0 * np.diff(np.log(y_arr))
        self._last_price = float(y_arr[-1])

        model = arch_model(
            log_returns,
            mean="AR",
            lags=self.mean_lags,
            vol="GARCH",
            p=self.p,
            q=self.q,
            dist="normal",
            rescale=False,
        )
        self._result = model.fit(disp="off")
        return self

    def predict(self, X):
        if self._result is None or self._last_price is None:
            raise RuntimeError("GARCHForecaster must be fit before predict().")

        horizon = len(X)
        forecast = self._result.forecast(horizon=horizon, reindex=False)
        mean_returns = np.asarray(forecast.mean.values[-1], dtype=float)[:horizon]
        self._last_variance = np.asarray(forecast.variance.values[-1], dtype=float)[:horizon]

        prices = np.empty(horizon, dtype=float)
        prev_price = self._last_price
        for i, r in enumerate(mean_returns):
            prev_price = prev_price * np.exp(r / 100.0)
            prices[i] = prev_price
        return prices

    def get_volatility_forecast(self):
        if self._last_variance is None:
            return None
        return np.sqrt(np.clip(self._last_variance, a_min=0.0, a_max=None))


class _LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


@dataclass
class LSTMForecaster:
    seq_len: int = 30
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 0.001
    epochs: int = 30
    batch_size: int = 32
    random_state: int = 42

    def __post_init__(self):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for LSTMForecaster.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model = None
        self._train_features_scaled = None

    def _build_sequences(self, X_scaled, y_scaled):
        X_seq, y_seq = [], []
        for i in range(self.seq_len, len(X_scaled)):
            X_seq.append(X_scaled[i - self.seq_len : i])
            y_seq.append(y_scaled[i])
        if not X_seq:
            raise ValueError("Not enough rows to build LSTM sequences.")
        return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)

    def fit(self, X, y):
        X_arr = _to_2d_array(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)

        X_scaled = self.feature_scaler.fit_transform(X_arr)
        y_scaled = self.target_scaler.fit_transform(y_arr).ravel()

        X_seq, y_seq = self._build_sequences(X_scaled, y_scaled)
        ds = TensorDataset(
            torch.from_numpy(X_seq),
            torch.from_numpy(y_seq).unsqueeze(1),
        )
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self.model = _LSTMRegressor(
            input_size=X_seq.shape[2],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

        self._train_features_scaled = X_scaled
        return self

    def predict(self, X):
        if self.model is None or self._train_features_scaled is None:
            raise RuntimeError("LSTMForecaster must be fit before predict().")

        X_future = _to_2d_array(X)
        X_future_scaled = self.feature_scaler.transform(X_future)
        all_features = np.vstack([self._train_features_scaled, X_future_scaled])

        start_idx = len(self._train_features_scaled)
        preds_scaled = []

        self.model.eval()
        with torch.no_grad():
            for idx in range(start_idx, start_idx + len(X_future_scaled)):
                window = all_features[idx - self.seq_len : idx]
                if len(window) < self.seq_len:
                    pad_rows = np.repeat(window[:1], self.seq_len - len(window), axis=0)
                    window = np.vstack([pad_rows, window])
                x_tensor = (
                    torch.from_numpy(window.astype(np.float32))
                    .unsqueeze(0)
                    .to(self.device)
                )
                pred = self.model(x_tensor).cpu().numpy().ravel()[0]
                preds_scaled.append(pred)

        preds_scaled = np.asarray(preds_scaled, dtype=float).reshape(-1, 1)
        preds = self.target_scaler.inverse_transform(preds_scaled).ravel()
        return preds
