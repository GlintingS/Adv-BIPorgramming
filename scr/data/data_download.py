"""Download raw market & economic data from the internet and store under data/external/."""

import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from fredapi import Fred

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# FRED API key from environment
FRED_API_KEY = os.environ.get("FRED_API_KEY")

# Default collection window
START_DATE = "2018-01-01"
END_DATE = "2026-12-31"
MAX_DOWNLOAD_RETRIES = 3


# ── Helper: download a single yfinance ticker ────────────────────────
def _download_ticker(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    logger.info("Downloading %s ...", ticker)
    raw = None
    for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
            )
            if raw is not None and not raw.empty:
                break
            logger.warning("%s attempt %d returned empty DataFrame", ticker, attempt)
        except Exception as e:
            logger.warning("%s attempt %d failed: %s", ticker, attempt, e)
        if attempt < MAX_DOWNLOAD_RETRIES:
            time.sleep(2 * attempt)

    if raw is None or raw.empty:
        raise RuntimeError(
            f"Failed to download {ticker} after {MAX_DOWNLOAD_RETRIES} attempts (empty or None)."
        )

    raw = raw.reset_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.tz_localize(None)
    return raw.sort_values("Date").reset_index(drop=True)


def _load_cached_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    return pd.read_csv(path, parse_dates=["Date"])


# ── Public API ────────────────────────────────────────────────────────
def download_all(
    start: str = START_DATE,
    end: str = END_DATE,
    ext_dir: Path | str | None = None,
    fred_api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Download all raw datasets and save CSVs to *ext_dir* (default ``data/external/``).

    Parameters
    ----------
    fred_api_key : str, optional
        Override the FRED API key (useful for Streamlit secrets).

    Returns a dict mapping logical name → DataFrame.
    """
    ext_dir = Path(ext_dir) if ext_dir else EXTERNAL_DIR
    ext_dir.mkdir(parents=True, exist_ok=True)
    api_key = fred_api_key or FRED_API_KEY

    datasets = {}

    # ── Yahoo Finance tickers ─────────────────────────────────────────
    ticker_map = {
        "TSLA": "tesla_stock_raw.csv",
        "^GSPC": "sp500_raw.csv",
        "^NDX": "nasdaq_raw.csv",
        "^VIX": "vix_raw.csv",
    }

    for ticker, filename in ticker_map.items():
        path = ext_dir / filename
        try:
            df = _download_ticker(ticker, start, end)
            df.to_csv(path, index=False)
            logger.info("Saved %s → %s (%d rows)", ticker, path, len(df))
        except RuntimeError:
            if path.exists():
                logger.warning("Using cached %s from %s", ticker, path)
                df = _load_cached_csv(path)
            else:
                raise
        datasets[ticker] = df
        time.sleep(1)  # brief pause to avoid rate-limiting

    # ── FRED economic series ──────────────────────────────────────────
    fred: Any | None = None
    if api_key:
        fred = Fred(api_key=api_key)
    else:
        logger.warning(
            "FRED_API_KEY is not configured. Falling back to cached FRED CSV files if available."
        )

    fred_map = {
        "FEDFUNDS": ("Fed_Funds_Rate", "fed_funds_rate_raw.csv"),
        "CPIAUCSL": ("CPI", "cpi_raw.csv"),
        "UNRATE": ("Unemployment_Rate", "unemployment_rate_raw.csv"),
    }

    for series_id, (col_name, filename) in fred_map.items():
        series = None
        if fred is not None:
            logger.info("Downloading FRED: %s ...", series_id)
            for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
                try:
                    series = fred.get_series(
                        series_id, observation_start=start, observation_end=end
                    )
                    break
                except Exception as e:
                    logger.warning("FRED %s attempt %d failed: %s", series_id, attempt, e)
                    if attempt < MAX_DOWNLOAD_RETRIES:
                        time.sleep(2 * attempt)
        if series is None:
            # Fall back to existing CSV if available
            fallback = ext_dir / filename
            if fallback.exists():
                logger.warning("Using cached %s from %s", series_id, fallback)
                df = _load_cached_csv(fallback)
                datasets[series_id] = df
                continue
            raise RuntimeError(
                f"Failed to load {series_id}: no FRED API key and no cache file at {fallback}."
            )
        df = pd.DataFrame({"Date": series.index, col_name: series.values})
        df["Date"] = pd.to_datetime(df["Date"])
        path = ext_dir / filename
        df.to_csv(path, index=False)
        logger.info("Saved %s → %s (%d rows)", series_id, path, len(df))
        datasets[series_id] = df

    logger.info("All raw downloads complete (%d datasets).", len(datasets))
    return datasets


def load_raw_datasets(ext_dir=None):
    """Load previously saved raw CSVs from *ext_dir*. Returns the same dict
    structure as ``download_all``."""
    ext_dir = Path(ext_dir) if ext_dir else EXTERNAL_DIR

    files = {
        "TSLA": ("tesla_stock_raw.csv", None),
        "^GSPC": ("sp500_raw.csv", None),
        "^NDX": ("nasdaq_raw.csv", None),
        "^VIX": ("vix_raw.csv", None),
        "FEDFUNDS": ("fed_funds_rate_raw.csv", "Fed_Funds_Rate"),
        "CPIAUCSL": ("cpi_raw.csv", "CPI"),
        "UNRATE": ("unemployment_rate_raw.csv", "Unemployment_Rate"),
    }

    datasets: dict[str, pd.DataFrame] = {}
    for key, (filename, _) in files.items():
        path = ext_dir / filename
        df = _load_cached_csv(path)
        datasets[key] = df
        logger.info("Loaded %s from %s (%d rows)", key, path, len(df))

    return datasets


def merge_raw(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the raw datasets dict (from ``download_all`` or ``load_raw_datasets``)
    into a single DataFrame aligned on Tesla trading dates."""

    # Tesla base columns
    tsla = datasets["TSLA"].copy()
    if "Adj Close" not in tsla.columns:
        tsla["Adj Close"] = tsla["Close"]
    tsla = (
        tsla[["Date", "Close", "Volume", "Adj Close"]]
        .rename(
            columns={
                "Close": "TSLA_Close",
                "Adj Close": "TSLA_Adj_Close",
                "Volume": "TSLA_Volume",
            }
        )
        .sort_values("Date")
    )

    # Market indices
    index_map = {
        "^GSPC": "SP500_Close",
        "^NDX": "NASDAQ_Close",
        "^VIX": "VIX_Close",
    }
    merged = tsla.copy()
    for key, col_name in index_map.items():
        idx_df = datasets[key][["Date", "Close"]].rename(columns={"Close": col_name})
        merged = merged.merge(idx_df, on="Date", how="left")

    # FRED economic data
    econ_map = {
        "FEDFUNDS": "Fed_Funds_Rate",
        "CPIAUCSL": "CPI",
        "UNRATE": "Unemployment_Rate",
    }
    for key, col_name in econ_map.items():
        econ_df = datasets[key][["Date", col_name]].copy()
        merged = merged.merge(econ_df, on="Date", how="left")
        merged[col_name] = merged[col_name].ffill()

    # Fill short gaps
    merged = merged.sort_values("Date").reset_index(drop=True)
    for col in merged.columns:
        if col != "Date":
            merged[col] = merged[col].ffill(limit=5).bfill()

    # Save merged dataset to data/raw/
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / "tesla_merged_dataset.csv"
    merged.to_csv(raw_path, index=False)
    logger.info("Saved merged dataset → %s (%d rows)", raw_path, len(merged))

    logger.info("Merge complete: %s", merged.shape)
    return merged
