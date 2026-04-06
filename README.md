# Tesla Stock Price Prediction

A time-series forecasting project that predicts Tesla (TSLA) closing prices using market indices, economic indicators, and engineered features. The modeling stack combines ARIMA for trend, GARCH for volatility, and LSTM for long-sequence learning.

## Project Structure

```
AdvBIP Final - Tesla Stock Price Prediction/
├── main.py                              # Full ML pipeline (download → train → tune → evaluate → plot)
├── streamlit_AdvProgrammingFinal.py     # Streamlit web dashboard
├── requirements.txt                     # Python dependencies
├── data/
│   ├── external/                        # Raw downloads (yfinance + FRED CSVs)
│   ├── raw/                             # Merged dataset (tesla_merged_dataset.csv)
│   └── processed/                       # Cleaned & feature-engineered dataset
├── models/                              # Trained model files (.pkl)
├── scr/
│   ├── data/
│   │   ├── data_download.py             # Download data from yfinance & FRED APIs
│   │   └── make_dataset.py              # Feature engineering, train/test split, scaling
│   ├── Model/
│   │   ├── train_models.py              # Model training, saving, cross-validation
│   │   ├── hyper_tuning.py              # GridSearchCV hyperparameter tuning
│   │   └── predict_models.py            # Evaluation metrics & error analysis
│   └── visuals/
│       └── visualize.py                 # Visualization functions (6 plot types)
```

## Data Sources

- **Yahoo Finance** (via `yfinance`): TSLA, S&P 500 (^GSPC), NASDAQ (^NDX), VIX (^VIX)
- **FRED API** (via `fredapi`): Federal Funds Rate, CPI, Unemployment Rate

## Models

- ARIMA (AutoRegressive Integrated Moving Average) for trend analysis
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity) for volatility modeling
- LSTM (Long Short-Term Memory) neural network for sequential dependency learning
- Time-series hyperparameter tuning with rolling validation

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - **streamlit** — Web dashboard
  - **pandas / numpy** — Data manipulation
  - **scikit-learn** — metrics, preprocessing
  - **statsmodels** — ARIMA/SARIMAX implementation
  - **arch** — GARCH volatility modeling
  - **torch** — LSTM implementation
  - **matplotlib / seaborn** — Visualization
  - **yfinance** — Yahoo Finance API
  - **fredapi** — FRED economic data API

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure secrets:
  - Copy `.env.example` to `.env` and set `FRED_API_KEY`, or
  - For Streamlit Cloud, set `FRED_API_KEY` in App Settings -> Secrets.

## Usage

Run the full pipeline:
```bash
python main.py
```

Launch the Streamlit dashboard:
```bash
streamlit run streamlit_AdvProgrammingFinal.py
```
This will:
- Load and preprocess Tesla + macroeconomic time-series data
- Save cleaned data to `data/processed/tesla_processed_dataset.csv`
- Train ARIMA, GARCH, and LSTM models
- Save trained models to the `models/` directory (`arima.pkl`, `garch.pkl`, `lstm.pkl`)
- Evaluate all models and display comparison charts
- Write logs to `pipeline.log`

The Streamlit app allows you to:
- Compare ARIMA, GARCH, and LSTM performance (MAE, RMSE, R², MAPE)
- Visualize actual vs predicted Tesla prices over time
- Analyze regime and monthly errors
- Inspect forecasted volatility from the GARCH model

- Optionally run cross-validation from the sidebar (disabled by default for faster startup)

> **Note:** The app expects trained model files in `models/`. Run the training pipeline first.

## Streamlit Deployment

1. Ensure `runtime.txt` exists with your Python version (included in this repo).
2. Add `FRED_API_KEY` in Streamlit Cloud secrets.
3. Deploy with main file: `streamlit_AdvProgrammingFinal.py`.
4. If external APIs are temporarily unavailable, the app will fall back to cached CSV data when available.

## Logging

All modules use Python's built-in `logging` library. When running the training pipeline, logs are written to both the console and `pipeline.log`. Log messages cover data loading, model training, evaluation, and any errors encountered.

## Author

Frank Song
