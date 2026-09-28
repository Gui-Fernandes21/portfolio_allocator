from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import pandas as pd

# Stubbing the FinRL modules I don't need
for _shortcut in ("finrl.test", "finrl.trade", "finrl.train"):
    _stub_module = types.ModuleType(_shortcut)
    setattr(_stub_module, _shortcut.rsplit(".", 1)[-1], None)
    sys.modules[_shortcut] = _stub_module

import yfinance as yf
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.config import INDICATORS

# constants
TICKER_LIST = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT"]
COVARIANCE_LOOKBACK = 252  # trading days used for each covariance matrix

# Full data span
DATA_START_DATE = "2012-01-01"
DATA_END_DATE = "2024-12-31"

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
RAW_CSV_PATH = CACHE_DIR / "raw_prices.csv"
PROCESSED_PICKLE_PATH = CACHE_DIR / "processed_dataset.pkl"


# Data fetching
def _standardize_ticker_frame(raw_frame, ticker) -> pd.DataFrame:
    """Reshape one yfinance download into FinRL's long-format schema."""
    frame = raw_frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):  # modern multi-ticker layout
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame = frame.rename(
        columns={
            frame.columns[0]: "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["tic"] = ticker
    return frame[["date", "open", "high", "low", "close", "volume", "tic"]]


def fetch_yahoo_data(ticker_list, start_date, end_date) -> pd.DataFrame:
    """Replacement for FinRL's YahooDownloader.fetch_data().
    
    'auto_adjust=True' returns an adjusted close price that.
    """
    frames = []
    
    for ticker in ticker_list:
        raw_frame = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        if raw_frame is not None and len(raw_frame) > 0:
            frames.append(_standardize_ticker_frame(raw_frame, ticker))
    if not frames:
        raise ValueError("no data fetched")
    
    data = pd.concat(frames, axis=0, ignore_index=True) # Combine all the data
    data["date"] = pd.to_datetime(data["date"]) # Convert date column to datetime
    data["day"] = data["date"].dt.dayofweek # Add a day of the week column
    data["date"] = data["date"].dt.strftime("%Y-%m-%d") # Format date column
    data = data.dropna().sort_values(["date", "tic"]).reset_index(drop=True) # Sort the data and clean NaN values
    
    return data 


# data preprocessing
def build_dataset(raw_dataframe) -> pd.DataFrame:
    """FinRL FeatureEngineer indicators + per-date covariance matrices.

    Output schema is the input of FinRLPortfolioEnv / StockPortfolioEnv
        -> long format with tech indicator columns plus 'cov_list' which is a list of covariance matrices.
    """
    
    feature_engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False,
    )
    
    processed_dataframe = feature_engineer.preprocess_data(raw_dataframe)
    processed_dataframe = processed_dataframe.sort_values(["date", "tic"], ignore_index=True)
    processed_dataframe.index = processed_dataframe.date.factorize()[0]

    # StockPortfolioEnv state = per-date covariance matrix + tech indicators.
    # The covariance at date t uses only the trailing window up to t. the first COVARIANCE_LOOKBACK days are dropped.
    covariance_list = [] 
    return_list = [] 
    date_list = []
    
    unique_days = processed_dataframe.index.unique()
    
    # Compute the covariance matrix for each day in the dataset, starting from the COVARIANCE_LOOKBACK-th day
    for day_index in range(COVARIANCE_LOOKBACK, len(unique_days)):
        lookback_slice = processed_dataframe.loc[day_index - COVARIANCE_LOOKBACK : day_index, :] # Get the lookback window
        price_pivot = lookback_slice.pivot_table(index="date", columns="tic", values="close")
        return_window = price_pivot.pct_change().dropna()
        covariance_list.append(return_window.cov().values)
        return_list.append(return_window)
        date_list.append(processed_dataframe.date.unique()[day_index])

    covariance_dataframe = pd.DataFrame({"date": date_list, "cov_list": covariance_list, "return_list": return_list})
    processed_dataframe = processed_dataframe.merge(covariance_dataframe, on="date")
    processed_dataframe = processed_dataframe.sort_values(["date", "tic"]).reset_index(drop=True)
    processed_dataframe.index = processed_dataframe.date.factorize()[0]
    
    return processed_dataframe


#  caching logic
def build_cache(start_date: str = DATA_START_DATE, end_date: str = DATA_END_DATE):
    
    CACHE_DIR.mkdir(exist_ok=True)
    
    print(f"fetching {TICKER_LIST} from {start_date} to {end_date} ...")
    
    raw_dataframe = fetch_yahoo_data(TICKER_LIST, start_date, end_date)
    raw_dataframe.to_csv(RAW_CSV_PATH, index=False)
    
    processed_dataframe = build_dataset(raw_dataframe)
    processed_dataframe.to_pickle(PROCESSED_PICKLE_PATH)
    
    first_date = processed_dataframe.date.min()
    last_date = processed_dataframe.date.max()
    
    print(
        f"saved processed dataset -> {PROCESSED_PICKLE_PATH} "
        f"({processed_dataframe.date.nunique()} trading days, "
        f"{first_date} .. {last_date})"
    )
    
    return processed_dataframe


def load_cached_dataset() -> pd.DataFrame:
    
    if not PROCESSED_PICKLE_PATH.exists():
        raise FileNotFoundError(f"no cached dataset at {PROCESSED_PICKLE_PATH}. try running 'python data_cache.py'.")
    
    return pd.read_pickle(PROCESSED_PICKLE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DATA_START_DATE)
    parser.add_argument("--end", default=DATA_END_DATE)
    parser.add_argument("--refresh", action="store_true", help="rebuild even if the cache exists")
    args = parser.parse_args()

    if PROCESSED_PICKLE_PATH.exists() and not args.refresh:
        print(f"cache already exists at {PROCESSED_PICKLE_PATH}; use --refresh")
    else:
        build_cache(args.start, args.end)
