"""
-> Builds TODAY's observation for the frozen PPO policies, in exactly the form
the training environment produced.

-> One yfinance fetch per calendar day per process, however many allocation requests arrive.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from app.core.config import BACKEND_DIR

# 252 trading days of covariance lookback plus indicator warmup fits in ~600 calendar days.
FETCH_CALENDAR_DAYS = 600

_OFFLINE_DIR = BACKEND_DIR / "offline"


@dataclass(frozen=True)
class Observation:
    """One day's policy input plus the metadata the API response needs."""

    state: np.ndarray          # (stock_dim + n_indicators, stock_dim)
    as_of: date                # last trading day the state is built from
    tickers: list[str]         # column order of the state 


_cache: dict[date, Observation] = {}


def _offline_pipeline():
    """Import the offline feature pipeline"""
    if str(_OFFLINE_DIR) not in sys.path:
        sys.path.insert(0, str(_OFFLINE_DIR))
    import data_cache  # backend/offline/data_cache.py
    return data_cache


def observation_from_processed(processed_dataframe: pd.DataFrame) -> Observation:
    """
    Extract the LAST date's observation from a processed dataset"""
    from finrl.config import INDICATORS  # same list training used

    last_date = processed_dataframe.date.max()
    day_slice = processed_dataframe[processed_dataframe.date == last_date]
    day_slice = day_slice.sort_values("tic")  # env sorted by ticker

    covariance_matrix = np.array(day_slice["cov_list"].values[0])
    state = np.append(
        covariance_matrix,
        [day_slice[indicator].values.tolist() for indicator in INDICATORS],
        axis=0,
    )
    return Observation(
        state=state,
        as_of=pd.Timestamp(last_date).date(),
        tickers=day_slice.tic.tolist(),
    )


def get_todays_observation() -> Observation:
    """Fetch fresh market data and build today's observation (cached daily)."""
    today = date.today()
    
    if today in _cache:
        return _cache[today]

    data_cache = _offline_pipeline()
    start_date = (today - timedelta(days=FETCH_CALENDAR_DAYS)).isoformat()
    end_date = (today + timedelta(days=1)).isoformat()

    raw_dataframe = data_cache.fetch_yahoo_data(data_cache.TICKER_LIST, start_date, end_date)
    processed_dataframe = data_cache.build_dataset(raw_dataframe)
    observation = observation_from_processed(processed_dataframe)

    _cache.clear()  # yesterday's entry is not relevant anymore
    _cache[today] = observation
    return observation
