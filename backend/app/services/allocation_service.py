"""
Turns a frozen PPO policy into a portfolio allocation for the API.

Live inference steps:
    1. observation_builder fetches fresh market data and builds today's
       state exactly as the training environment did.
    2. The frozen policy predicts an action (deterministic)
    3. the action is softmax-normalised into weights.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from app.core.config import TICKER_LIST
from app.schemas.allocation import AllocationResponse, TickerWeight
from app.services.policy_store import load_policy


def _softmax(action):
    """Mirror StockPortfolioEnv.softmax_normalization."""
    exponentials = np.exp(action)
    return exponentials / np.sum(exponentials)


def _stub_response(variant, note) -> AllocationResponse:
    """Deterministic equal-weight fallback, clearly flagged."""
    weight = 1.0 / len(TICKER_LIST)
    return AllocationResponse(
        as_of=date.today(),
        variant=variant,
        weights=[TickerWeight(ticker=ticker, weight=weight) for ticker in sorted(TICKER_LIST)],
        policy_loaded=False,
        note=note,
    )


def get_allocation(variant) -> AllocationResponse:
    """Compute today's recommended allocation for one reward variant."""
    try:
        policy = load_policy(variant)
    except Exception as error:  # weights present but can't be loaded
        return _stub_response(variant, note=f"failed to load weights: {error}")
    if policy is None:
        return _stub_response(variant,note=f"no frozen weights for '{variant}'")

    try:
        from app.services.observation_builder import get_todays_observation
        observation = get_todays_observation()
    except Exception as error: 
        return _stub_response(variant, note=f"live observation unavailable: {error}")

    action, _ = policy.predict(observation.state, deterministic=True)
    weights = _softmax(np.asarray(action, dtype=float).flatten())

    produced_weights = [TickerWeight(ticker=ticker, weight=round(float(weight), 6)) for ticker, weight in zip(observation.tickers, weights)]
    
    return AllocationResponse(
        as_of=observation.as_of,
        variant=variant,
        weights=produced_weights,
        policy_loaded=True,
    )
