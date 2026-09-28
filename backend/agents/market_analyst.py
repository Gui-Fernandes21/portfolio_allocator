"""
Goals of this implementation:
- Computes trailing performance statistics for a list of ETFs.
- Turns recent price action into a short, readable market context for the user.
"""

from __future__ import annotations

from datetime import date, timedelta

from orchestration.base import Agent
from orchestration.llm import LLMUnavailable, complete
from orchestration.messages import AgentContext, AgentOutput

TICKER_LIST = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT"]

# Trading-day windows used for the trailing statistics.
WINDOW_ONE_MONTH = 21
WINDOW_THREE_MONTHS = 63
WINDOW_ONE_YEAR = 252
VOLATILITY_WINDOW = 60
FETCH_CALENDAR_DAYS = 500  # enough calendar days to cover 252 trading days

SYSTEM_PROMPT = (
    "You are a market analyst writing two or three sentences for a "
    "non-expert retail investor. You will be given computed performance "
    "figures. Describe only those figures: do not add any number that is "
    "not given to you, do not predict, and do not recommend buying, selling "
    "or any portfolio weighting. Plain language, no jargon."
)

# One narrative per calendar day is good enough.
_cache: dict[date, dict] = {}


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def compute_market_facts(as_of: date | None = None) -> dict:
    """
    Trailing performance statistics per ETF, computed from price data.

    Returns a dict with as_of and per_asset.
    """
    import sys
    from pathlib import Path

    offline_dir = Path(__file__).resolve().parents[1] / "offline"
    if str(offline_dir) not in sys.path:
        sys.path.insert(0, str(offline_dir))

    import numpy as np
    import pandas as pd
    from data_cache import fetch_yahoo_data  # same fetch code as training

    today = as_of or date.today()
    start = (today - timedelta(days=FETCH_CALENDAR_DAYS)).isoformat()
    end = (today + timedelta(days=1)).isoformat()

    raw = fetch_yahoo_data(TICKER_LIST, start, end)
    prices = raw.pivot_table(index="date", columns="tic", values="close").sort_index()
    
    if prices.empty:
        raise ValueError("no price data returned")

    daily_returns = prices.pct_change()
    per_asset: dict[str, dict] = {}

    for ticker in sorted(prices.columns):
        series = prices[ticker].dropna()
        if len(series) < 2:
            continue

        def trailing_return(window: int) -> float | None:
            if len(series) <= window:
                return None
            return float(series.iloc[-1] / series.iloc[-(window + 1)] - 1.0)

        recent = daily_returns[ticker].dropna().tail(VOLATILITY_WINDOW)
        volatility = (
            float(recent.std(ddof=1) * np.sqrt(252)) if len(recent) > 1 else None
        )
        year_window = series.tail(WINDOW_ONE_YEAR)
        peak = float(year_window.max())
        drawdown = float(series.iloc[-1] / peak - 1.0) if peak > 0 else None

        per_asset[ticker] = {
            "last_close": round(float(series.iloc[-1]), 2),
            "return_1m": trailing_return(WINDOW_ONE_MONTH),
            "return_3m": trailing_return(WINDOW_THREE_MONTHS),
            "return_12m": trailing_return(WINDOW_ONE_YEAR),
            "volatility_annualised_60d": volatility,
            "drawdown_from_12m_high": drawdown,
        }

    return {
        "as_of": str(prices.index[-1]),
        "per_asset": per_asset,
    }


def build_prompt(facts: dict) -> str:
    lines = [
        f"Market data as of {facts['as_of']}. Trailing total returns, "
        "annualised 60-day volatility, and drawdown from the 12-month high:",
        "",
        f"{'ETF':<5} {'1M':>8} {'3M':>8} {'12M':>8} {'Vol':>8} {'DD':>8}",
    ]
    for ticker, stats in facts["per_asset"].items():
        lines.append(
            f"{ticker:<5} {_percent(stats['return_1m']):>8} "
            f"{_percent(stats['return_3m']):>8} "
            f"{_percent(stats['return_12m']):>8} "
            f"{_percent(stats['volatility_annualised_60d']):>8} "
            f"{_percent(stats['drawdown_from_12m_high']):>8}"
        )
    lines += [
        "",
        "The seven ETFs are US large cap (SPY), US technology (QQQ), US small "
        "cap (IWM), developed international (EFA), emerging markets (EEM), "
        "gold (GLD) and long-dated US Treasuries (TLT).",
        "",
        "In two or three sentences, describe what these figures show about "
        "recent conditions across these asset classes.",
    ]
    return "\n".join(lines)


class MarketAnalystAgent(Agent):
    name = "market_analyst"
    uses_llm = True

    def run(self, context: AgentContext) -> AgentOutput:
        today = date.today()
        if today in _cache:
            cached = _cache[today]
            return self.output(payload=cached, summary=cached["summary_line"])

        # deterministic part
        try:
            facts = compute_market_facts()
        except Exception as error:
            return self.output(
                error=f"{type(error).__name__}: {error}",
                summary="Market context unavailable: price data could not be retrieved.",
            )

        # narration
        narrative = None
        narrative_error = None
        try:
            narrative = complete(build_prompt(facts),
                                 role="market_analyst",
                                 system=SYSTEM_PROMPT,
                                 max_tokens=300)
            
            narrative = narrative.text.strip()
        except LLMUnavailable as error:
            narrative_error = str(error)

        payload = {
            "as_of": facts["as_of"],
            "per_asset": facts["per_asset"],
            "narrative": narrative,
            "narrative_available": narrative is not None,
            "narrative_error": narrative_error,
        }
        payload["summary_line"] = (
            narrative if narrative else
            f"Computed trailing performance for "
            f"{len(facts['per_asset'])} ETFs as of {facts['as_of']}; "
            "written commentary unavailable."
        )

        _cache.clear()
        _cache[today] = payload
        return self.output(payload=payload, summary=payload["summary_line"])
