"""Central configuration for backend API."""
from __future__ import annotations

from pathlib import Path

# Project layout 
BACKEND_DIR = Path(__file__).resolve().parents[2]  
MODELS_DIR = BACKEND_DIR / "models"                

#  Portfolio universe 
TICKER_LIST = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT"]

#  Reward variants 
REWARD_VARIANTS = ["raw", "dsr", "sortino", "cost"]
DEFAULT_VARIANT = "dsr"

# Multi-agent layer 
LLM_AGENT_ROLES = ["market_analyst", "sentiment_analyser", "coordinator"]
DETERMINISTIC_AGENT_ROLES = ["execution_agent", "risk_manager"]

# LLM access is configured entirely with environment variables:
#   OPENROUTER_API_KEY   required for the LLM-backed agents
#   OPENROUTER_MODEL     optional model override

# API configuration 
API_TITLE = "ETF Portfolio Allocation API"
API_VERSION = "0.1.0"

# Origins allowed to call the API (CORS).
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

def model_weights_path(variant: str):
    """Return the expected path of the frozen policy for a reward variant."""
    return MODELS_DIR / f"ppo_{variant}.zip"
