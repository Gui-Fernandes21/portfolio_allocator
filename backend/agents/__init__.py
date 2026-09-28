"""
agents
======
The five project agents, assigned by functional role.

    market_analyst       LLM     price/macro context for the user
    sentiment_analyser   LLM     news sentiment per asset
    execution_agent      no LLM  runs the frozen PPO policy  <- allocation path
    risk_manager         no LLM  deterministic constraint checks <- allocation path
    coordinator          LLM     assembles the explanation

Execution order is fixed by ``build_pipeline`` below. The two agents on the
allocation path import nothing from ``orchestration.llm``, which is the
mechanical guarantee behind the claim that allocations are reproducible;
``tests/test_agent_boundaries.py`` enforces it.
"""
from __future__ import annotations

from orchestration.pipeline import Pipeline


def build_pipeline() -> Pipeline:
    """
    The standard allocation pipeline.

    Analysts run first so their context is available to the Coordinator,
    but note that their outputs are NOT consumed by the Execution Agent:
    they are presented to the user as independent evidence, never as policy
    inputs. This ordering is what the design chapter describes.
    """
    from agents.coordinator import CoordinatorAgent
    from agents.execution_agent import ExecutionAgent
    from agents.market_analyst import MarketAnalystAgent
    from agents.risk_manager import RiskManagerAgent
    from agents.sentiment_analyser import SentimentAnalyserAgent

    return Pipeline([
        MarketAnalystAgent(),
        SentimentAnalyserAgent(),
        ExecutionAgent(),      # allocation produced here
        RiskManagerAgent(),    # allocation constrained here
        CoordinatorAgent(),    # explanation assembled here
    ])
