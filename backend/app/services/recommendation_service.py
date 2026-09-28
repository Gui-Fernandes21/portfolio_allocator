"""
Runs the multi-agent pipeline and maps its output onto the API schema.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from app.schemas.recommendation import (
    Explanation,
    MarketContext,
    NewsSource,
    PipelineDescription,
    RecommendationContext,
    RecommendationResponse,
    RiskReport,
    SentimentContext,
    TickerWeight,
    TraceEntry,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

class NoAllocationAvailable(RuntimeError):
    """Raised when the pipeline could not produce any weights at all"""


def _parse_as_of(value, fallback) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            pass
    return fallback


def get_recommendation(variant) -> RecommendationResponse:
    """Run the five-agent pipeline and shape the result for the frontend."""
    from agents import build_pipeline

    context = build_pipeline().run({"variant": variant})

    allocation = (context.payload_of("risk_manager") or context.payload_of("execution_agent"))
    coordinator = context.payload_of("coordinator")
    market = context.payload_of("market_analyst")
    sentiment = context.payload_of("sentiment_analyser")

    weights = allocation.get("weights") or []
    
    if not weights:
        failures = [o.error for o in context.outputs if not o.ok and o.error]
        raise NoAllocationAvailable("; ".join(failures) or "pipeline produced no allocation")

    explanation_text = coordinator.get("explanation")
    if not explanation_text:
        explanation_text = "An allocation was produced, but with no explanation."

    trace = [TraceEntry(**entry) for entry in context.trace()]
    # if any agent failed | the recommendation is degraded | if produced a fallback allocation
    degraded = (
        not allocation.get("policy_loaded", False)
        or any(not entry.ok for entry in trace)
        or not coordinator.get("generated", False)
    )

    return RecommendationResponse(
        as_of=_parse_as_of(allocation.get("as_of"), date.today()),
        variant=allocation.get("variant") or variant,
        policy_loaded=bool(allocation.get("policy_loaded", False)),
        degraded=degraded,
        weights=[TickerWeight(**entry) for entry in weights],
        risk=RiskReport(adjustments=allocation.get("adjustments", []),constraints=allocation.get("constraints", {})),
        explanation=Explanation(
            text=explanation_text,
            generated=bool(coordinator.get("generated", False)),
            fallback_reason=coordinator.get("fallback_reason"),
        ),
        context=RecommendationContext(
            market=MarketContext(
                available=bool(market),
                as_of=market.get("as_of"),
                narrative=market.get("narrative"),
                per_asset=market.get("per_asset", {}),
            ),
            sentiment=SentimentContext(
                available=bool(sentiment),
                classified=bool(sentiment.get("classified", False)),
                per_asset=sentiment.get("per_asset", {}),
                sources=[NewsSource(**source) for source in sentiment.get("sources", [])],
            ),
        ),
        trace=trace,
    )


def describe_pipeline() -> PipelineDescription:
    """The pipeline's shape, including which deterministic agents."""
    from agents import build_pipeline

    described = build_pipeline().describe()
    deterministic_agents = [a["agent"] for a in described if not a["uses_llm"]]
    return PipelineDescription(agents=described, llm_free_agents=deterministic_agents)
