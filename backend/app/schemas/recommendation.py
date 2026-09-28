from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.allocation import RewardVariant, TickerWeight

class RecommendationRequest(BaseModel):
    """What the frontend sends to ask for a full recommendation."""

    variant: RewardVariant = Field(default="dsr",description="Which trained reward-variant policy to use.")


class TraceEntry(BaseModel):
    """One agent's contribution, for the reasoning trail."""

    agent: str
    summary: str
    used_llm: bool = Field(description=("True if this line was written by a language model."))
    ok: bool
    error: str | None = None


class RiskAdjustment(BaseModel):
    """One constraint that fired, and what it did."""

    rule: str
    detail: str


class RiskReport(BaseModel):
    """The deterministic risk layer's output."""

    adjustments: list[RiskAdjustment] = Field(default_factory=list,description="Empty when no limit was exceeded")
    constraints: dict[str, Any] = Field(default_factory=dict,description="The limits imposed - inform the user")


class Explanation(BaseModel):
    """The plain-language account shown to the user."""

    text: str
    generated: bool = Field(
        description=(
            "True if written by a language model and verified against the allocation. False if composed from deterministic template."
        )
    )
    fallback_reason: str | None = Field(
        default=None,
        description=("Why the template was used.")
    )


class MarketContext(BaseModel):
    """Computed market statistics."""

    available: bool
    as_of: str | None = None
    narrative: str | None = None
    per_asset: dict[str, Any] = Field(default_factory=dict)


class NewsSource(BaseModel):
    """One headline, with the attribution."""

    ticker: str
    title: str
    publisher: str
    published: str | None = None
    url: str | None = None
    sentiment: str | None = None
    reason: str | None = None


class SentimentContext(BaseModel):
    """News tone per ETF, with every headline attached."""

    available: bool
    classified: bool = False
    per_asset: dict[str, Any] = Field(default_factory=dict)
    sources: list[NewsSource] = Field(default_factory=list)


class RecommendationContext(BaseModel):
    """
    Evidence presented alongside the recommendation.
    
    For the user to weigh in their decision.
    """

    market: MarketContext
    sentiment: SentimentContext


class RecommendationResponse(BaseModel):
    """The complete multi-agent recommendation."""

    as_of: date
    variant: RewardVariant
    policy_loaded: bool = Field(
        description=(
            "True when the weights came from the frozen policy. False when changed for a equal-weight placeholder."
        )
    )
    degraded: bool = Field(
        description=("True if any part of the pipeline failed or fell back.")
    )
    weights: list[TickerWeight] = Field(
        description="Final weights, after any risk adjustment."
    )
    risk: RiskReport
    explanation: Explanation
    context: RecommendationContext
    trace: list[TraceEntry] = Field(
        description="What each agent contributed"
    )


class PipelineDescription(BaseModel):
    agents: list[dict[str, Any]]
    llm_free_agents: list[str] = Field(
        description=(
            "Agents that never call a language model. Allocation production and risk constraints."
        )
    )
