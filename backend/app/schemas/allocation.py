from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

RewardVariant = Literal["raw", "dsr", "sortino", "cost"]

class AllocationRequest(BaseModel):
    """What the frontend sends when asking for a portfolio allocation."""

    variant: RewardVariant = Field(default="dsr",description="Which trained variant policy to use.")

class TickerWeight(BaseModel):
    """One ETF and its recommended portfolio weight."""

    ticker: str
    weight: float = Field(ge=0.0, le=1.0, description="Fraction of portfolio.")

class AllocationResponse(BaseModel):
    """Portfolio allocation returned to the frontend."""

    as_of: date
    variant: RewardVariant
    weights: list[TickerWeight]
    policy_loaded: bool = Field(
        description=(
            "True when the allocation came from frozen PPO weights. False when the API fell back to equal-weight."
        )
    )
    note: str | None = Field(
        default=None,
        description="why the fallback happened.",
    )

class HealthResponse(BaseModel):
    """Health check response with which policies are available."""

    status: Literal["ok"]
    available_variants: list[str]
