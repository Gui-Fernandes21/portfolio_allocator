from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.recommendation import (
    PipelineDescription,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.recommendation_service import (
    NoAllocationAvailable,
    describe_pipeline,
    get_recommendation,
)

router = APIRouter(tags=["recommendation"])


@router.post("/recommendation", response_model=RecommendationResponse)
# Produce a full recommendation.
def create_recommendation(request: RecommendationRequest) -> RecommendationResponse:
    """
    - if first call of the day, fetch market data and call a language model. Otherwise use cached data.
    - 'degraded' is true if any analyst or the model is unavailable
    - 'trace' records what was missing.
    """

    try:
        return get_recommendation(request.variant)
    except NoAllocationAvailable as error:
        raise HTTPException(
            status_code=503,
            detail=f"No allocation could be produced: {error}",
        ) from error


@router.get("/pipeline", response_model=PipelineDescription)
def get_pipeline() -> PipelineDescription:
    """The agents in execution order and which of them uses llm"""
    return describe_pipeline()
