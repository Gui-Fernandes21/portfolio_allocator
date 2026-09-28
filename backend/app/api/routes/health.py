"""
- Health check endpoint 
- Returns which frozen policies are available for the frontend to populate its selector.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.schemas.allocation import HealthResponse
from app.services.policy_store import available_variants

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", available_variants=available_variants())
