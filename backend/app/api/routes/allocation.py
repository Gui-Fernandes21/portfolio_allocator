from __future__ import annotations

from fastapi import APIRouter

from app.schemas.allocation import AllocationRequest, AllocationResponse
from app.services.allocation_service import get_allocation

router = APIRouter(prefix="/allocation", tags=["allocation"])

@router.post("", response_model=AllocationResponse)
# Return the recommended portfolio weights for each ETF produced by the frozen PPO policy for the requested reward variant.
def create_allocation(request: AllocationRequest) -> AllocationResponse:
    return get_allocation(request.variant)
