"""
FastAPI application entry point.

run API with:
uvicorn app.main:app --reload

Docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import allocation, health, recommendation
from app.core.config import API_TITLE, API_VERSION, CORS_ORIGINS

app = FastAPI(title=API_TITLE, version=API_VERSION)

# Setting up CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All endpoints are versioned under /api/v1.
app.include_router(health.router, prefix="/api/v1")
app.include_router(allocation.router, prefix="/api/v1")
app.include_router(recommendation.router, prefix="/api/v1")
