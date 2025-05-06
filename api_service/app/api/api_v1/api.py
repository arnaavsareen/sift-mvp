"""
Main API router that includes all endpoint groups.
"""
from fastapi import APIRouter

from app.api.api_v1.endpoints import (
    auth,
    detections,
    cameras,
    violations,
    analytics,
    users,
    health
)

# Create the main API router
api_router = APIRouter()

# Include all endpoint groups
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

api_router.include_router(
    detections.router,
    prefix="/detections",
    tags=["Detections"]
)

api_router.include_router(
    cameras.router,
    prefix="/cameras",
    tags=["Cameras"]
)

api_router.include_router(
    violations.router,
    prefix="/violations",
    tags=["Violations"]
)

api_router.include_router(
    analytics.router,
    prefix="/analytics",
    tags=["Analytics"]
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"]
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)
