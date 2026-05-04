"""Health check endpoint."""

from fastapi import APIRouter, status
from pydantic import BaseModel

from promptlab.core.config import settings


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    version: str
    environment: str


router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """
    Perform a health check on the application.

    Returns:
        HealthResponse: Application health status
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def root() -> HealthResponse:
    """Root endpoint returning basic application info."""
    return HealthResponse(
        status="running",
        version=settings.app_version,
        environment=settings.environment,
    )
