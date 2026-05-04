"""Tests for health check endpoints."""

import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """Test the health check endpoint."""
    response = await client.get("/api/v1/health")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "environment" in data


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient) -> None:
    """Test the root endpoint."""
    response = await client.get("/api/v1/")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "running"
    assert "version" in data
    assert "environment" in data
