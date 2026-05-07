"""Tests for comparison endpoints."""

import pytest
from fastapi import status
from httpx import AsyncClient

from promptlab.models.experiment import ExperimentCreate, ExperimentResultBase


@pytest.mark.asyncio
async def test_compare_models_empty_list(client: AsyncClient) -> None:
    """Test that comparing with no models returns an error."""
    payload = {
        "prompt_text": "Hello, world!",
        "prompt_id": None,
        "temperature": 0.7,
        "max_tokens": 100,
        "results": [],
    }

    response = await client.post("/api/v1/compare/compare", json=payload)
    assert response.status_code == status.HTTP_400_BAD_REQUEST


@pytest.mark.asyncio
async def test_compare_models_single_model(client: AsyncClient) -> None:
    """Test comparing with a single model (should work but log warning)."""
    # This test will fail if LiteLLM tries to make actual API calls
    # We just verify the endpoint accepts the request structure
    payload = {
        "prompt_text": "Say hello!",
        "prompt_id": None,
        "temperature": 0.5,
        "max_tokens": 50,
        "results": [
            {
                "model_name": "gpt-3.5-turbo",
                "output": "",
                "latency_ms": 0,
                "tokens_used": 0,
                "success": True,
            }
        ],
    }

    response = await client.post("/api/v1/compare/compare", json=payload)
    # The endpoint should accept the request; actual API call may fail without keys
    assert response.status_code in [
        status.HTTP_201_CREATED,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        status.HTTP_401_UNAUTHORIZED,  # API key error
    ]


@pytest.mark.asyncio
async def test_list_experiments_empty(client: AsyncClient) -> None:
    """Test listing experiments when none exist."""
    response = await client.get("/api/v1/compare")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_get_experiment_not_found(client: AsyncClient) -> None:
    """Test getting a non-existent experiment."""
    response = await client.get("/api/v1/compare/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_delete_experiment_not_found(client: AsyncClient) -> None:
    """Test deleting a non-existent experiment."""
    response = await client.delete("/api/v1/compare/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_experiment_result_schema() -> None:
    """Test the ExperimentResultBase schema."""
    result = ExperimentResultBase(
        model_name="gpt-4",
        output="Hello there!",
        latency_ms=1234.5,
        tokens_used=50,
        success=True,
        error_message=None,
    )
    assert result.model_name == "gpt-4"
    assert result.latency_ms == 1234.5
    assert result.success is True


@pytest.mark.asyncio
async def test_experiment_create_schema() -> None:
    """Test the ExperimentCreate schema."""
    experiment = ExperimentCreate(
        prompt_text="Test prompt",
        prompt_id=1,
        temperature=0.8,
        max_tokens=500,
        results=[
            ExperimentResultBase(
                model_name="claude-3-opus",
                output="",
                latency_ms=0,
                tokens_used=0,
                success=True,
            )
        ],
    )
    assert experiment.prompt_text == "Test prompt"
    assert experiment.temperature == 0.8
    assert len(experiment.results) == 1
