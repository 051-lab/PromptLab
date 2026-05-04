"""Tests for prompt endpoints."""

import pytest
from fastapi import status
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_prompts_empty(client: AsyncClient) -> None:
    """Test listing prompts when database is empty."""
    response = await client.get("/api/v1/prompts")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@pytest.mark.asyncio
async def test_create_prompt(client: AsyncClient) -> None:
    """Test creating a new prompt."""
    prompt_data = {
        "name": "Test Prompt",
        "description": "A test prompt",
        "template": "You are a helpful assistant. {{question}}",
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    response = await client.post("/api/v1/prompts", json=prompt_data)

    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == prompt_data["name"]
    assert data["description"] == prompt_data["description"]
    assert data["template"] == prompt_data["template"]
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


@pytest.mark.asyncio
async def test_get_prompt(client: AsyncClient) -> None:
    """Test getting a specific prompt."""
    # First create a prompt
    prompt_data = {
        "name": "Get Test Prompt",
        "template": "Test template",
    }
    create_response = await client.post("/api/v1/prompts", json=prompt_data)
    prompt_id = create_response.json()["id"]

    # Then get it
    response = await client.get(f"/api/v1/prompts/{prompt_id}")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["id"] == prompt_id
    assert data["name"] == prompt_data["name"]


@pytest.mark.asyncio
async def test_get_prompt_not_found(client: AsyncClient) -> None:
    """Test getting a non-existent prompt."""
    response = await client.get("/api/v1/prompts/99999")

    assert response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_update_prompt(client: AsyncClient) -> None:
    """Test updating a prompt."""
    # First create a prompt
    prompt_data = {
        "name": "Update Test Prompt",
        "template": "Original template",
    }
    create_response = await client.post("/api/v1/prompts", json=prompt_data)
    prompt_id = create_response.json()["id"]

    # Then update it
    update_data = {
        "name": "Updated Prompt Name",
        "temperature": 0.9,
    }
    response = await client.put(f"/api/v1/prompts/{prompt_id}", json=update_data)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["temperature"] == update_data["temperature"]


@pytest.mark.asyncio
async def test_delete_prompt(client: AsyncClient) -> None:
    """Test deleting a prompt."""
    # First create a prompt
    prompt_data = {
        "name": "Delete Test Prompt",
        "template": "To be deleted",
    }
    create_response = await client.post("/api/v1/prompts", json=prompt_data)
    prompt_id = create_response.json()["id"]

    # Then delete it
    response = await client.delete(f"/api/v1/prompts/{prompt_id}")

    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's deleted
    get_response = await client.get(f"/api/v1/prompts/{prompt_id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_validation_error(client: AsyncClient) -> None:
    """Test validation error on invalid data."""
    invalid_data = {
        "name": "",  # Empty name should fail validation
        "template": "Some template",
    }

    response = await client.post("/api/v1/prompts", json=invalid_data)

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
