"""Prompt management endpoints."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from promptlab.core.exceptions import ModelNotFoundError
from promptlab.database.session import get_db
from promptlab.models.prompt import Prompt, PromptCreate, PromptResponse, PromptUpdate

router = APIRouter()


@router.get("", response_model=List[PromptResponse])
async def list_prompts(db: AsyncSession = Depends(get_db)) -> List[PromptResponse]:
    """
    List all prompts.

    Returns:
        List of all prompts
    """
    from sqlalchemy import select

    result = await db.execute(select(Prompt).order_by(Prompt.created_at.desc()))
    prompts = result.scalars().all()
    logger.info(f"Retrieved {len(prompts)} prompts")
    return [PromptResponse.model_validate(p) for p in prompts]


@router.post("", response_model=PromptResponse, status_code=status.HTTP_201_CREATED)
async def create_prompt(
    prompt_data: PromptCreate,
    db: AsyncSession = Depends(get_db),
) -> PromptResponse:
    """
    Create a new prompt.

    Args:
        prompt_data: Prompt creation data

    Returns:
        Created prompt with metadata
    """
    prompt = Prompt(**prompt_data.model_dump())
    db.add(prompt)
    await db.commit()
    await db.refresh(prompt)
    logger.info(f"Created prompt: {prompt.name} (id={prompt.id})")
    return PromptResponse.model_validate(prompt)


@router.get("/{prompt_id}", response_model=PromptResponse)
async def get_prompt(prompt_id: int, db: AsyncSession = Depends(get_db)) -> PromptResponse:
    """
    Get a specific prompt by ID.

    Args:
        prompt_id: Prompt ID

    Returns:
        Requested prompt

    Raises:
        HTTPException: If prompt not found
    """
    from sqlalchemy import select

    result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = result.scalar_one_or_none()

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with id {prompt_id} not found",
        )

    return PromptResponse.model_validate(prompt)


@router.put("/{prompt_id}", response_model=PromptResponse)
async def update_prompt(
    prompt_id: int,
    prompt_data: PromptUpdate,
    db: AsyncSession = Depends(get_db),
) -> PromptResponse:
    """
    Update an existing prompt.

    Args:
        prompt_id: Prompt ID
        prompt_data: Update data

    Returns:
        Updated prompt

    Raises:
        HTTPException: If prompt not found
    """
    from sqlalchemy import select

    result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = result.scalar_one_or_none()

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with id {prompt_id} not found",
        )

    # Update only provided fields
    update_data = prompt_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(prompt, field, value)

    await db.commit()
    await db.refresh(prompt)
    logger.info(f"Updated prompt: {prompt.name} (id={prompt.id})")
    return PromptResponse.model_validate(prompt)


@router.delete("/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(prompt_id: int, db: AsyncSession = Depends(get_db)) -> None:
    """
    Delete a prompt.

    Args:
        prompt_id: Prompt ID

    Raises:
        HTTPException: If prompt not found
    """
    from sqlalchemy import select

    result = await db.execute(select(Prompt).where(Prompt.id == prompt_id))
    prompt = result.scalar_one_or_none()

    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt with id {prompt_id} not found",
        )

    await db.delete(prompt)
    await db.commit()
    logger.info(f"Deleted prompt: {prompt.name} (id={prompt.id})")
