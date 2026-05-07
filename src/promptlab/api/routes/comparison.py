"""Model comparison endpoints for running experiments across multiple LLMs."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from promptlab.database.session import get_db
from promptlab.llm_client import LiteLLMClient, ModelResponse
from promptlab.models.experiment import (
    Experiment,
    ExperimentCreate,
    ExperimentListItem,
    ExperimentResponse,
    ExperimentResult,
    ExperimentResultBase,
    ExperimentResultResponse,
)
from promptlab.models.prompt import Prompt

router = APIRouter()


@router.post("/compare", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def compare_models(
    experiment_data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """
    Run a comparison experiment across multiple LLM models.

    Args:
        experiment_data: Contains prompt text, model list, and generation settings
        db: Database session

    Returns:
        ExperimentResponse with all model outputs and metrics

    Raises:
        HTTPException: If no models specified or invalid configuration
    """
    # Extract unique model names from results
    model_names = list(set(r.model_name for r in experiment_data.results))

    if not model_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one model must be specified for comparison",
        )

    if len(model_names) < 2:
        logger.warning("Comparison with less than 2 models requested")

    logger.info(
        f"Starting comparison experiment with {len(model_names)} models: {model_names}"
    )

    # Create the experiment record
    experiment = Experiment(
        prompt_id=experiment_data.prompt_id,
        prompt_text=experiment_data.prompt_text,
        temperature=experiment_data.temperature,
        max_tokens=experiment_data.max_tokens,
    )
    db.add(experiment)
    await db.flush()  # Get the experiment ID

    # Query all models in parallel using LiteLLM client
    client = LiteLLMClient()
    responses: list[ModelResponse] = await client.query_all_models(
        prompt=experiment_data.prompt_text,
        model_names=model_names,
        temperature=experiment_data.temperature,
        max_tokens=experiment_data.max_tokens,
    )

    # Store results in database
    result_objects: list[ExperimentResult] = []
    for response in responses:
        result = ExperimentResult(
            experiment_id=experiment.id,
            model_name=response.model_name,
            output=response.output,
            latency_ms=response.latency_ms,
            tokens_used=response.tokens_used,
            success=response.success,
            error_message=response.error,
        )
        result_objects.append(result)
        db.add(result)

    await db.commit()
    await db.refresh(experiment)

    # Build response with results
    results_response = [
        ExperimentResultResponse.model_validate(r)
        for r in result_objects
    ]

    logger.info(
        f"Completed comparison experiment {experiment.id}: "
        f"{sum(1 for r in responses if r.success)}/{len(responses)} succeeded"
    )

    return ExperimentResponse(
        id=experiment.id,
        prompt_id=experiment.prompt_id,
        prompt_text=experiment.prompt_text,
        temperature=experiment.temperature,
        max_tokens=experiment.max_tokens,
        created_at=experiment.created_at,
        results=results_response,
    )


@router.get("", response_model=List[ExperimentListItem])
async def list_experiments(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> List[ExperimentListItem]:
    """
    List past comparison experiments.

    Args:
        limit: Maximum number of experiments to return
        offset: Offset for pagination
        db: Database session

    Returns:
        List of experiments with summary information
    """
    query = (
        select(Experiment)
        .order_by(Experiment.created_at.desc())
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(query)
    experiments = result.scalars().all()

    # Build list items with result counts
    items = []
    for exp in experiments:
        # Count results for this experiment
        count_query = select(ExperimentResult).where(
            ExperimentResult.experiment_id == exp.id
        )
        count_result = await db.execute(count_query)
        result_count = len(count_result.scalars().all())

        items.append(
            ExperimentListItem(
                id=exp.id,
                prompt_id=exp.prompt_id,
                prompt_text=exp.prompt_text[:200] + "..." if len(exp.prompt_text) > 200 else exp.prompt_text,
                temperature=exp.temperature,
                max_tokens=exp.max_tokens,
                created_at=exp.created_at,
                result_count=result_count,
            )
        )

    logger.info(f"Retrieved {len(items)} experiments")
    return items


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
) -> ExperimentResponse:
    """
    Get a specific experiment by ID with full results.

    Args:
        experiment_id: Experiment ID
        db: Database session

    Returns:
        Experiment with all results

    Raises:
        HTTPException: If experiment not found
    """
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )

    # Get all results for this experiment
    results_query = select(ExperimentResult).where(
        ExperimentResult.experiment_id == experiment_id
    )
    results_result = await db.execute(results_query)
    results = results_result.scalars().all()

    results_response = [
        ExperimentResultBase.model_validate(r) for r in results
    ]

    return ExperimentResponse(
        id=experiment.id,
        prompt_id=experiment.prompt_id,
        prompt_text=experiment.prompt_text,
        temperature=experiment.temperature,
        max_tokens=experiment.max_tokens,
        created_at=experiment.created_at,
        results=results_response,  # type: ignore[arg-type]
    )


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_experiment(
    experiment_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete an experiment and all its results.

    Args:
        experiment_id: Experiment ID
        db: Database session

    Raises:
        HTTPException: If experiment not found
    """
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment with id {experiment_id} not found",
        )

    await db.delete(experiment)
    await db.commit()
    logger.info(f"Deleted experiment {experiment_id}")
