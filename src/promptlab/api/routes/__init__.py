"""API routes for PromptLab."""

from fastapi import APIRouter

from .health import router as health_router
from .prompts import router as prompts_router

router = APIRouter()
router.include_router(health_router, tags=["Health"])
router.include_router(prompts_router, prefix="/prompts", tags=["Prompts"])
