"""Exception handlers for PromptLab API."""

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import ValidationError


class PromptLabException(Exception):
    """Base exception for PromptLab."""

    def __init__(
        self,
        message: str = "An error occurred",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.detail = detail or message
        super().__init__(self.message)


class ModelNotFoundError(PromptLabException):
    """Raised when a model is not found."""

    def __init__(self, model_name: str):
        super().__init__(
            message=f"Model '{model_name}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class PromptExecutionError(PromptLabException):
    """Raised when prompt execution fails."""

    def __init__(self, message: str = "Failed to execute prompt"):
        super().__init__(
            message=message,
            status_code=status.HTTP_502_BAD_GATEWAY,
        )


async def promptlab_exception_handler(
    request: Request, exc: PromptLabException
) -> JSONResponse:
    """Handle custom PromptLab exceptions."""
    logger.error(f"{exc.__class__.__name__}: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "message": exc.message},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors."""
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Request validation failed",
            "details": exc.errors(),
        },
    )


async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle uncaught exceptions."""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(PromptLabException, promptlab_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
