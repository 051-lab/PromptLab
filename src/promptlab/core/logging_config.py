"""Logging configuration for PromptLab."""

import sys
from typing import Optional

from loguru import logger

from .config import settings


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (default from settings)
        log_format: Log format string (default from settings)
    """
    level = level or settings.log_level
    log_format = log_format or settings.log_format

    # Remove default handler
    logger.remove()

    # Add console handler with custom format
    logger.add(
        sys.stdout,
        format=log_format,
        level=level.upper(),
        colorize=settings.is_development,
        backtrace=True,
        diagnose=settings.debug,
    )

    # Add file handler for production
    if settings.is_production:
        logger.add(
            "logs/promptlab.log",
            format=log_format,
            level=level.upper(),
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=True,
            diagnose=settings.debug,
        )

    logger.info(f"Logging configured with level: {level}")


def get_logger(name: str) -> logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)
