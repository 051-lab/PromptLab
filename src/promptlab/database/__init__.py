"""Database module for PromptLab."""

from .session import get_db, init_db

__all__ = ["get_db", "init_db"]
