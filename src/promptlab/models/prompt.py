"""Prompt database models and Pydantic schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from promptlab.database.session import Base


class Prompt(Base):
    """SQLAlchemy model for storing prompts."""

    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, default="gpt-3.5-turbo")
    temperature: Mapped[float] = mapped_column(default=0.7)
    max_tokens: Mapped[int] = mapped_column(default=2048)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Prompt(id={self.id}, name='{self.name}')>"


class PromptBase(BaseModel):
    """Base schema for prompt data."""

    name: str = Field(..., min_length=1, max_length=255, description="Name of the prompt")
    description: Optional[str] = Field(None, description="Optional description")
    template: str = Field(..., min_length=1, description="Prompt template text")
    model_name: str = Field(default="gpt-3.5-turbo", description="LLM model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(default=2048, ge=1, le=100000, description="Max tokens to generate")


class PromptCreate(PromptBase):
    """Schema for creating a new prompt."""

    pass


class PromptUpdate(BaseModel):
    """Schema for updating an existing prompt."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    template: Optional[str] = Field(None, min_length=1)
    model_name: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=100000)


class PromptResponse(PromptBase):
    """Schema for prompt response with metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime
