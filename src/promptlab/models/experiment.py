"""Experiment and ExperimentResult database models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from promptlab.database.session import Base


class Experiment(Base):
    """SQLAlchemy model for storing comparison experiments."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    prompt_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("prompts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    max_tokens: Mapped[int] = mapped_column(Integer, default=2048)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False, index=True
    )

    # Relationship to results
    results: Mapped[list["ExperimentResult"]] = relationship(
        "ExperimentResult", back_populates="experiment", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Experiment(id={self.id}, prompt_id={self.prompt_id})>"


class ExperimentResult(Base):
    """SQLAlchemy model for storing individual model results from an experiment."""

    __tablename__ = "experiment_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    experiment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    output: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )

    # Relationship back to experiment
    experiment: Mapped["Experiment"] = relationship(
        "Experiment", back_populates="results"
    )

    def __repr__(self) -> str:
        return f"<ExperimentResult(id={self.id}, model='{self.model_name}')>"


# Pydantic schemas for API responses


class ExperimentResultBase(BaseModel):
    """Base schema for experiment result data."""

    model_name: str = Field(..., description="Name of the LLM model")
    output: str = Field(..., description="Model output text")
    latency_ms: float = Field(..., description="Response latency in milliseconds")
    tokens_used: int = Field(default=0, description="Number of tokens used")
    success: bool = Field(default=True, description="Whether the query succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class ExperimentResultResponse(ExperimentResultBase):
    """Schema for experiment result response with metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    experiment_id: int
    created_at: datetime


class ExperimentBase(BaseModel):
    """Base schema for experiment data."""

    prompt_text: str = Field(..., description="The prompt text used")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature setting")
    max_tokens: int = Field(default=2048, ge=1, le=100000, description="Max tokens setting")


class ExperimentCreate(ExperimentBase):
    """Schema for creating a new experiment."""

    prompt_id: Optional[int] = Field(None, description="Optional linked prompt ID")
    results: list[ExperimentResultBase] = Field(
        default_factory=list, description="List of model results"
    )


class ExperimentResponse(ExperimentBase):
    """Schema for experiment response with metadata and results."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    prompt_id: Optional[int]
    created_at: datetime
    results: list[ExperimentResultResponse] = Field(default_factory=list)


class ExperimentListItem(BaseModel):
    """Simplified experiment info for list view."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    prompt_id: Optional[int]
    prompt_text: str
    temperature: float
    max_tokens: int
    created_at: datetime
    result_count: int
