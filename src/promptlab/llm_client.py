"""LiteLLM client for querying multiple LLM models."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional

import litellm
from loguru import logger

from promptlab.core.config import settings


@dataclass
class ModelResponse:
    """Response from a single model query."""

    model_name: str
    output: str
    latency_ms: float
    tokens_used: int
    success: bool
    error: Optional[str] = None


class LiteLLMClient:
    """Async client for querying LLM models via LiteLLM."""

    def __init__(self):
        """Initialize the LiteLLM client with API keys from settings."""
        # Configure API keys from environment
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key
        if settings.anthropic_api_key:
            litellm.anthropic_key = settings.anthropic_api_key
        if settings.google_api_key:
            litellm.google_key = settings.google_api_key
        if settings.qwen_api_key:
            litellm.api_key = settings.qwen_api_key  # Qwen uses generic api_key

        self.default_temperature = settings.litellm_temperature
        self.default_max_tokens = settings.litellm_max_tokens
        self.default_timeout = settings.litellm_timeout

    async def query_model(
        self,
        prompt: str,
        model_name: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> ModelResponse:
        """
        Query a single LLM model asynchronously.

        Args:
            prompt: The prompt text to send
            model_name: Name of the model (e.g., 'gpt-4', 'claude-3-opus')
            temperature: Temperature for generation (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds

        Returns:
            ModelResponse with output, latency, and token usage
        """
        start_time = time.perf_counter()

        try:
            # Run LiteLLM completion in executor since it's not fully async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: litellm.completion(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature or self.default_temperature,
                    max_tokens=max_tokens or self.default_max_tokens,
                    timeout=timeout or self.default_timeout,
                ),
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            # Extract output and token usage
            output = response.choices[0].message.content or ""
            tokens_used = response.usage.total_tokens if response.usage else 0

            logger.info(
                f"Query completed for {model_name}: {latency_ms:.2f}ms, {tokens_used} tokens"
            )

            return ModelResponse(
                model_name=model_name,
                output=output,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                success=True,
            )

        except Exception as e:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            logger.error(f"Query failed for {model_name}: {str(e)}")

            return ModelResponse(
                model_name=model_name,
                output="",
                latency_ms=latency_ms,
                tokens_used=0,
                success=False,
                error=str(e),
            )

    async def query_all_models(
        self,
        prompt: str,
        model_names: list[str],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> list[ModelResponse]:
        """
        Query multiple LLM models in parallel using asyncio.gather().

        Args:
            prompt: The prompt text to send
            model_names: List of model names to query
            temperature: Temperature for generation (0.0-2.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds

        Returns:
            List of ModelResponse objects, one per model
        """
        tasks = [
            self.query_model(
                prompt=prompt,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            for model in model_names
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=False)

        successful = sum(1 for r in responses if r.success)
        logger.info(
            f"Batch query completed: {successful}/{len(model_names)} models succeeded"
        )

        return list(responses)

    @staticmethod
    def get_available_models() -> dict[str, str]:
        """
        Get a dictionary of available model identifiers and their display names.

        Returns:
            Dict mapping model identifier to human-readable name
        """
        return {
            "gpt-4o": "GPT-4o (OpenAI)",
            "gpt-4-turbo": "GPT-4 Turbo (OpenAI)",
            "gpt-3.5-turbo": "GPT-3.5 Turbo (OpenAI)",
            "claude-3-opus-20240229": "Claude 3 Opus (Anthropic)",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet (Anthropic)",
            "claude-3-haiku-20240307": "Claude 3 Haiku (Anthropic)",
            "gemini/gemini-pro": "Gemini Pro (Google)",
            "gemini/gemini-1.5-pro": "Gemini 1.5 Pro (Google)",
            "qwen/qwen-max": "Qwen Max (Alibaba)",
            "qwen/qwen-plus": "Qwen Plus (Alibaba)",
        }
