"""OpenRouter embedding provider for cloud-based embedding."""

from __future__ import annotations

import os
from typing import Any, Sequence

from src.rag_module.embedding.base import EmbeddingInfo, EmbeddingProvider

DEFAULT_MAX_INPUT_TOKENS = 8_000


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenRouter API."""

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        dimension: int = 1536,
        max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    ) -> None:
        if max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        self._model = model
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._dimension = dimension
        self._max_input_tokens = max_input_tokens
        self._client = None
        self._encoding: Any | None = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. "
                "Set it as environment variable or pass api_key parameter."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenRouterEmbeddingProvider. "
                "Install with: pip install openai"
            ) from exc
        self._client = OpenAI(
            api_key=self._api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="openrouter",
            model_id=self._model,
            dimension=self._dimension,
            normalized=True,
        )

    @property
    def runtime_config(self) -> dict[str, object]:
        return {
            "provider": "openrouter",
            "model_id": self._model,
            "max_input_tokens": self._max_input_tokens,
        }

    @property
    def cache_revision(self) -> str:
        return f"2:max-input-tokens={self._max_input_tokens}"

    def embed_query(self, text: str) -> list[float]:
        self._ensure_client()
        response = self._client.embeddings.create(
            model=self._model,
            input=self._truncate(text),
        )
        if not response.data:
            raise ValueError("No embedding data received from OpenRouter")
        embedding = response.data[0].embedding
        if not embedding:
            raise ValueError("Empty embedding received from OpenRouter")
        # Update dimension based on actual response
        self._dimension = len(embedding)
        return embedding

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._ensure_client()
        # OpenRouter supports batch embedding
        response = self._client.embeddings.create(
            model=self._model,
            input=[self._truncate(text) for text in texts],
        )
        if not response.data:
            raise ValueError("No embedding data received from OpenRouter")
        result = [item.embedding for item in response.data]
        if result:
            # Update dimension based on actual response
            self._dimension = len(result[0])
        return result

    def _truncate(self, text: str) -> str:
        encoding = self._ensure_encoding()
        tokens = encoding.encode(text, disallowed_special=())
        if len(tokens) <= self._max_input_tokens:
            return text
        return encoding.decode(tokens[: self._max_input_tokens])

    def _ensure_encoding(self):
        if self._encoding is not None:
            return self._encoding
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "tiktoken is required to enforce OpenRouter embedding input limits"
            ) from exc
        model_name = self._model.rsplit("/", 1)[-1]
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._encoding = tiktoken.get_encoding("cl100k_base")
        return self._encoding
