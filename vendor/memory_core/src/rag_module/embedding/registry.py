"""Unified embedding provider registry with support for multiple backends.

Backends:
- local-qwen: Local Qwen3 embedding (requires model download)
- feature-hash: Fast hash-based embedding (no model needed)
- openrouter: Cloud embedding via OpenRouter API
- kylin: Kylin OS native embedding SDK

The deployment default is ``kylin``. Other providers remain available only as
explicit baselines and test fixtures.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from src.rag_module.embedding.base import EmbeddingProvider

DEFAULT_EMBEDDING_PROVIDER = "kylin"


class EmbeddingProviderRegistry:
    """Central registry for all embedding providers."""

    _providers: dict[str, Callable[..., EmbeddingProvider]] = {}
    _cloud_providers: set[str] = {"openrouter"}
    _model_free_providers: set[str] = {"feature-hash", "kylin"}

    @classmethod
    def register(
        cls,
        name: str,
        provider_factory: Callable[..., EmbeddingProvider],
        is_cloud: bool = False,
    ) -> None:
        """Register an embedding provider."""
        cls._providers[name] = provider_factory
        if is_cloud:
            cls._cloud_providers.add(name)

    @classmethod
    def create(cls, name: str, **kwargs) -> EmbeddingProvider:
        """Create an embedding provider by name."""
        if name not in cls._providers:
            available = ", ".join(sorted(cls._providers.keys()))
            raise ValueError(f"Unknown embedding provider: {name}. Available: {available}")
        return cls._providers[name](**kwargs)

    @classmethod
    def is_cloud_provider(cls, name: str) -> bool:
        """Check if provider requires cloud API (no local model)."""
        return name in cls._cloud_providers

    @classmethod
    def requires_local_model(cls, name: str) -> bool:
        """Check if provider requires local model files."""
        return not cls.is_cloud_provider(name) and name not in cls._model_free_providers

    @classmethod
    def available_providers(cls) -> list[str]:
        """List all registered providers."""
        return sorted(cls._providers.keys())


def _register_default_providers() -> None:
    """Register all built-in providers."""
    def _lazy_feature_hash(**kwargs):
        from src.memory_module.retriever.sqlite import FeatureHashEmbeddingProvider

        return FeatureHashEmbeddingProvider(**kwargs)

    def _lazy_memory_qwen(**kwargs):
        from src.memory_module.schema.embedding import MemoryQwenEmbeddingProvider

        return MemoryQwenEmbeddingProvider(**kwargs)

    def _lazy_rag_qwen(**kwargs):
        from src.rag_module.embedding.qwen_provider import QwenEmbeddingProvider

        return QwenEmbeddingProvider(**kwargs)

    def _lazy_kylin(**kwargs):
        from src.rag_module.embedding.kylin_provider import KylinEmbeddingProvider

        return KylinEmbeddingProvider(**kwargs)

    EmbeddingProviderRegistry.register("feature-hash", _lazy_feature_hash)
    EmbeddingProviderRegistry.register("local-qwen", _lazy_memory_qwen)
    EmbeddingProviderRegistry.register("rag-qwen", _lazy_rag_qwen)
    EmbeddingProviderRegistry.register("kylin", _lazy_kylin)

    # Lazy register cloud providers to avoid import errors
    def _lazy_openrouter(**kwargs):
        from src.rag_module.embedding.openrouter_provider import OpenRouterEmbeddingProvider
        return OpenRouterEmbeddingProvider(**kwargs)

    EmbeddingProviderRegistry._providers["openrouter"] = _lazy_openrouter
    EmbeddingProviderRegistry._cloud_providers.add("openrouter")


# Initialize on module load
_register_default_providers()


@dataclass
class EmbeddingProviderConfig:
    """Configuration for creating an embedding provider."""

    name: str
    device: str | None = None
    model_path: str | None = None
    api_key: str | None = None
    batch_size: int = 32
    cache_path: str | None = None

    @classmethod
    def from_env(cls, name: str | None = None) -> EmbeddingProviderConfig:
        """Create config from environment variables."""
        return cls(
            name=name or os.environ.get("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER),
            device=os.environ.get("EMBEDDING_DEVICE"),
            model_path=os.environ.get("EMBEDDING_MODEL_PATH"),
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
            cache_path=os.environ.get("EMBEDDING_CACHE_PATH"),
        )


def create_embedding_provider(config: EmbeddingProviderConfig) -> EmbeddingProvider:
    """Create an embedding provider from configuration."""
    kwargs = {}

    if config.name == "openrouter":
        if config.api_key:
            kwargs["api_key"] = config.api_key
    elif config.name in ("local-qwen", "rag-qwen"):
        if config.model_path:
            kwargs["model_id"] = config.model_path
        if config.device:
            kwargs["device"] = config.device
        if config.name == "rag-qwen":
            kwargs["batch_size"] = config.batch_size
            if config.cache_path:
                kwargs["cache_path"] = config.cache_path

    return EmbeddingProviderRegistry.create(config.name, **kwargs)


def create_default_embedding_provider() -> EmbeddingProvider:
    """Create the deployment embedding provider (Kylin unless explicitly overridden)."""

    return create_embedding_provider(EmbeddingProviderConfig.from_env())
