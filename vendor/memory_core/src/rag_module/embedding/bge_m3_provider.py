"""BGE-M3 embedding provider using FlagEmbedding."""

from __future__ import annotations

from typing import Sequence

from src.rag_module.embedding.base import EmbeddingInfo, EmbeddingProvider

DEFAULT_MODEL_ID = "BAAI/bge-m3"
DEFAULT_DIMENSION = 1024


class BGEM3EmbeddingProvider(EmbeddingProvider):
    """Embedding provider using BAAI/bge-m3 via FlagEmbedding."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        use_fp16: bool = True,
        normalize: bool = True,
    ) -> None:
        self._model_id = model_id
        self._use_fp16 = use_fp16
        self._normalize = normalize
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required for BGEM3EmbeddingProvider. "
                "Install with: pip install FlagEmbedding"
            ) from exc
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = BGEM3FlagModel(
            self._model_id,
            use_fp16=self._use_fp16 and device == "cuda",
            device=device,
        )

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="bge-m3",
            model_id=self._model_id,
            dimension=DEFAULT_DIMENSION,
            normalized=self._normalize,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None

        output = self._model.encode(
            texts,
            batch_size=32,
            max_length=512,
        )
        embeddings = output["dense_vecs"]

        import numpy as np

        if self._normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)

        return embeddings.tolist()
