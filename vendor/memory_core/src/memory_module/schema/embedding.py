"""Memory-specific Qwen embedding behavior isolated from the RAG provider."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Sequence

from src.rag_module.embedding.base import EmbeddingInfo, EmbeddingProvider

DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DIMENSION = 1024
DEFAULT_QUERY_INSTRUCTION = "Given a user query, retrieve relevant passages that answer the query"
MEMORY_EMBEDDING_REVISION = "last-token-v1"


class MemoryQwenEmbeddingProvider(EmbeddingProvider):
    """Qwen provider configured specifically for semantic memory retrieval."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        normalize: bool = True,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        max_length: int = 8192,
    ) -> None:
        self._model_id = model_id
        self._normalize = normalize
        self._device = device
        self._query_instruction = query_instruction
        self._max_length = max_length
        self._model = None
        self._tokenizer = None
        self._cache_revision_value = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for MemoryQwenEmbeddingProvider. "
                "Install with: pip install transformers torch"
            ) from exc
        import torch

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            padding_side="left",
            trust_remote_code=True,
        )
        model_kwargs = {"trust_remote_code": True}
        if self._device.startswith("cuda"):
            model_kwargs["dtype"] = torch.float16
        self._model = AutoModel.from_pretrained(self._model_id, **model_kwargs)
        self._model = self._model.to(self._device)
        self._model.eval()

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="memory-qwen",
            model_id=f"{self._model_id}:{MEMORY_EMBEDDING_REVISION}",
            dimension=DEFAULT_DIMENSION,
            normalized=self._normalize,
        )

    def embed_query(self, text: str) -> list[float]:
        query = f"Instruct: {self._query_instruction}\nQuery:{text}"
        return self._embed([query])[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([f"Instruct: {self._query_instruction}\nQuery:{text}" for text in texts])

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(list(texts))

    @property
    def cache_revision(self) -> str:
        if self._cache_revision_value is not None:
            return self._cache_revision_value
        digest = hashlib.sha256()
        digest.update(self._query_instruction.encode("utf-8"))
        digest.update(str(self._max_length).encode("ascii"))
        model_path = Path(self._model_id)
        if model_path.is_dir():
            for path in sorted(model_path.iterdir()):
                if not path.is_file() or not (
                    path.name == "config.json" or path.suffix in {".safetensors", ".bin"}
                ):
                    continue
                stat = path.stat()
                digest.update(path.name.encode("utf-8"))
                digest.update(str(stat.st_size).encode("ascii"))
                digest.update(str(stat.st_mtime_ns).encode("ascii"))
                if path.name == "config.json":
                    digest.update(path.read_bytes())
        else:
            digest.update(self._model_id.encode("utf-8"))
        self._cache_revision_value = f"{MEMORY_EMBEDDING_REVISION}:{digest.hexdigest()[:20]}"
        return self._cache_revision_value

    def release(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        import torch

        with torch.inference_mode():
            outputs = self._model(**encoded)

        embeddings = self._last_token_pool(
            outputs.last_hidden_state,
            encoded["attention_mask"],
        )
        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.float().cpu().numpy().tolist()

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[range(last_hidden_states.shape[0]), sequence_lengths]
