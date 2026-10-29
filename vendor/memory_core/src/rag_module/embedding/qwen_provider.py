"""Qwen3-Embedding-0.6B embedding provider."""

from __future__ import annotations

import gc
import hashlib
import sqlite3
import struct
import time
from pathlib import Path
from typing import Sequence

from src.rag_module.embedding.base import EmbeddingInfo, EmbeddingProvider

DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_DIMENSION = 1024
DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
EMBEDDING_REVISION = "last-token-v1"
DEFAULT_BATCH_SIZE = 8
DEFAULT_CACHE_PATH = Path("data/cache/rag_embeddings.sqlite3")


class QwenEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Qwen3-Embedding-0.6B via transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        normalize: bool = True,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        max_length: int = 8192,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._model_id = model_id
        self._normalize = normalize
        self._device = device
        self._query_instruction = query_instruction
        self._max_length = max_length
        self._batch_size = batch_size
        self._cache_path = Path(cache_path) if cache_path is not None else None
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
                "transformers is required for QwenEmbeddingProvider. "
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
            provider="qwen",
            model_id=f"{self._model_id}:{EMBEDDING_REVISION}",
            dimension=DEFAULT_DIMENSION,
            normalized=self._normalize,
        )

    def embed_query(self, text: str) -> list[float]:
        query = f"Instruct: {self._query_instruction}\nQuery:{text}"
        return self._embed_cached([query], namespace="query")[0]

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_cached(
            [f"Instruct: {self._query_instruction}\nQuery:{text}" for text in texts],
            namespace="query",
        )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_cached(list(texts), namespace="document")

    @property
    def cache_revision(self) -> str:
        if self._cache_revision_value is not None:
            return self._cache_revision_value
        digest = hashlib.sha256()
        digest.update(self._query_instruction.encode("utf-8"))
        digest.update(str(self._max_length).encode("ascii"))
        digest.update(str(self._normalize).encode("ascii"))
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
        self._cache_revision_value = f"{EMBEDDING_REVISION}:{digest.hexdigest()[:20]}"
        return self._cache_revision_value

    def release(self) -> None:
        """Drop lazy-loaded weights and return cached CUDA blocks to PyTorch."""

        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _embed_cached(self, texts: list[str], *, namespace: str) -> list[list[float]]:
        if not texts:
            return []
        if self._cache_path is None:
            return self._embed_batched(texts)

        keys = [self._cache_key(text, namespace) for text in texts]
        cached = self._read_cache(keys)
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        seen_missing: set[str] = set()
        for key, text in zip(keys, texts):
            if key not in cached and key not in seen_missing:
                seen_missing.add(key)
                missing_keys.append(key)
                missing_texts.append(text)
        if missing_texts:
            vectors = self._embed_batched(missing_texts)
            self._write_cache(list(zip(missing_keys, vectors)))
            cached.update(zip(missing_keys, vectors))
        return [cached[key] for key in keys]

    def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        result: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(order), self._batch_size):
            indices = order[start : start + self._batch_size]
            vectors = self._embed_batch_with_oom_retry([texts[index] for index in indices])
            for index, vector in zip(indices, vectors):
                result[index] = vector
        if any(vector is None for vector in result):
            raise RuntimeError("embedding batch did not produce every vector")
        return [vector for vector in result if vector is not None]

    def _embed_batch_with_oom_retry(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._embed(texts)
        except Exception as exc:
            if not self._is_cuda_oom(exc) or len(texts) <= 1:
                raise
            self._clear_cuda_cache()
            midpoint = max(1, len(texts) // 2)
            return self._embed_batch_with_oom_retry(
                texts[:midpoint]
            ) + self._embed_batch_with_oom_retry(texts[midpoint:])

    def _embed(self, texts: list[str]) -> list[list[float]]:
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
            embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
        return embeddings.float().cpu().numpy().tolist()

    def _cache_key(self, text: str, namespace: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.cache_revision.encode("utf-8"))
        digest.update(namespace.encode("ascii"))
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _open_cache(self) -> sqlite3.Connection:
        assert self._cache_path is not None
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._cache_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_embedding_cache (
                cache_key TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL,
                created_at INTEGER NOT NULL
            )
            """
        )
        return conn

    def _read_cache(self, keys: list[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        conn = self._open_cache()
        try:
            result: dict[str, list[float]] = {}
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT cache_key, dimension, vector FROM rag_embedding_cache "
                    f"WHERE cache_key IN ({placeholders})",
                    batch,
                ).fetchall()
                for key, dimension, blob in rows:
                    if int(dimension) != self.info.dimension:
                        continue
                    result[str(key)] = list(struct.unpack(f"<{dimension}f", blob))
            return result
        finally:
            conn.close()

    def _write_cache(self, values: list[tuple[str, list[float]]]) -> None:
        if not values:
            return
        conn = self._open_cache()
        try:
            conn.executemany(
                """
                INSERT OR REPLACE INTO rag_embedding_cache
                    (cache_key, dimension, vector, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        key,
                        len(vector),
                        struct.pack(f"<{len(vector)}f", *vector),
                        int(time.time()),
                    )
                    for key, vector in values
                ],
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        try:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        return "out of memory" in str(exc).casefold() and "cuda" in str(exc).casefold()

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _last_token_pool(last_hidden_states, attention_mask):
        left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        sequence_lengths = attention_mask.sum(dim=1) - 1
        return last_hidden_states[range(last_hidden_states.shape[0]), sequence_lengths]
