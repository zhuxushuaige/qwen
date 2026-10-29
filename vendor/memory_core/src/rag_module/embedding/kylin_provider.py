"""Python adapter for the Kylin CoreAI text-embedding C SDK.

The official SDK is exposed as a C ABI by ``libkylin-coreai-embedding``.
This module keeps that platform dependency behind the project's existing
``EmbeddingProvider`` interface so Memory, RAG, and evaluation code can inject
the provider without importing or linking Kylin headers themselves.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import importlib.util
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Protocol, Sequence

from src.rag_module.embedding.base import EmbeddingInfo, EmbeddingProvider

DEFAULT_KYLIN_DIMENSION = 768
DEFAULT_KYLIN_MODEL_ID = "kylin-system-default"

# ``KYLIN_EMBEDDING_SDK_PATH`` existed in the first project stub. Keep it as a
# compatibility alias, but prefer the accurately named library variable.
LIBRARY_ENV_VAR = "KYLIN_EMBEDDING_LIBRARY"
SDK_PATH_ENV_VAR = "KYLIN_EMBEDDING_SDK_PATH"
DIMENSION_ENV_VAR = "KYLIN_EMBEDDING_DIMENSION"
MODEL_ID_ENV_VAR = "KYLIN_EMBEDDING_MODEL_ID"
SESSION_MAX_TEXTS_ENV_VAR = "KYLIN_EMBEDDING_SESSION_MAX_TEXTS"
SESSION_MAX_TOKENS_ENV_VAR = "KYLIN_EMBEDDING_SESSION_MAX_TOKENS"
MAX_INPUT_TOKENS_ENV_VAR = "KYLIN_EMBEDDING_MAX_INPUT_TOKENS"
TOKENIZER_PATH_ENV_VAR = "KYLIN_EMBEDDING_TOKENIZER_PATH"
BATCH_WORKERS_ENV_VAR = "KYLIN_EMBEDDING_BATCH_WORKERS"
DEFAULT_SESSION_MAX_TEXTS = 128
DEFAULT_SESSION_MAX_TOKENS = 8_192
DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_TOKENIZER_PATH = Path(
    "/usr/share/kylin-ai/model-repository/tokenizer_gte-base_uint8-text/tokenizer.json"
)

_LIBRARY_BASENAMES = (
    "kysdk-coreai-embedding",
    "kylin-coreai-embedding",
)
_LIBRARY_FILENAMES = tuple(f"lib{name}.so" for name in _LIBRARY_BASENAMES)


class KylinEmbeddingError(RuntimeError):
    """Raised when the Kylin SDK cannot be loaded or returns an error."""


class KylinEmbeddingBackend(Protocol):
    """Small internal seam used by the C adapter and isolated unit tests."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts with the platform runtime."""

    def model_info(self) -> str | None:
        """Return the SDK model description when available."""

    def close(self) -> None:
        """Release the underlying SDK session."""


class KylinEmbeddingProvider(EmbeddingProvider):
    """Use the Kylin CoreAI SDK through the project embedding protocol.

    The SDK session is initialized lazily so importing the project remains safe
    on Windows and non-Kylin Linux hosts. On a Kylin host the shared library is
    discovered with ``ldconfig``/the dynamic loader, or can be supplied through
    ``KYLIN_EMBEDDING_LIBRARY``.

    The Kylin model is selected by the operating-system AI subsystem rather
    than this C API. ``model_id`` is therefore an index/cache identity supplied
    by the deployer; it must be changed whenever the configured system model
    changes.
    """

    def __init__(
        self,
        model_id: str | None = None,
        dimension: int | None = None,
        *,
        library_path: str | os.PathLike[str] | None = None,
        normalize: bool = True,
        backend: KylinEmbeddingBackend | None = None,
        session_max_texts: int | None = None,
        session_max_tokens: int | None = None,
        max_input_tokens: int | None = None,
        tokenizer_path: str | os.PathLike[str] | None = None,
        batch_workers: int | None = None,
    ) -> None:
        resolved_dimension = dimension
        if resolved_dimension is None:
            resolved_dimension = _positive_int_from_env(
                DIMENSION_ENV_VAR,
                DEFAULT_KYLIN_DIMENSION,
            )
        if resolved_dimension <= 0:
            raise ValueError("Kylin embedding dimension must be positive")

        self._model_id = (
            model_id or os.environ.get(MODEL_ID_ENV_VAR, "").strip() or DEFAULT_KYLIN_MODEL_ID
        )
        self._dimension = resolved_dimension
        self._normalize = normalize
        self._library_path = _configured_library_path(library_path)
        self._backend = backend
        self._backend_is_injected = backend is not None
        self._session_max_texts = (
            _non_negative_int_from_env(
                SESSION_MAX_TEXTS_ENV_VAR,
                DEFAULT_SESSION_MAX_TEXTS,
            )
            if session_max_texts is None
            else session_max_texts
        )
        if self._session_max_texts < 0:
            raise ValueError("Kylin embedding session_max_texts cannot be negative")
        self._session_max_tokens = (
            _non_negative_int_from_env(
                SESSION_MAX_TOKENS_ENV_VAR,
                DEFAULT_SESSION_MAX_TOKENS,
            )
            if session_max_tokens is None
            else session_max_tokens
        )
        if self._session_max_tokens < 0:
            raise ValueError("Kylin embedding session_max_tokens cannot be negative")
        self._max_input_tokens = (
            _positive_int_from_env(
                MAX_INPUT_TOKENS_ENV_VAR,
                DEFAULT_MAX_INPUT_TOKENS,
            )
            if max_input_tokens is None
            else max_input_tokens
        )
        if self._max_input_tokens <= 0:
            raise ValueError("Kylin embedding max_input_tokens must be positive")
        configured_tokenizer = (
            os.fspath(tokenizer_path)
            if tokenizer_path is not None
            else os.environ.get(TOKENIZER_PATH_ENV_VAR, "").strip()
        )
        self._tokenizer_path = Path(configured_tokenizer or DEFAULT_TOKENIZER_PATH)
        self._tokenizer: Any | None = None
        self._texts_in_session = 0
        self._tokens_in_session = 0
        self._lock = threading.RLock()
        self._closed = False
        self._batch_workers = (
            _positive_int_from_env(BATCH_WORKERS_ENV_VAR, 1)
            if batch_workers is None else batch_workers
        )
        if not 1 <= self._batch_workers <= 4:
            raise ValueError("Kylin embedding batch_workers must be between 1 and 4")
        if backend is not None and self._batch_workers != 1:
            raise ValueError("injected embedding backends require batch_workers=1")
        self._batch_pool: ThreadPoolExecutor | None = None
        self._batch_providers: list[KylinEmbeddingProvider] = []

    @property
    def info(self) -> EmbeddingInfo:
        return EmbeddingInfo(
            provider="kylin",
            model_id=self._model_id,
            dimension=self._dimension,
            normalized=self._normalize,
        )

    @property
    def cache_revision(self) -> str:
        normalization = "l2" if self._normalize else "raw"
        try:
            tokenizer_stat = self._tokenizer_path.stat()
            tokenizer_revision = (
                f"{self._tokenizer_path.resolve()}:{tokenizer_stat.st_size}:"
                f"{tokenizer_stat.st_mtime_ns}"
            )
        except OSError:
            tokenizer_revision = f"missing:{self._tokenizer_path}"
        return (
            f"coreai-c-abi-v2:{normalization}:max-input-tokens={self._max_input_tokens}:"
            f"tokenizer={tokenizer_revision}"
        )

    @property
    def runtime_config(self) -> dict[str, Any]:
        return {
            "session_max_texts": self._session_max_texts,
            "session_max_tokens": self._session_max_tokens,
            "max_input_tokens": self._max_input_tokens,
            "tokenizer_path": str(self._tokenizer_path),
            "session_recycling": bool(
                (self._session_max_texts or self._session_max_tokens)
                and not self._backend_is_injected
            ),
        }

    @property
    def available(self) -> bool:
        """Return whether a usable SDK backend can be initialized."""

        try:
            self._ensure_backend()
        except KylinEmbeddingError:
            return False
        return True

    @property
    def sdk_model_info(self) -> str | None:
        """Return the model information reported by the Kylin runtime."""

        with self._lock:
            return self._ensure_backend().model_info()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    @property
    def execution_config(self) -> dict[str, Any]:
        # Scheduling is not vector identity: validated equivalent outputs must
        # remain compatible with existing embedding caches and checkpoints.
        return {"batch_workers": self._batch_workers, "independent_sdk_sessions": True}

    def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        for text in values:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Kylin embedding input must be a non-empty string")

        if self._batch_workers > 1 and len(values) > 1:
            return self._embed_parallel(values)
        return self._embed_serial(values)

    def _embed_parallel(self, values: list[str]) -> list[list[float]]:
        with self._lock:
            if self._closed:
                raise KylinEmbeddingError("Kylin embedding provider has been released")
            if self._batch_pool is None:
                self._batch_providers = [
                    KylinEmbeddingProvider(
                        model_id=self._model_id, dimension=self._dimension,
                        library_path=self._library_path, normalize=self._normalize,
                        session_max_texts=self._session_max_texts,
                        session_max_tokens=self._session_max_tokens,
                        max_input_tokens=self._max_input_tokens,
                        tokenizer_path=self._tokenizer_path, batch_workers=1,
                    ) for _ in range(self._batch_workers)
                ]
                self._batch_pool = ThreadPoolExecutor(
                    max_workers=self._batch_workers, thread_name_prefix="kylin-embedding",
                )
            futures = [
                self._batch_pool.submit(provider.embed_documents, values[i::self._batch_workers])
                for i, provider in enumerate(self._batch_providers)
            ]
            wait(futures)
            results: list[list[float]] = [[] for _ in values]
            for i, future in enumerate(futures):
                vectors = future.result()
                positions = list(range(i, len(values), self._batch_workers))
                if len(vectors) != len(positions):
                    raise KylinEmbeddingError("parallel embedding result count mismatch")
                for position, vector in zip(positions, vectors):
                    results[position] = vector
            return results

    def _embed_serial(self, values: list[str]) -> list[list[float]]:
        prepared = [self._prepare_text(text) for text in values]
        with self._lock:
            if self._closed:
                raise KylinEmbeddingError("Kylin embedding provider has been released")
            if not self._session_recycling_enabled:
                vectors = self._ensure_backend().embed_documents(
                    [text for text, _token_count in prepared]
                )
            else:
                vectors = []
                for text, token_count in prepared:
                    if self._should_recycle_before(token_count):
                        self._recycle_session_locked()
                    vectors.extend(self._ensure_backend().embed_documents([text]))
                    self._texts_in_session += 1
                    self._tokens_in_session += token_count
                    if self._session_limit_reached:
                        self._recycle_session_locked()

        if len(vectors) != len(values):
            raise KylinEmbeddingError(
                "Kylin SDK returned a different number of vectors than inputs"
            )
        return [self._prepare_vector(vector) for vector in vectors]

    def release(self) -> None:
        """Release the SDK session; safe to call more than once."""

        with self._lock:
            if self._closed:
                return
            if self._batch_pool is not None:
                self._batch_pool.shutdown(wait=True)
                self._batch_pool = None
            for provider in self._batch_providers:
                provider.release()
            self._batch_providers = []
            if self._backend is not None:
                self._backend.close()
                self._backend = None
            self._texts_in_session = 0
            self._tokens_in_session = 0
            self._closed = True

    close = release

    def __enter__(self) -> KylinEmbeddingProvider:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.release()

    def _ensure_backend(self) -> KylinEmbeddingBackend:
        if self._closed:
            raise KylinEmbeddingError("Kylin embedding provider has been released")
        if self._backend is not None:
            return self._backend

        try:
            if self._library_path and self._library_path.lower().endswith(".py"):
                self._backend = _PythonShimBackend(
                    self._library_path,
                    model_id=self._model_id,
                )
            else:
                self._backend = _CtypesKylinEmbeddingBackend(self._library_path)
        except KylinEmbeddingError:
            raise
        except Exception as exc:  # pragma: no cover - defensive platform boundary
            raise KylinEmbeddingError(f"Failed to initialize Kylin embedding SDK: {exc}") from exc
        return self._backend

    @property
    def _session_recycling_enabled(self) -> bool:
        return bool(
            (self._session_max_texts or self._session_max_tokens) and not self._backend_is_injected
        )

    @property
    def _session_limit_reached(self) -> bool:
        return bool(
            (self._session_max_texts and self._texts_in_session >= self._session_max_texts)
            or (self._session_max_tokens and self._tokens_in_session >= self._session_max_tokens)
        )

    def _should_recycle_before(self, token_count: int) -> bool:
        if self._texts_in_session == 0:
            return False
        return bool(
            (self._session_max_texts and self._texts_in_session >= self._session_max_texts)
            or (
                self._session_max_tokens
                and self._tokens_in_session + token_count > self._session_max_tokens
            )
        )

    def _recycle_session_locked(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        self._texts_in_session = 0
        self._tokens_in_session = 0

    def _prepare_text(self, text: str) -> tuple[str, int]:
        # Short inputs cannot exceed the token limit by more than a tiny number
        # of special tokens. Count characters conservatively without loading the
        # tokenizer on ordinary online queries.
        if len(text) <= self._max_input_tokens - 2:
            return text, max(1, len(text) + 2)

        tokenizer = self._ensure_tokenizer()
        encoding = tokenizer.encode(text)
        token_ids = encoding.ids
        if len(token_ids) <= self._max_input_tokens:
            return text, max(1, len(token_ids))
        truncated = tokenizer.decode(
            token_ids[: self._max_input_tokens],
            skip_special_tokens=True,
        ).strip()
        if not truncated:
            raise KylinEmbeddingError("Kylin tokenizer produced an empty truncated input")
        return truncated, self._max_input_tokens

    def _ensure_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if not self._tokenizer_path.is_file():
            raise KylinEmbeddingError(
                f"Kylin embedding tokenizer not found: {self._tokenizer_path}. "
                f"Set {TOKENIZER_PATH_ENV_VAR} to the system tokenizer.json path."
            )
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise KylinEmbeddingError(
                "tokenizers is required to enforce the Kylin embedding input limit"
            ) from exc
        self._tokenizer = Tokenizer.from_file(str(self._tokenizer_path))
        return self._tokenizer

    def _prepare_vector(self, vector: Sequence[float]) -> list[float]:
        values = [float(value) for value in vector]
        if len(values) != self._dimension:
            raise KylinEmbeddingError(
                "Kylin embedding dimension mismatch: "
                f"configured={self._dimension}, actual={len(values)}. "
                f"Set {DIMENSION_ENV_VAR} to the SDK model dimension and rebuild the vector index."
            )
        if not all(math.isfinite(value) for value in values):
            raise KylinEmbeddingError("Kylin SDK returned a non-finite embedding value")
        if not self._normalize:
            return values
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise KylinEmbeddingError("Kylin SDK returned a zero embedding vector")
        return [value / norm for value in values]


class _CtypesKylinEmbeddingBackend:
    """Thin owner of one official ``TextEmbeddingSession``."""

    def __init__(self, library_path: str | None) -> None:
        self._library = _load_c_library(library_path)
        self._configure_signatures()
        self._session = ctypes.c_void_p(self._library.text_embedding_create_session())
        if not self._session.value:
            raise KylinEmbeddingError("Kylin SDK failed to create a text embedding session")
        error_code = int(self._library.text_embedding_init_session(self._session))
        if error_code != 0:
            self.close()
            raise KylinEmbeddingError(
                f"Kylin SDK failed to initialize the embedding session (error={error_code})"
            )

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def model_info(self) -> str | None:
        if not self._session.value:
            return None
        value = ctypes.c_char_p()
        success = bool(
            self._library.text_embedding_get_model_info(
                self._session,
                ctypes.byref(value),
            )
        )
        if not success or not value.value:
            return None
        try:
            return value.value.decode("utf-8", errors="replace")
        finally:
            self._library.embedding_model_info_destroy(value)

    def close(self) -> None:
        if getattr(self, "_session", None) is not None and self._session.value:
            self._library.text_embedding_destroy_session(ctypes.byref(self._session))
            self._session = ctypes.c_void_p()

    def _embed_one(self, text: str) -> list[float]:
        if not self._session.value:
            raise KylinEmbeddingError("Kylin embedding session is closed")
        result = ctypes.c_void_p()
        try:
            success = bool(
                self._library.text_embedding(
                    self._session,
                    text.encode("utf-8"),
                    ctypes.byref(result),
                )
            )
            if not result.value:
                raise KylinEmbeddingError(
                    "Kylin SDK returned no embedding result"
                    + ("" if success else " after reporting failure")
                )

            error_code = int(self._library.embedding_result_get_error_code(result))
            if not success or error_code != 0:
                message = self._library.embedding_result_get_error_message(result)
                detail = (
                    message.decode("utf-8", errors="replace")
                    if isinstance(message, bytes)
                    else "unknown SDK error"
                )
                raise KylinEmbeddingError(
                    f"Kylin text embedding failed (error={error_code}): {detail}"
                )

            length = int(self._library.embedding_result_get_vector_length(result))
            data = self._library.embedding_result_get_vector_data(result)
            if length <= 0 or not data:
                raise KylinEmbeddingError("Kylin SDK returned an empty embedding vector")
            return [float(data[index]) for index in range(length)]
        finally:
            if result.value:
                self._library.embedding_result_destroy(ctypes.byref(result))

    def _configure_signatures(self) -> None:
        library = self._library
        required = (
            "text_embedding_create_session",
            "text_embedding_init_session",
            "text_embedding_destroy_session",
            "text_embedding_get_model_info",
            "text_embedding",
            "embedding_result_get_vector_data",
            "embedding_result_get_vector_length",
            "embedding_result_get_error_code",
            "embedding_result_get_error_message",
            "embedding_result_destroy",
            "embedding_model_info_destroy",
        )
        missing = [name for name in required if not hasattr(library, name)]
        if missing:
            raise KylinEmbeddingError(
                "Kylin embedding library is missing required symbols: " + ", ".join(missing)
            )

        library.text_embedding_create_session.argtypes = []
        library.text_embedding_create_session.restype = ctypes.c_void_p
        library.text_embedding_init_session.argtypes = [ctypes.c_void_p]
        library.text_embedding_init_session.restype = ctypes.c_int
        library.text_embedding_destroy_session.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.text_embedding_destroy_session.restype = None
        library.text_embedding_get_model_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        library.text_embedding_get_model_info.restype = ctypes.c_bool
        library.text_embedding.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        library.text_embedding.restype = ctypes.c_bool
        library.embedding_result_get_vector_data.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_vector_data.restype = ctypes.POINTER(ctypes.c_float)
        library.embedding_result_get_vector_length.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_vector_length.restype = ctypes.c_int
        library.embedding_result_get_error_code.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_error_code.restype = ctypes.c_int
        library.embedding_result_get_error_message.argtypes = [ctypes.c_void_p]
        library.embedding_result_get_error_message.restype = ctypes.c_char_p
        library.embedding_result_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.embedding_result_destroy.restype = None
        library.embedding_model_info_destroy.argtypes = [ctypes.c_char_p]
        library.embedding_model_info_destroy.restype = None


class _PythonShimBackend:
    """Compatibility backend for the original project-specific Python shim."""

    def __init__(self, path: str, *, model_id: str) -> None:
        module_path = Path(path).expanduser()
        if not module_path.is_file():
            raise KylinEmbeddingError(f"Kylin Python SDK shim not found: {module_path}")
        spec = importlib.util.spec_from_file_location("kylin_embedding_sdk", str(module_path))
        if spec is None or spec.loader is None:
            raise KylinEmbeddingError(f"Cannot import Kylin Python SDK shim: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "embed_texts", None)):
            raise KylinEmbeddingError("Kylin Python SDK shim must define embed_texts()")
        self._module: Any = module
        self._model_id = model_id

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._module.embed_texts(list(texts), model_id=self._model_id)

    def model_info(self) -> str | None:
        callback = getattr(self._module, "model_info", None)
        return str(callback()) if callable(callback) else None

    def close(self) -> None:
        callback = getattr(self._module, "close", None)
        if callable(callback):
            callback()


def _load_c_library(library_path: str | None):
    if library_path:
        if os.path.sep in library_path and not Path(library_path).expanduser().is_file():
            raise KylinEmbeddingError(f"Kylin embedding library not found: {library_path}")
        candidates = [library_path]
    else:
        discovered = [ctypes.util.find_library(name) for name in _LIBRARY_BASENAMES]
        candidates = [candidate for candidate in (*discovered, *_LIBRARY_FILENAMES) if candidate]

    loader_errors: list[str] = []
    for candidate in dict.fromkeys(candidates):
        target = str(Path(candidate).expanduser()) if os.path.sep in candidate else candidate
        try:
            return ctypes.CDLL(target)
        except OSError as exc:
            loader_errors.append(f"{candidate}: {exc}")

    detail = "; ".join(loader_errors) or "no library candidates were found"
    raise KylinEmbeddingError(
        "Kylin embedding shared library is unavailable. Install the platform package "
        "libkylin-coreai-embedding-dev, ensure the AI subsystem is enabled, or set "
        f"{LIBRARY_ENV_VAR} to the shared-library path. Loader errors: {detail}"
    )


def _configured_library_path(value: str | os.PathLike[str] | None) -> str | None:
    if value is not None:
        return os.fspath(value)
    return (
        os.environ.get(LIBRARY_ENV_VAR, "").strip()
        or os.environ.get(SDK_PATH_ENV_VAR, "").strip()
        or None
    )


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int_from_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = [
    "DEFAULT_KYLIN_DIMENSION",
    "DEFAULT_MAX_INPUT_TOKENS",
    "DEFAULT_SESSION_MAX_TOKENS",
    "DEFAULT_SESSION_MAX_TEXTS",
    "DEFAULT_TOKENIZER_PATH",
    "DIMENSION_ENV_VAR",
    "KylinEmbeddingBackend",
    "KylinEmbeddingError",
    "KylinEmbeddingProvider",
    "LIBRARY_ENV_VAR",
    "MODEL_ID_ENV_VAR",
    "MAX_INPUT_TOKENS_ENV_VAR",
    "SESSION_MAX_TOKENS_ENV_VAR",
    "SESSION_MAX_TEXTS_ENV_VAR",
    "SDK_PATH_ENV_VAR",
    "TOKENIZER_PATH_ENV_VAR",
]
