"""OpenRouter reranker for RAG retrieval candidates."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from src.rag_module.retrieval.models import RagCandidate

DEFAULT_RERANKER_MODEL = "cohere/rerank-4-fast"
DEFAULT_RERANKER_URL = "https://openrouter.ai/api/v1/rerank"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0

Transport = Callable[[Request, float], bytes]
Sleeper = Callable[[float], None]


class OpenRouterReranker:
    """Rerank RAG candidates through OpenRouter's rerank endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_RERANKER_MODEL,
        *,
        endpoint: str = DEFAULT_RERANKER_URL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
        transport: Transport | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is required for RAG reranking")
        if not model.strip():
            raise ValueError("reranker model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        self._model = model
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_delay_seconds = retry_delay_seconds
        self._transport = transport or _send_request
        self._sleep = sleeper or time.sleep

    @property
    def runtime_config(self) -> dict[str, object]:
        return {
            "provider": "openrouter",
            "model_id": self._model,
            "endpoint": self._endpoint,
            "timeout_seconds": self._timeout_seconds,
            "max_attempts": self._max_attempts,
        }

    def rerank(
        self,
        query: str,
        candidates: list[RagCandidate],
        top_k: int,
    ) -> list[RagCandidate]:
        if top_k <= 0 or not candidates:
            return []

        request = Request(
            self._endpoint,
            data=json.dumps(
                {
                    "model": self._model,
                    "query": query,
                    "documents": [candidate.content for candidate in candidates],
                    "top_n": len(candidates),
                },
                ensure_ascii=False,
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        ranked = self._request_rankings(request, len(candidates))

        selected_indices = {index for index, _ in ranked}
        ordered: list[RagCandidate] = []
        for index, score in ranked:
            candidate = candidates[index]
            candidate.rerank_score = score
            ordered.append(candidate)

        # Preserve candidates omitted by the service so reranking cannot reduce recall.
        ordered.extend(
            candidate for index, candidate in enumerate(candidates) if index not in selected_indices
        )
        return ordered[:top_k]

    def _request_rankings(
        self,
        request: Request,
        candidate_count: int,
    ) -> list[tuple[int, float]]:
        for attempt in range(self._max_attempts):
            try:
                payload = json.loads(self._transport(request, self._timeout_seconds))
                return _validate_results(payload, candidate_count)
            except (OSError, ValueError):
                if attempt + 1 >= self._max_attempts:
                    raise
                self._sleep(self._retry_delay_seconds * (2**attempt))
        raise AssertionError("unreachable")


def _validate_results(payload: Any, candidate_count: int) -> list[tuple[int, float]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise ValueError("OpenRouter rerank response must contain a results list")

    ranked: list[tuple[int, float]] = []
    seen: set[int] = set()
    for item in payload["results"]:
        if not isinstance(item, dict):
            raise ValueError("OpenRouter rerank result must be an object")
        index = item.get("index")
        score = item.get("relevance_score")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= candidate_count
            or index in seen
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
        ):
            raise ValueError("OpenRouter returned an invalid rerank result")
        seen.add(index)
        ranked.append((index, float(score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _send_request(request: Request, timeout_seconds: float) -> bytes:
    with urlopen(request, timeout=timeout_seconds) as response:
        return response.read()
