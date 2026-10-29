"""Lightweight DeepSeek API client (OpenAI-compatible protocol)."""

from __future__ import annotations

import logging
import json
import os
import random
import re
import threading
import time
from typing import Any
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
_RESPONSE_LOG_LOCK = threading.Lock()


def chat_checkpoint_model() -> str:
    """Preserve identity only for this run's explicitly approved alias migration."""
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    previous = os.environ.get("EVALUATION_CHAT_CHECKPOINT_MODEL", "")
    if model == "deepseek-flash" and previous == "deepseek-v4.1-flash-expires-on-0910":
        return previous
    return model


def record_chat_response(response: Any, *, requested_model: str, component: str) -> None:
    """Optionally audit model aliases and usage, never prompts, responses or keys."""
    destination = os.environ.get("DEEPSEEK_USAGE_LOG", "")
    if not destination:
        return
    usage = getattr(response, "usage", None)
    entry = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "task": os.environ.get("EVALUATION_TASK", "unspecified"),
        "component": component,
        "requested_model": requested_model,
        "returned_model": getattr(response, "model", None),
        "response_id": getattr(response, "id", None),
        "usage_available": usage is not None,
        "usage": {name: _usage_int(usage, name) for name in (
            "prompt_tokens", "completion_tokens", "total_tokens",
            "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
        )},
    }
    try:
        with _RESPONSE_LOG_LOCK:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        logger.warning("Could not append Chat usage audit", exc_info=True)


class MalformedChatResponseError(RuntimeError):
    """Raised when an OpenAI-compatible endpoint returns no usable content."""


class DeepSeekClient:
    """Thin wrapper around the OpenAI-compatible DeepSeek chat API.

    Reads ``DEEPSEEK_API_KEY`` from the environment when no *api_key* is
    provided.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        timeout: float = 60.0,
        response_attempts: int = 3,
        transport_attempts: int | None = None,
        retry_backoff_seconds: float = 1.0,
        retry_max_seconds: float | None = None,
        thinking: bool | None = False,
        max_concurrency: int | None = None,
    ) -> None:
        resolved = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not resolved:
            raise ValueError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable."
            )
        self._api_key = resolved
        self._base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self._model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._thinking = thinking
        configured_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else int(os.environ.get("DEEPSEEK_MAX_CONCURRENCY", "1"))
        )
        if configured_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        self._max_concurrency = configured_concurrency
        self._request_semaphore = threading.BoundedSemaphore(configured_concurrency)
        if response_attempts <= 0:
            raise ValueError("response_attempts must be positive")
        configured_transport_attempts = (
            transport_attempts
            if transport_attempts is not None
            else int(os.environ.get("DEEPSEEK_TRANSPORT_ATTEMPTS", "8"))
        )
        if configured_transport_attempts <= 0:
            raise ValueError("transport_attempts must be positive")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        configured_retry_max_seconds = (
            retry_max_seconds
            if retry_max_seconds is not None
            else float(os.environ.get("DEEPSEEK_RETRY_MAX_SECONDS", "30"))
        )
        if configured_retry_max_seconds < 0:
            raise ValueError("retry_max_seconds must be non-negative")
        self._response_attempts = response_attempts
        self._transport_attempts = configured_transport_attempts
        self._retry_backoff_seconds = retry_backoff_seconds
        self._retry_max_seconds = configured_retry_max_seconds
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self._usage_totals = {
            "requests": 0,
            "prompt_tokens": 0,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def thinking(self) -> bool | None:
        return self._thinking

    @property
    def max_concurrency(self) -> int:
        return self._max_concurrency

    @property
    def usage(self) -> dict[str, int | float]:
        """Return aggregate API usage without exposing request or response text."""

        with self._usage_lock:
            result: dict[str, int | float] = dict(self._usage_totals)
        prompt_tokens = int(result["prompt_tokens"])
        result["prompt_cache_hit_rate"] = (
            int(result["prompt_cache_hit_tokens"]) / prompt_tokens
            if prompt_tokens
            else 0.0
        )
        return result

    # ------------------------------------------------------------------
    # lazy client init
    # ------------------------------------------------------------------

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        with self._client_lock:
            if self._client is not None:
                return
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, str] | None = None,
    ) -> str:
        """Send a chat completion request and return the text content.

        Parameters
        ----------
        messages:
            List of ``{"role": "...", "content": "..."}`` dicts.
        model:
            Override the default model.
        temperature:
            Override the default temperature.
        max_tokens:
            Override the default max_tokens.
        response_format:
            e.g. ``{"type": "json_object"}``.
        """
        self._ensure_client()
        assert self._client is not None

        kwargs: dict[str, Any] = {
            "model": model or self._model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self._temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if self._thinking is not None:
            kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if self._thinking else "disabled"}
            }

        with self._request_semaphore:
            return self._chat_with_retries(kwargs, response_format=response_format)

    def _chat_with_retries(
        self,
        kwargs: dict[str, Any],
        *,
        response_format: dict[str, str] | None,
    ) -> str:
        assert self._client is not None
        last_error: Exception | None = None
        for attempt in range(1, self._response_attempts + 1):
            response = self._create_with_transport_retries(kwargs)
            record_chat_response(response, requested_model=kwargs["model"], component="deepseek-client")
            self._record_usage(response)
            try:
                content = _chat_content(response)
                if response_format and response_format.get("type") == "json_object":
                    return _strip_json_fence(content)
                return content
            except MalformedChatResponseError as exc:
                last_error = exc
                if attempt < self._response_attempts:
                    logger.warning(
                        "Malformed chat response from %s; retrying (%d/%d): %s",
                        self._base_url,
                        attempt,
                        self._response_attempts,
                        exc,
                    )
        assert last_error is not None
        raise last_error

    def _create_with_transport_retries(self, kwargs: dict[str, Any]) -> Any:
        """Retry transient transport failures without consuming response retries."""

        assert self._client is not None
        for attempt in range(1, self._transport_attempts + 1):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not _is_retryable_chat_error(exc) or attempt >= self._transport_attempts:
                    raise
                base_delay = min(
                    self._retry_max_seconds,
                    self._retry_backoff_seconds * (2 ** (attempt - 1)),
                )
                delay = base_delay * random.uniform(0.75, 1.25) if base_delay else 0.0
                logger.warning(
                    "Transient chat transport failure from %s; retrying in %.1fs "
                    "(%d/%d): %s",
                    self._base_url,
                    delay,
                    attempt,
                    self._transport_attempts,
                    exc,
                )
                if delay:
                    time.sleep(delay)
        raise AssertionError("transport retry loop exited unexpectedly")

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        values = {
            "requests": 1,
            "prompt_tokens": _usage_int(usage, "prompt_tokens"),
            "prompt_cache_hit_tokens": _usage_int(
                usage,
                "prompt_cache_hit_tokens",
            ),
            "prompt_cache_miss_tokens": _usage_int(
                usage,
                "prompt_cache_miss_tokens",
            ),
            "completion_tokens": _usage_int(usage, "completion_tokens"),
            "reasoning_tokens": _usage_int(
                getattr(usage, "completion_tokens_details", None),
                "reasoning_tokens",
            ),
        }
        with self._usage_lock:
            for key, value in values.items():
                self._usage_totals[key] += value


def build_chat_client(provider: str | None = None) -> DeepSeekClient:
    """Build the configured shared DeepSeek client used by project Chat tasks."""

    normalized = (provider or os.environ.get("CHAT_PROVIDER", "deepseek")).strip().lower()
    if normalized in {"", "deepseek"}:
        return DeepSeekClient()
    raise ValueError(f"unsupported chat provider: {provider}")


def _chat_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise MalformedChatResponseError("chat response has no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise MalformedChatResponseError("chat response has no text content")
    return content


def _strip_json_fence(content: str) -> str:
    """Remove a complete Markdown JSON fence from structured output."""

    fenced = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", content, flags=re.DOTALL)
    return fenced.group(1).strip() if fenced else content


def _usage_int(value: Any, field: str) -> int:
    if value is None:
        return 0
    raw = value.get(field, 0) if isinstance(value, dict) else getattr(value, field, 0)
    return int(raw or 0)


def _is_retryable_chat_error(exc: Exception) -> bool:
    """Return whether a failed request is safe to retry without changing its payload."""

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        from openai import APIConnectionError, APIStatusError, APITimeoutError
    except ImportError:
        return False
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = int(getattr(exc, "status_code", 0) or 0)
        return status_code in {408, 409, 429} or status_code >= 500
    return False
