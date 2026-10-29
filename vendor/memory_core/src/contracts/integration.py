"""Stable v1 integration constants and failure vocabulary."""

from __future__ import annotations

from enum import Enum

CONTRACT_VERSION = "1.0"
PUBLIC_METADATA_KEYS = frozenset(
    {
        "provider",
        "model",
        "cache_hit",
        "provenance",
        "effective_at",
        "sensitivity",
    }
)


class IntegrationFailureCode(str, Enum):
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"
    SCOPE_VIOLATION = "scope_violation"
    SECURITY_BLOCKED = "security_blocked"


class IntegrationError(RuntimeError):
    """Safe cross-module failure without leaking provider details."""

    def __init__(
        self,
        code: IntegrationFailureCode,
        *,
        component: str,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.component = component
        self.retryable = retryable
        super().__init__(f"{component}:{code.value}")
