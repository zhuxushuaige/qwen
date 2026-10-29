"""Sensitive-content filtering contract shared by Agent integrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class SensitivitySource(str, Enum):
    USER = "user"
    TOOL = "tool"
    MANUAL_CONFIG = "manual_config"
    BEHAVIOR = "behavior"
    DOCUMENT = "document"


class SecurityAction(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


@dataclass(frozen=True)
class SensitivityDecision:
    action: SecurityAction
    sanitized_text: str
    rule_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class SensitiveContentFilter(Protocol):
    def inspect(
        self,
        text: str,
        *,
        source: SensitivitySource,
        user_id: str,
        project_id: str,
    ) -> SensitivityDecision:
        """Return allow, redact, or block without persisting the original text."""
