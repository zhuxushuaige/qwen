"""Deterministic decomposition for multi-condition retrieval queries."""

from __future__ import annotations

import re

_BOUNDARY = re.compile(
    r"[。！？!?；;\n]+|(?:以及|并且|同时|另外|此外|还有|然后|分别)"
)
_MIN_SUBQUERY_CHARS = 4


class QueryDecomposer:
    """Produce a small set of coverage-oriented subqueries."""

    def __init__(self, max_subqueries: int = 3) -> None:
        if max_subqueries < 1:
            raise ValueError("max_subqueries must be positive")
        self._max_subqueries = max_subqueries

    def decompose(self, query: str) -> list[str]:
        original = query.strip()
        if not original:
            return []
        parts = []
        seen = {original}
        for value in _BOUNDARY.split(original):
            value = value.strip(" ，,。！？!?；;：:")
            if len(value) < _MIN_SUBQUERY_CHARS or value in seen:
                continue
            seen.add(value)
            parts.append(value)
            if len(parts) >= self._max_subqueries:
                break
        return [original, *parts] if len(parts) >= 2 else [original]
