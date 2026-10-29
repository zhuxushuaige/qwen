"""Ebbinghaus forgetting curve with retrieval-based reinforcement.

Matches MemoryBank approach: use-it-or-lose-it.
- On retrieval: memory_strength += 1, last_recall_date refreshed
- On load/search: retention = exp(-days / (5 * strength))
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Protocol


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _days_since(iso_date: str) -> float:
    """Days elapsed since the given ISO date."""
    then = _parse_iso(iso_date)
    now = datetime.now(timezone.utc)
    return max((now - then).total_seconds() / 86400.0, 0.0)


def forgetting_curve(t_days: float, strength: float) -> float:
    """Ebbinghaus retention: exp(-t / (5 * S)).

    - S=1: 5 days → 37%, 10 days → 14%
    - S=5: 5 days → 82%, 10 days → 67%
    - S=10: 5 days → 90%, 10 days → 82%
    """
    if strength <= 0:
        strength = 0.1
    return math.exp(-t_days / (5.0 * strength))


class TemporalDecay(Protocol):
    def compute(self, created_at: str, now: str | None = None) -> float: ...


class EbbinghausDecay:
    """Ebbinghaus forgetting curve — matches MemoryBank use-it-or-lose-it.

    - load_and_forget(): iterate ALL memories, randomly delete weak ones
    - update_on_recall(): strength += 1, refresh date (called on retrieval)
    - compute(): retention score for display/sorting
    """

    def __init__(self, random_filter: bool = True) -> None:
        self._random_filter = random_filter

    def compute(self, last_recall_date: str, strength: float = 1.0) -> float:
        t = _days_since(last_recall_date)
        return forgetting_curve(t, strength)

    def load_and_forget(self, store, user_id: str, project_id: str) -> list[str]:
        """Iterate ALL active memories, randomly delete weak ones.

        Returns list of forgotten memory IDs.
        """
        if not self._random_filter:
            return []
        records = store.search(user_id, project_id, status="active")
        if not records:
            return []
        forget_ids: list[str] = []
        for record in records:
            last_date = record.metadata.get("last_recall_date", record.created_at)
            strength = record.metadata.get("memory_strength", 1)
            retention = self.compute(last_date, float(strength))
            if random.random() > retention:
                forget_ids.append(record.memory_id)
        if forget_ids:
            store.delete(forget_ids)
        return forget_ids

    @staticmethod
    def update_on_recall(memory_id: str, store) -> None:
        """Called when a memory is retrieved: strength += 1, refresh date."""
        record = store.get(memory_id)
        if record is None:
            return
        strength = record.metadata.get("memory_strength", 1)
        record.metadata["memory_strength"] = int(strength) + 1
        record.metadata["last_recall_date"] = datetime.now(timezone.utc).isoformat()
        store.upsert(record)


class ExponentialDecay:
    """Simple exponential decay (no strength parameter)."""

    def __init__(self, lam: float = 0.01) -> None:
        self.lam = lam

    def compute(self, created_at: str, now: str | None = None) -> float:
        then = _parse_iso(created_at)
        current = _parse_iso(now) if now is not None else datetime.now(timezone.utc)
        days = max((current - then).total_seconds() / 86400.0, 0.0)
        return math.exp(-self.lam * days)


class StepDecay:
    """Step-wise decay: 7d=1.0, 30d=0.5, 90d=0.25."""

    def __init__(self, thresholds: list[tuple[int, float]] | None = None) -> None:
        self.thresholds = thresholds or [(7, 1.0), (30, 0.5)]

    def compute(self, created_at: str, now: str | None = None) -> float:
        then = _parse_iso(created_at)
        current = _parse_iso(now) if now is not None else datetime.now(timezone.utc)
        days = max((current - then).total_seconds() / 86400.0, 0.0)
        result = 0.25
        for days_limit, factor in self.thresholds:
            if days < days_limit:
                result = factor
                break
        return result


class NoDecay:
    """No decay — always returns 1.0."""

    def compute(self, created_at: str, now: str | None = None) -> float:
        return 1.0
