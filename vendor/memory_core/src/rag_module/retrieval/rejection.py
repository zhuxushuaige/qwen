"""Optional rejection policy for unanswerable retrieval queries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.rag_module.retrieval.models import RagCandidate


@dataclass(frozen=True)
class RejectionConfig:
    """Scene-specific thresholds supplied explicitly by the caller."""

    dense_threshold: float = 0.0
    rerank_threshold: float = 0.0
    min_score_gap: float = 0.0
    require_sparse_hit: bool = False

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RejectionConfig":
        return cls(
            dense_threshold=float(value.get("dense_threshold", 0.0)),
            rerank_threshold=float(value.get("rerank_threshold", 0.0)),
            min_score_gap=float(value.get("min_score_gap", 0.0)),
            require_sparse_hit=bool(value.get("require_sparse_hit", False)),
        )


class RejectionPolicy:
    """Decide whether the top candidate is sufficiently supported."""

    def __init__(self, config: RejectionConfig | None = None) -> None:
        self._config = config or RejectionConfig()

    def should_reject(
        self,
        candidates: list[RagCandidate],
        top1_has_sparse_hit: bool = False,
    ) -> bool:
        if not candidates:
            return True

        config = self._config
        top1 = candidates[0]
        if top1.dense_score is not None and top1.dense_score < config.dense_threshold:
            return True
        if top1.rerank_score is not None and top1.rerank_score < config.rerank_threshold:
            return True

        if len(candidates) >= 2 and config.min_score_gap > 0:
            score1, score2 = _comparable_scores(candidates[0], candidates[1])
            if score1 is not None and score2 is not None:
                if score1 - score2 < config.min_score_gap and not top1_has_sparse_hit:
                    return True

        if config.require_sparse_hit and not top1_has_sparse_hit:
            return True
        return False


def save_rejection_config(config: RejectionConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


def load_rejection_config(path: str | Path) -> RejectionConfig:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("rejection config must be a JSON object")
    return RejectionConfig.from_dict(value)


def _comparable_scores(
    first: RagCandidate,
    second: RagCandidate,
) -> tuple[float | None, float | None]:
    for attribute in ("rerank_score", "fusion_score", "dense_score"):
        first_value = getattr(first, attribute)
        second_value = getattr(second, attribute)
        if first_value is not None and second_value is not None:
            return float(first_value), float(second_value)
    return None, None
