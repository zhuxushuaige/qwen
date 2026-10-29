"""Rerank policy: decide whether to trigger reranking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankPolicyConfig:
    fast_mode: bool = False
    min_candidates_for_rerank: int = 1


class RerankPolicy:
    """Decide whether reranking should be applied."""

    def __init__(self, config: RerankPolicyConfig | None = None) -> None:
        self._config = config or RerankPolicyConfig()

    def should_rerank(
        self,
        final_top_k: int,
        candidate_count: int,
    ) -> bool:
        if self._config.fast_mode:
            return False
        if candidate_count <= final_top_k:
            return False
        if candidate_count < self._config.min_candidates_for_rerank:
            return False
        return True
