"""Context assembler: deduplicate, merge, and budget retrieval results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from src.rag_module.retrieval.models import RagCandidate

DEFAULT_TOKEN_BUDGET = 2000
MAX_PER_DOC = 3
OVERLAP_THRESHOLD = 0.8


@dataclass
class AssembledContext:
    """Result of context assembly."""

    candidates: list[RagCandidate]
    total_tokens: int
    warnings: list[str]


class ContextAssembler:
    """Assemble retrieval results into LLM-ready context."""

    def __init__(
        self,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_per_doc: int = MAX_PER_DOC,
    ) -> None:
        self._token_budget = token_budget
        self._max_per_doc = max_per_doc

    def assemble(
        self,
        candidates: list[RagCandidate],
        top_k: int | None = None,
    ) -> AssembledContext:
        if not candidates:
            return AssembledContext(candidates=[], total_tokens=0, warnings=[])

        warnings: list[str] = []

        # Expand child chunks to parent content for LLM context
        expanded = self._expand_to_parent(candidates)

        deduped = self._deduplicate(expanded)
        result: list[RagCandidate] = []
        doc_counts: dict[int, int] = {}
        total_tokens = 0
        budget_exceeded = False
        target_count = top_k if top_k is not None else len(deduped)

        # Walk the complete ranked list until enough contexts have survived
        # parent expansion, deduplication, document quotas, merging and budget.
        for c in deduped:
            merge_index = self._adjacent_result_index(result, c)
            if merge_index is not None:
                previous = result[merge_index]
                merged_content = previous.content + "\n" + c.content
                merged_tokens = _estimate_tokens(merged_content)
                delta = merged_tokens - _estimate_tokens(previous.content)
                if total_tokens + delta <= self._token_budget:
                    result[merge_index] = replace(
                        previous,
                        content=merged_content,
                        page_end=c.page_end,
                    )
                    total_tokens += delta
                    continue
                budget_exceeded = True

            if len(result) >= target_count:
                break
            if doc_counts.get(c.doc_pk, 0) >= self._max_per_doc:
                continue
            tokens = _estimate_tokens(c.content)
            if total_tokens + tokens > self._token_budget:
                budget_exceeded = True
                continue
            result.append(c)
            doc_counts[c.doc_pk] = doc_counts.get(c.doc_pk, 0) + 1
            total_tokens += tokens

        if budget_exceeded:
            warnings.append("token_budget_exceeded")

        return AssembledContext(
            candidates=result,
            total_tokens=total_tokens,
            warnings=warnings,
        )

    def _expand_to_parent(self, candidates: list[RagCandidate]) -> list[RagCandidate]:
        """Expand child chunks to parent content for richer LLM context."""
        result: list[RagCandidate] = []

        for c in candidates:
            parent_content = c.metadata.get("parent_content") if c.metadata else None
            if parent_content:
                result.append(replace(c, content=str(parent_content)))
            else:
                result.append(c)

        return result

    def _deduplicate(self, candidates: list[RagCandidate]) -> list[RagCandidate]:
        seen_keys: set[tuple[int, str]] = set()
        result: list[RagCandidate] = []
        for c in candidates:
            parent_id = c.metadata.get("parent_id") if c.metadata else None
            content_key = str(parent_id) if parent_id else hashlib.sha256(
                c.content.encode("utf-8")
            ).hexdigest()
            key = (c.doc_pk, content_key)
            if key not in seen_keys:
                seen_keys.add(key)
                result.append(c)
        return result

    @staticmethod
    def _adjacent_result_index(
        candidates: list[RagCandidate],
        candidate: RagCandidate,
    ) -> int | None:
        for index in range(len(candidates) - 1, -1, -1):
            previous = candidates[index]
            if previous.doc_pk != candidate.doc_pk:
                continue
            if previous.chunk_index is None or candidate.chunk_index is None:
                return None
            return index if abs(previous.chunk_index - candidate.chunk_index) == 1 else None
        return None


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 2)
