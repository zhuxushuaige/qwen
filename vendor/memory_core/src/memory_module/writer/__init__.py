"""Writer implementations, loaded without importing optional storage backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BatchConflictPlanner": ("batch_conflict", "BatchConflictPlanner"),
    "BatchCandidateEvaluation": ("batch_conflict", "BatchCandidateEvaluation"),
    "BatchDecisionError": ("batch_conflict", "BatchDecisionError"),
    "BatchMemoryDecision": ("batch_conflict", "BatchMemoryDecision"),
    "BatchMemoryInput": ("batch_conflict", "BatchMemoryInput"),
    "LLMBatchConflictPlanner": ("batch_conflict", "LLMBatchConflictPlanner"),
    "LLMMemOSConflictOrganizer": (
        "batch_conflict",
        "LLMMemOSConflictOrganizer",
    ),
    "MemOSCandidateRelation": ("batch_conflict", "MemOSCandidateRelation"),
    "MemOSDetectionInput": ("batch_conflict", "MemOSDetectionInput"),
    "MemOSConflictOrganizer": ("batch_conflict", "MemOSConflictOrganizer"),
    "MemOSFusionResult": ("batch_conflict", "MemOSFusionResult"),
    "LayeredExtractionResult": ("tencent_extractor", "LayeredExtractionResult"),
    "TencentDBLayeredExtractor": (
        "tencent_extractor",
        "TencentDBLayeredExtractor",
    ),
    "ConflictDecision": ("conflict", "ConflictDecision"),
    "MemoryConflictDetector": ("conflict", "MemoryConflictDetector"),
    "MemoryRelation": ("conflict", "MemoryRelation"),
    "StructuredConflictDetector": ("conflict", "StructuredConflictDetector"),
    "LayeredConflictDetector": ("conflict", "LayeredConflictDetector"),
    "MemoryExtractor": ("extractor", "MemoryExtractor"),
    "PromptMemoryExtractor": ("extractor", "PromptMemoryExtractor"),
    "DynamicPreferenceExtractor": ("preference", "DynamicPreferenceExtractor"),
    "LocalTransformersPreferenceClient": (
        "preference",
        "LocalTransformersPreferenceClient",
    ),
    "OpenAICompatiblePreferenceClient": (
        "preference",
        "OpenAICompatiblePreferenceClient",
    ),
    "PreferenceContext": ("preference", "PreferenceContext"),
    "PreferenceEvidenceStore": ("preference", "PreferenceEvidenceStore"),
    "PreferenceFeedbackStore": ("preference", "PreferenceFeedbackStore"),
    "StructuredPreferenceExtractor": ("preference", "StructuredPreferenceExtractor"),
    "MemoryWriter": ("interface", "MemoryWriter"),
    "EntityLinker": ("entity_linker", "EntityLinker"),
    "DeepSeekClient": ("llm_client", "DeepSeekClient"),
    "RuleBasedMemoryWriter": ("rule_based", "RuleBasedMemoryWriter"),
    "MemoryLifecycleManager": ("lifecycle", "MemoryLifecycleManager"),
    "SessionConsolidator": ("consolidator", "SessionConsolidator"),
    "ConsolidationResult": ("consolidator", "ConsolidationResult"),
    "MemoryPromotionPolicy": ("promotion", "MemoryPromotionPolicy"),
    "PromotionDecision": ("promotion", "PromotionDecision"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
