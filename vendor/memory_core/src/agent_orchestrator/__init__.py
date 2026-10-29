"""Public agent API with lazy imports to avoid cross-module import cycles."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AgentOrchestrator": ("src.agent_orchestrator.orchestrator", "AgentOrchestrator"),
    "LLMClient": ("src.agent_orchestrator.llm_generator", "LLMClient"),
    "LLMGenerator": ("src.agent_orchestrator.llm_generator", "LLMGenerator"),
    "MockLLMClient": ("src.agent_orchestrator.llm_generator", "MockLLMClient"),
    "MockLLMGenerator": ("src.agent_orchestrator.llm_generator", "MockLLMGenerator"),
    "WorkingMemoryManager": (
        "src.agent_orchestrator.working_memory",
        "WorkingMemoryManager",
    ),
    "build_agent_request": ("src.agent_orchestrator.api", "build_agent_request"),
    "create_agent_orchestrator": ("src.agent_orchestrator.api", "create_agent_orchestrator"),
    "handle_agent_request": ("src.agent_orchestrator.api", "handle_agent_request"),
    "run_agent": ("src.agent_orchestrator.api", "run_agent"),
    "serialize_agent_response": ("src.agent_orchestrator.api", "serialize_agent_response"),
}

_TYPE_EXPORTS = {
    "AgentOperationContext",
    "AgentRequest",
    "AgentResponse",
    "AgentTrace",
    "BuiltContext",
    "LLMConfig",
    "LLMResult",
    "MemoryResult",
    "OperationResult",
    "QueryAnalysis",
    "QueryRoute",
    "RagResult",
    "RetrievalPlan",
    "RewrittenQueries",
}

__all__ = sorted([*_EXPORTS, *_TYPE_EXPORTS])


def __getattr__(name: str):
    if name in _TYPE_EXPORTS:
        module = import_module("src.agent_orchestrator.types")
    elif name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(module, name)
    globals()[name] = value
    return value
