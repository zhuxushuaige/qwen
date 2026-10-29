"""Lifecycle coordination for persistent provenance, medium, and long memory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.memory_module.schema.layered_store import LayeredMemoryStore
from src.memory_module.schema.sqlite_store import SQLiteMemoryStore
from src.memory_module.writer.preference import PreferenceEvidenceStore


class MemoryLifecycleManager:
    """Coordinate existing L0, L2/evidence, and durable L1 stores.

    The manager deliberately does not duplicate extraction, conflict handling,
    or retrieval. It owns lifecycle transitions shared by those components.
    """

    def __init__(self, db_path: str | Path = ".memory_module.sqlite3") -> None:
        self.layered_store = LayeredMemoryStore(db_path)
        self.preference_evidence = PreferenceEvidenceStore(db_path)
        self.long_term_store = SQLiteMemoryStore(db_path)

    def maintain(self) -> dict[str, int]:
        """Expire L0 provenance and archive stale medium-term entries."""

        result = self.layered_store.expire_lifecycle()
        result["preference_evidence"] = self.preference_evidence.expire()
        return result

    def mark_promoted(self, metadata: dict[str, Any], memory_id: str) -> int:
        """Link medium-term evidence or a scene to its durable memory."""

        evidence_key = str(metadata.get("evidence_key", "")).strip()
        scene_id = str(metadata.get("medium_scene_id", "")).strip()
        changed = 0
        if evidence_key:
            changed += self.preference_evidence.mark_promoted(evidence_key, memory_id)
        if scene_id:
            changed += self.layered_store.mark_scene_promoted(scene_id, memory_id)
        return changed

    def status(self, user_id: str, project_id: str) -> dict[str, int]:
        """Return auditable persistent-store counts.

        Process-local short-term memory belongs to WorkingMemoryManager and is
        intentionally absent from this persistent lifecycle view.
        """

        l0_count = len(self.layered_store.recent_messages(user_id, project_id, limit=10_000))
        medium_scenes = len(self.layered_store.list_scenes(user_id, project_id, limit=10_000))
        medium_evidence = len(self.preference_evidence.pending(user_id, project_id, limit=10_000))
        return {
            "l0_provenance": l0_count,
            "medium_term": medium_scenes + medium_evidence,
            "long_term": len(self.long_term_store.search(user_id, project_id)),
        }
