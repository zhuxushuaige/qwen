"""Adapter for the existing local assistant memory; mutations require explicit calls."""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(os.environ.get("MEMORY_PROJECT_ROOT", Path(__file__).resolve().parent / "vendor" / "memory_core"))
DEFAULT_DB = Path(__file__).resolve().parent / "local_assistant_data" / "memory.sqlite3"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent_orchestrator.types import AgentRequest
from src.memory_module.retriever.sqlite import FeatureHashEmbeddingProvider, SQLiteMemoryRetriever
from src.memory_module.schema.sqlite_store import SQLiteMemoryStore
from src.memory_module.writer.rule_based import RuleBasedMemoryWriter


def _item(record):
    return {"id": "M" + hashlib.sha256(record.memory_id.encode("utf-8")).hexdigest()[:12],
            "memory_id": record.memory_id, "content": record.content,
            "type": record.type, "source": record.source, "status": record.status}


def can_save(text):
    """Conservative gate for a direct command; UI buttons are separate consent."""
    return bool(re.match(r"^(?:请|帮我|替我)?(?:记住|保存(?:这条)?记忆)[:：,，\s]*\S", text.strip(), re.I))


def can_forget(text):
    return bool(re.match(r"^(?:请|帮我|替我)?(?:忘记|遗忘|删除(?:这条)?记忆)[:：,，\s]*\S", text.strip(), re.I))


class MemoryBridge:
    """Reuse the original SQLite memory components within one user/project scope.

    save and forget are intended exclusively for explicit user actions. context
    returns preferences as behavioral guidance, not evidence about experiments.
    """
    def __init__(self, db_path=DEFAULT_DB, user_id="local_user", project_id="research_assistant"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.user_id, self.project_id = user_id, project_id
        self.embedding = FeatureHashEmbeddingProvider()
        self.store = SQLiteMemoryStore(self.db_path)
        self.retriever = SQLiteMemoryRetriever(self.db_path, self.embedding, vector_store_backend="sqlite")
        # The original writer constructs its own retriever using this setting.
        with patch.dict(os.environ, {"VECTOR_STORE_BACKEND": "sqlite"}):
            self.writer = RuleBasedMemoryWriter(self.db_path, self.embedding)

    def list_active(self):
        return [_item(r) for r in self.store.list_memories(self.user_id, self.project_id, status="active", limit=1000000)]

    def search(self, query, limit=6):
        return [_item(r) for r in self.retriever.search(query, self.user_id, self.project_id, top_k=limit)]

    def context(self, query, limit=4):
        if limit <= 0:
            return []
        preferences = [r for r in self.list_active() if r["type"] == "preference" or re.search(r"(?:以后|之后|下次).{0,12}(?:回答|回复|答复).{0,12}(?:简短|简洁|详细|中文|英文)", r["content"])]
        matches = self.search(query, limit)
        # Always include preferences, even for unrelated queries; the limit bounds
        # retrieved items, not the user's existing response preferences.
        seen = set()
        result = []
        for item in preferences + matches:
            if item["memory_id"] not in seen:
                seen.add(item["memory_id"])
                result.append(item)
        return result

    def save(self, text, session):
        if not text or not text.strip():
            return {"success": False, "message": "记忆内容不能为空。"}
        text = re.sub(r'^(?:请|帮我|替我)?保存(?:这条)?记忆[:：,，\s]*', '记住：', text.strip())
        request = AgentRequest(user_input=text, user_id=self.user_id, project_id=self.project_id,
                               conversation_id=str(session) if session is not None else None,
                               metadata={"preference_source": "explicit_user"})
        result = self.writer.write_from_request(request)
        mid = result.metadata.get("memory_id")
        record = self.store.get(mid) if mid else None
        return {"success": result.success, "message": result.message, "warnings": result.warnings,
                "errors": result.errors, "memory": _item(record) if record else None}

    def forget(self, memory_id):
        # Resolve either the original ID or its stable display ID, within scope.
        items = self.list_active()
        found = next((r for r in items if memory_id in (r["memory_id"], r["id"])), None)
        if found is None:
            return {"success": False, "message": "未找到当前范围内的有效记忆。"}
        count = self.store.forget([found["memory_id"]])
        record = self.store.get(found["memory_id"])
        return {"success": bool(count), "memory": _item(record) if record else None}
