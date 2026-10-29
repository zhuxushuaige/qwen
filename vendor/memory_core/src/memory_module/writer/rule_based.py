"""Memory writer with MemOS-style post-write conflict organization."""

from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any

from src.agent_orchestrator.types import AgentOperationContext, AgentRequest, OperationResult
from src.memory_module.retriever.sqlite import SQLiteMemoryRetriever
from src.memory_module.schema.models import (
    ACTIVE,
    DEFAULT_MEMORY_TYPE,
    PREFERENCE_MEMORY,
    PROJECT_MEMORY,
    SUPERSEDED,
    TASK_MEMORY,
    MemoryCandidate,
    MemoryRecord,
)
from src.memory_module.schema.sqlite_store import SQLiteMemoryStore
from src.memory_module.writer.batch_conflict import (
    MEMOS_CONTRADICTORY,
    MEMOS_INDEPENDENT,
    MEMOS_REDUNDANT,
    MemOSCandidateRelation,
    MemOSConflictOrganizer,
    MemOSDetectionInput,
)
from src.memory_module.writer.extractor import MemoryExtractor
from src.memory_module.writer.lifecycle import MemoryLifecycleManager
from src.memory_module.writer.preference import (
    EXPLICIT_USER,
    PreferenceContext,
    PreferenceFeedbackStore,
    normalize_preference_source,
)
from src.rag_module.embedding.base import EmbeddingProvider

DEFAULT_SEMANTIC_CONFIRMATION_THRESHOLD = 0.80
DEFAULT_SEMANTIC_CANDIDATE_K = 5
DEFAULT_ORGANIZATION_MAX_ATTEMPTS = 3
DEFAULT_ORGANIZATION_LEASE_SECONDS = 600.0
DEFAULT_ORGANIZATION_BATCH_SIZE = 12


class RuleBasedMemoryWriter:
    """Store first, then organize related nodes like MemOS.

    Exact duplicates are refreshed deterministically. Newly created records are
    immediately visible, recalled against existing active nodes, and then either
    left independent or consolidated into a new version node. If the organizer
    is unavailable, both memories remain active instead of risking data loss.
    """

    def __init__(
        self,
        db_path: str | Path = ".memory_module.sqlite3",
        embedding_provider: EmbeddingProvider | None = None,
        semantic_confirmation_threshold: float = DEFAULT_SEMANTIC_CONFIRMATION_THRESHOLD,
        semantic_candidate_k: int = DEFAULT_SEMANTIC_CANDIDATE_K,
        memory_extractor: MemoryExtractor | None = None,
        conflict_organizer: MemOSConflictOrganizer | None = None,
    ) -> None:
        if not 0.0 <= semantic_confirmation_threshold <= 1.0:
            raise ValueError("semantic_confirmation_threshold must be between 0 and 1")
        if semantic_candidate_k <= 0:
            raise ValueError("semantic_candidate_k must be positive")
        self.store = SQLiteMemoryStore(db_path)
        self.retriever = SQLiteMemoryRetriever(db_path, embedding_provider)
        self.semantic_confirmation_threshold = semantic_confirmation_threshold
        self.semantic_candidate_k = semantic_candidate_k
        self.memory_extractor = memory_extractor
        self.conflict_organizer = conflict_organizer
        self.lifecycle = MemoryLifecycleManager(db_path)
        self.preference_feedback_store = PreferenceFeedbackStore(db_path)
        self._organization_queue: Queue[bool | None] = Queue()
        self._organization_lock = Lock()
        self._organization_closed = False
        self._organization_max_attempts = DEFAULT_ORGANIZATION_MAX_ATTEMPTS
        self._organization_lease_seconds = DEFAULT_ORGANIZATION_LEASE_SECONDS
        self._organization_thread: Thread | None = None
        if conflict_organizer is not None:
            self._organization_thread = Thread(
                target=self._run_organization_loop,
                name="memory-conflict-organizer",
                daemon=True,
            )
            self._organization_thread.start()
            self._organization_queue.put(True)

    def record_preference_feedback(
        self,
        request: AgentRequest,
        memory_id: str,
        outcome: str,
        *,
        event_id: str | None = None,
        context: AgentOperationContext | None = None,
    ) -> OperationResult:
        """Record an explicit accept, modify, or reject signal."""

        try:
            feedback = self.preference_feedback_store.record(
                PreferenceContext(request.user_id, request.project_id),
                memory_id,
                outcome,
                event_id=event_id,
            )
        except ValueError as exc:
            return OperationResult(
                success=False,
                message="Preference feedback was not recorded.",
                errors=[f"preference_feedback_invalid:{exc}"],
                metadata=_base_metadata(context),
            )
        return OperationResult(
            success=True,
            message=(
                "Preference feedback recorded."
                if feedback["applied"]
                else "Preference feedback was already recorded."
            ),
            metadata={
                **_base_metadata(context),
                "operation": "preference_feedback_recorded",
                **feedback,
            },
        )

    def organization_status(self, memory_id: str) -> dict[str, Any]:
        """Return a snapshot of one asynchronous organization job."""

        state = self.store.get_organization_job(memory_id)
        if state is None:
            return {"memory_id": memory_id, "status": "not_queued"}
        return state

    def wait_for_organization(self, timeout: float = 600.0) -> bool:
        """Wait until all currently queued conflict work is finished."""

        if timeout < 0:
            raise ValueError("timeout must be non-negative")
        deadline = time.monotonic() + timeout
        while self.store.pending_organization_jobs():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, *, wait: bool = True, timeout: float = 30.0) -> bool:
        """Stop the background worker, optionally draining queued jobs first."""

        if self._organization_thread is None:
            return True
        if wait and not self.wait_for_organization(timeout):
            return False
        with self._organization_lock:
            if self._organization_closed:
                return not self._organization_thread.is_alive()
            self._organization_closed = True
        self._organization_queue.put(None)
        self._organization_thread.join(timeout=max(timeout, 0.0))
        return not self._organization_thread.is_alive()

    def write_from_request(
        self,
        request: AgentRequest,
        context: AgentOperationContext | None = None,
    ) -> OperationResult:
        extracted = self._extract_request(request, context)
        if isinstance(extracted, OperationResult):
            return extracted
        extracted, review_candidates = _partition_preference_review_candidates(extracted)
        if not extracted:
            return _preference_review_result(review_candidates, context)
        if self.conflict_organizer is not None:
            result = self._write_memos_batch(extracted, request, context)
            return _attach_preference_review_metadata(result, review_candidates)
        results = [
            self._store_direct(self._build_record(candidate, request), context)
            for candidate in extracted
        ]
        return _attach_preference_review_metadata(
            self._combine_results(results, context, llm_calls=0),
            review_candidates,
        )

    def promote_medium_memory(
        self,
        request: AgentRequest,
        scene: dict[str, Any],
        context: AgentOperationContext | None = None,
    ) -> OperationResult:
        """Promote one policy-approved medium-term item through the normal writer."""

        summary = str(scene.get("summary", "")).strip()
        scene_id = str(scene.get("scene_id", "")).strip()
        if not summary or not scene_id:
            return OperationResult(
                success=False,
                message="Medium-term memory is missing its summary or scene ID.",
                warnings=["medium_memory_promotion_invalid"],
                metadata=_base_metadata(context),
            )
        task_id = str(scene.get("task_id", ""))
        candidate = MemoryCandidate(
            content=summary,
            type=TASK_MEMORY if task_id else PROJECT_MEMORY,
            subject=task_id or str(scene.get("scene_name", "session")),
            attribute="task_summary" if task_id else "session_summary",
            value=summary,
            scope="task" if task_id else "project",
            confidence=min(1.0, max(0.0, float(scene.get("importance", 0.5)))),
            source="medium_term_promotion",
            metadata={
                "medium_scene_id": scene_id,
                "medium_kind": str(scene.get("kind", "scene")),
                "promotion_state": dict(scene.get("state", {})),
                "task_id": task_id,
            },
        )
        record = self._build_record(candidate, request)
        if self.conflict_organizer is not None:
            return self._store_then_enqueue(record, context)
        return self._store_direct(record, context)

    def capture_turn(
        self,
        request: AgentRequest,
        assistant_response: str | None = None,
        context: AgentOperationContext | None = None,
    ) -> OperationResult:
        """Capture L0 and organize extracted L1 nodes after they are stored."""

        capture = getattr(self.memory_extractor, "capture_turn", None)
        if not callable(capture):
            return OperationResult(
                success=True,
                message="Memory writer does not require conversation capture.",
                metadata={**_base_metadata(context), "action": "capture_not_required"},
            )
        try:
            extra_messages = _tool_messages(context)
            extracted = capture(
                request.user_input,
                assistant_response,
                context=PreferenceContext(
                    user_id=request.user_id,
                    project_id=request.project_id,
                    task_id=str(request.metadata.get("task_id", "")),
                ),
                conversation_id=request.conversation_id,
                force=False,
                extra_messages=extra_messages,
            )
        except Exception as exc:
            return OperationResult(
                success=False,
                message="Conversation capture or layered extraction failed.",
                errors=[f"layered_memory_capture_failed:{exc}"],
                metadata={**_base_metadata(context), "action": "capture_failed"},
            )
        extracted, review_candidates = _partition_preference_review_candidates(extracted)
        if not extracted and review_candidates:
            return _preference_review_result(review_candidates, context, layer="L1")
        if not extracted:
            return OperationResult(
                success=True,
                message="Conversation captured in L0; L1 extraction is not due.",
                metadata={
                    **_base_metadata(context),
                    "action": "l0_captured",
                    "layer": "L0",
                },
            )
        if self.conflict_organizer is not None:
            result = self._write_memos_batch(extracted, request, context)
        else:
            results = [
                self._store_direct(self._build_record(candidate, request), context)
                for candidate in extracted
            ]
            result = self._combine_results(results, context, llm_calls=0)
        if self.conflict_organizer is None:
            self._finalize_layered_write(result, request)
        result.metadata["layer"] = "L1"
        layered_result = getattr(self.memory_extractor, "last_result", None)
        if layered_result is not None:
            result.metadata["scene_ids"] = list(layered_result.scene_ids)
            result.metadata["persona_updated"] = bool(layered_result.persona_updated)
        return _attach_preference_review_metadata(result, review_candidates)

    def _extract_request(
        self,
        request: AgentRequest,
        context: AgentOperationContext | None,
    ) -> list[MemoryCandidate] | OperationResult:
        if self.memory_extractor is not None:
            try:
                observe = getattr(self.memory_extractor, "observe", None)
                if callable(observe):
                    source_type = normalize_preference_source(
                        str(request.metadata.get("preference_source", EXPLICIT_USER))
                    )
                    extracted = observe(
                        request.user_input,
                        source_type=source_type,
                        context=PreferenceContext(
                            user_id=request.user_id,
                            project_id=request.project_id,
                            task_id=str(request.metadata.get("task_id", "")),
                        ),
                        event_id=(
                            str(request.metadata["event_id"])
                            if request.metadata.get("event_id")
                            else request.conversation_id
                        ),
                    )
                else:
                    extracted = self.memory_extractor.extract(request.user_input)
            except Exception as exc:
                return OperationResult(
                    success=False,
                    message="Memory extraction returned invalid structured data.",
                    errors=[f"memory_extraction_failed:{exc}"],
                    metadata=_base_metadata(context),
                )
        else:
            content = extract_memory_content(request.user_input)
            extracted = (
                [
                    MemoryCandidate(
                        content=content,
                        type=classify_memory_type(content),
                        source="rule_based_user_input",
                    )
                ]
                if content
                else []
            )
        if extracted:
            return extracted
        if bool(getattr(self.memory_extractor, "last_evidence_recorded", False)):
            return OperationResult(
                success=True,
                message="Preference evidence recorded; awaiting repeated observations.",
                warnings=["preference_evidence_pending"],
                metadata={**_base_metadata(context), "operation": "evidence_recorded"},
            )
        return OperationResult(
            success=False,
            message="No durable memory content was found in the request.",
            warnings=["memory_write_empty"],
            metadata=_base_metadata(context),
        )

    def _finalize_layered_write(
        self,
        result: OperationResult,
        request: AgentRequest,
    ) -> None:
        finalize = getattr(self.memory_extractor, "finalize_write", None)
        if not callable(finalize) or not result.success:
            return
        try:
            result.metadata["persona_updated"] = bool(
                finalize(
                    PreferenceContext(
                        user_id=request.user_id,
                        project_id=request.project_id,
                        task_id=str(request.metadata.get("task_id", "")),
                    )
                )
            )
        except Exception as exc:
            result.warnings.append(f"persona_generation_failed:{exc}")

    @staticmethod
    def _build_record(candidate: MemoryCandidate, request: AgentRequest) -> MemoryRecord:
        provenance = {
            key: request.metadata[key]
            for key in ("event_date", "event_id", "benchmark")
            if request.metadata.get(key) is not None
        }
        record = MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:12]}",
            user_id=request.user_id,
            project_id=request.project_id,
            content=candidate.content,
            type=candidate.type,
            status=ACTIVE,
            source=candidate.source,
            subject=candidate.subject,
            attribute=candidate.attribute,
            value=candidate.value,
            polarity=candidate.polarity,
            scope=candidate.scope,
            confidence=candidate.confidence,
            metadata={
                **candidate.metadata,
                **provenance,
                **(
                    {"conversation_id": request.conversation_id}
                    if request.conversation_id is not None
                    else {}
                ),
                "extracted_from": "user_input",
            },
        )
        event_date = str(
            candidate.metadata.get("effective_at") or request.metadata.get("event_date", "")
        ).strip()
        if event_date:
            record.effective_at = event_date
        return record

    def _write_memos_batch(
        self,
        candidates: list[MemoryCandidate],
        request: AgentRequest,
        context: AgentOperationContext | None,
    ) -> OperationResult:
        results = [
            self._store_then_enqueue(
                self._build_record(candidate, request),
                context,
                notify_worker=False,
            )
            for candidate in candidates
        ]
        if any(result.metadata.get("organization_status") == "queued" for result in results):
            self._signal_organization_worker()
        return self._combine_results(results, context, llm_calls=0)

    def _store_then_enqueue(
        self,
        record: MemoryRecord,
        context: AgentOperationContext | None,
        *,
        notify_worker: bool = True,
    ) -> OperationResult:
        """Persist a node and return after scheduling MemOS organization."""

        saved, created = self.store.upsert(
            record,
            organization_max_attempts=(
                self._organization_max_attempts if self.conflict_organizer is not None else None
            ),
        )
        self.lifecycle.mark_promoted(record.metadata, saved.memory_id)
        self.retriever.index_records([saved])
        if not created:
            result = self._success_result(
                saved,
                action="refreshed",
                semantic_decision="memos_exact_redundant",
                context=context,
                relation=MEMOS_REDUNDANT,
                organization_llm_calls=0,
            )
            result.metadata["organization_status"] = "not_required"
            return result

        if not self._enqueue_organization(saved.memory_id, notify=notify_worker):
            result = self._success_result(
                saved,
                action="created",
                semantic_decision="memos_organization_deferred",
                context=context,
                organization_llm_calls=0,
            )
            result.metadata["organization_status"] = "queued"
            result.metadata["organization_job_id"] = saved.memory_id
            result.warnings.append("memos_organization_worker_closed:persisted_for_recovery")
            return result
        result = self._success_result(
            saved,
            action="created",
            semantic_decision="memos_organization_queued",
            context=context,
            organization_llm_calls=0,
        )
        result.metadata["organization_status"] = "queued"
        result.metadata["organization_job_id"] = saved.memory_id
        return result

    def _enqueue_organization(self, memory_id: str, *, notify: bool = True) -> bool:
        with self._organization_lock:
            if self._organization_closed or self._organization_thread is None:
                return False
        if notify:
            self._organization_queue.put(True)
        return True

    def _signal_organization_worker(self) -> None:
        with self._organization_lock:
            if self._organization_closed or self._organization_thread is None:
                return
        self._organization_queue.put(True)

    def _run_organization_loop(self) -> None:
        while True:
            token: bool | None = True
            dequeued = False
            jobs: list[dict[str, Any]] = []
            try:
                try:
                    token = self._organization_queue.get(timeout=0.1)
                    dequeued = True
                except Empty:
                    token = True
                if token is None:
                    return
                jobs = self.store.claim_organization_jobs(
                    DEFAULT_ORGANIZATION_BATCH_SIZE,
                    lease_seconds=self._organization_lease_seconds,
                )
                if not jobs:
                    continue
                results = self._organize_stored_jobs(jobs)
                for job in jobs:
                    memory_id = str(job["memory_id"])
                    result = results[memory_id]
                    persona_updated = self._finalize_layered_organization(
                        result.metadata.get("memory_id", memory_id),
                        memory_id,
                    )
                    self.store.complete_organization_job(
                        str(job["job_id"]),
                        {
                            "result_memory_id": result.metadata.get("memory_id", memory_id),
                            "action": result.metadata.get("action"),
                            "relation": result.metadata.get("relation"),
                            "superseded_memory_ids": result.metadata.get(
                                "superseded_memory_ids",
                                [],
                            ),
                            "organization_llm_calls": result.metadata.get(
                                "organization_llm_calls",
                                0,
                            ),
                            "persona_updated": persona_updated,
                        },
                    )
            except Exception as exc:
                for job in jobs:
                    self.store.fail_organization_job(
                        str(job["job_id"]),
                        f"{type(exc).__name__}: {exc}",
                    )
            finally:
                if dequeued:
                    self._organization_queue.task_done()

    def _organize_stored_jobs(
        self,
        jobs: list[dict[str, Any]],
    ) -> dict[str, OperationResult]:
        """Detect relations for one persisted job batch in one model request."""

        batch_ids = {str(job["memory_id"]) for job in jobs}
        prepared: list[MemOSDetectionInput] = []
        prior_ids: set[str] = set()
        immediate: dict[str, OperationResult] = {}
        for job in jobs:
            memory_id = str(job["memory_id"])
            saved = self.store.get(memory_id)
            if saved is None or saved.status != ACTIVE:
                immediate[memory_id] = self._missing_organization_result(memory_id, saved)
                prior_ids.add(memory_id)
                continue
            excluded = batch_ids - prior_ids - {memory_id}
            candidates = self._collect_candidates(saved, exclude_ids=excluded)
            prepared.append(MemOSDetectionInput(saved, tuple(candidates)))
            prior_ids.add(memory_id)

        assert self.conflict_organizer is not None
        detect_batch = getattr(self.conflict_organizer, "detect_batch", None)
        if len(prepared) > 1 and callable(detect_batch):
            detected = detect_batch(prepared)
        else:
            detected = {
                item.memory.memory_id: tuple(
                    self.conflict_organizer.detect(item.memory, item.candidates)
                )
                if item.candidates
                else ()
                for item in prepared
            }

        results = dict(immediate)
        for item in prepared:
            memory_id = item.memory.memory_id
            current = self.store.get(memory_id)
            if current is None or current.status != ACTIVE:
                results[memory_id] = self._missing_organization_result(memory_id, current)
                continue
            active_candidates = [
                candidate
                for candidate in item.candidates
                if (stored := self.store.get(candidate.memory_id)) is not None
                and stored.status == ACTIVE
            ]
            active_ids = {candidate.memory_id for candidate in active_candidates}
            relations = [
                relation
                for relation in detected.get(memory_id, ())
                if relation.memory_id in active_ids
            ]
            results[memory_id] = self._apply_organization_relations(
                current,
                active_candidates,
                relations,
            )
        return results

    def _missing_organization_result(
        self,
        memory_id: str,
        saved: MemoryRecord | None,
    ) -> OperationResult:
        return self._success_result(
            saved or MemoryRecord(memory_id, "", "", "organization target missing"),
            action="skipped",
            semantic_decision="memos_node_no_longer_active",
            context=None,
            organization_llm_calls=0,
        )

    def _organize_stored_record(self, memory_id: str) -> OperationResult:
        """Process one queued node on the single background worker."""

        saved = self.store.get(memory_id)
        if saved is None or saved.status != ACTIVE:
            return self._missing_organization_result(memory_id, saved)

        candidates = self._collect_candidates(saved)
        if not candidates:
            return self._success_result(
                saved,
                action="created",
                semantic_decision="memos_independent",
                context=None,
                relation=MEMOS_INDEPENDENT,
                organization_llm_calls=0,
            )

        assert self.conflict_organizer is not None
        relations = list(self.conflict_organizer.detect(saved, candidates))
        return self._apply_organization_relations(saved, candidates, relations)

    def _apply_organization_relations(
        self,
        saved: MemoryRecord,
        candidates: list[MemoryRecord],
        relations: list[MemOSCandidateRelation],
    ) -> OperationResult:
        assert self.conflict_organizer is not None
        confidence_threshold = float(getattr(self.conflict_organizer, "confidence_threshold", 0.78))
        related_ids = {
            item.memory_id
            for item in relations
            if item.relation in {MEMOS_REDUNDANT, MEMOS_CONTRADICTORY}
            and item.confidence >= confidence_threshold
        }
        related = [candidate for candidate in candidates if candidate.memory_id in related_ids]
        if not related:
            return self._success_result(
                saved,
                action="created",
                semantic_decision="memos_independent",
                context=None,
                relation=MEMOS_INDEPENDENT,
                organization_relations=relations,
                organization_llm_calls=self.conflict_organizer.last_call_count,
            )

        fusion = self.conflict_organizer.fuse(saved, related, relations)
        if fusion.resolved and fusion.content:
            merged = self._build_fused_record(saved, related, fusion.content, fusion.reason)
            merged_saved = self.store.replace_versions([saved, *related], merged)
            superseded_ids = [saved.memory_id, *(item.memory_id for item in related)]
            self._refresh_indexes(superseded_ids, merged_saved)
            return self._success_result(
                merged_saved,
                action="merged",
                semantic_decision="memos_fused",
                context=None,
                relation=_dominant_relation(relations, related_ids),
                superseded_ids=superseded_ids,
                organization_relations=relations,
                organization_llm_calls=self.conflict_organizer.last_call_count,
            )

        kept = self.store.supersede_with_existing(related, saved)
        superseded_ids = [item.memory_id for item in related]
        self._refresh_indexes(superseded_ids, kept)
        return self._success_result(
            kept,
            action="replaced",
            semantic_decision="memos_newer_node_kept",
            context=None,
            relation=_dominant_relation(relations, related_ids),
            superseded_ids=superseded_ids,
            organization_relations=relations,
            organization_llm_calls=self.conflict_organizer.last_call_count,
        )

    def _finalize_layered_organization(
        self,
        result_memory_id: object,
        original_memory_id: str,
    ) -> bool:
        """Update Persona only after durable conflict organization succeeds."""

        record = self.store.get(str(result_memory_id)) or self.store.get(original_memory_id)
        if record is None:
            return False
        self.lifecycle.mark_promoted(record.metadata, record.memory_id)
        finalize = getattr(self.memory_extractor, "finalize_write", None)
        if not callable(finalize):
            return False
        try:
            return bool(
                finalize(
                    PreferenceContext(
                        user_id=record.user_id,
                        project_id=record.project_id,
                        task_id=str(record.metadata.get("task_id", "")),
                    )
                )
            )
        except Exception:
            return False

    def _collect_candidates(
        self,
        record: MemoryRecord,
        *,
        exclude_ids: set[str] | None = None,
    ) -> list[MemoryRecord]:
        """Recall the unified MemOS neighborhood, excluding the new node itself."""

        excluded = exclude_ids or set()
        by_id: dict[str, MemoryRecord] = {}
        for previous in self.store.find_structured_slot(
            record.user_id,
            record.project_id,
            record.type,
            record.subject,
            record.attribute,
            record.scope,
        ):
            if previous.memory_id != record.memory_id and previous.memory_id not in excluded:
                by_id[previous.memory_id] = previous
        vector_failed = False
        try:
            vector_results = self.retriever.search(
                record.content,
                record.user_id,
                record.project_id,
                top_k=self.semantic_candidate_k + len(excluded) + 1,
                filters={"memory_status": ACTIVE},
            )
        except Exception:
            vector_failed = True
            vector_results = []
        for result in vector_results:
            if (
                result.memory_id == record.memory_id
                or result.memory_id in excluded
                or result.score < self.semantic_confirmation_threshold
            ):
                continue
            previous = self.store.get(result.memory_id)
            if previous is not None and previous.status == ACTIVE:
                by_id.setdefault(previous.memory_id, previous)
        if vector_failed:
            for memory_id, _score in self.retriever.fts_store.search(
                record.content,
                record.user_id,
                record.project_id,
                top_k=self.semantic_candidate_k + len(excluded) + 1,
            ):
                if memory_id == record.memory_id or memory_id in excluded:
                    continue
                previous = self.store.get(memory_id)
                if previous is not None and previous.status == ACTIVE:
                    by_id.setdefault(previous.memory_id, previous)
        return list(by_id.values())[: self.semantic_candidate_k]

    def _build_fused_record(
        self,
        current: MemoryRecord,
        related: list[MemoryRecord],
        content: str,
        reason: str,
    ) -> MemoryRecord:
        source_timestamps: list[str] = []
        for item in [current, *related]:
            raw = item.metadata.get("source_timestamps")
            if isinstance(raw, list):
                source_timestamps.extend(str(value) for value in raw)
            source_timestamps.extend((item.effective_at, item.updated_at))
        merged_from_ids = [current.memory_id, *(item.memory_id for item in related)]
        slot_source = _latest_slot_source(current, related)
        return MemoryRecord(
            memory_id=f"mem_{uuid.uuid4().hex[:12]}",
            user_id=current.user_id,
            project_id=current.project_id,
            content=content,
            type=current.type,
            source="memos_conflict_fusion",
            subject=slot_source.subject,
            attribute=slot_source.attribute,
            value=slot_source.value,
            polarity=slot_source.polarity,
            scope=slot_source.scope,
            confidence=max([current.confidence, *(item.confidence for item in related)]),
            effective_at=slot_source.effective_at,
            metadata={
                **current.metadata,
                "organization": "memos_post_write",
                "organization_reason": reason,
                "merged_from_ids": merged_from_ids,
                "authoritative_memory_id": slot_source.memory_id,
                "source_timestamps": list(
                    dict.fromkeys(value for value in source_timestamps if value)
                ),
            },
        )

    def _refresh_indexes(self, superseded_ids: list[str], active: MemoryRecord) -> None:
        self.retriever.update_status(superseded_ids, SUPERSEDED)
        inactive = [
            record
            for memory_id in superseded_ids
            if (record := self.store.get(memory_id)) is not None
        ]
        self.retriever.index_records([*inactive, active])

    def _store_direct(
        self,
        record: MemoryRecord,
        context: AgentOperationContext | None,
    ) -> OperationResult:
        saved, created = self.store.upsert(record)
        self.lifecycle.mark_promoted(record.metadata, saved.memory_id)
        self.retriever.index_records([saved])
        return self._success_result(
            saved,
            action="created" if created else "refreshed",
            semantic_decision="direct_store" if created else "exact_redundant",
            context=context,
            relation=MEMOS_INDEPENDENT if created else MEMOS_REDUNDANT,
            organization_llm_calls=0,
        )

    @staticmethod
    def _combine_results(
        results: list[OperationResult],
        context: AgentOperationContext | None,
        *,
        llm_calls: int,
    ) -> OperationResult:
        if len(results) == 1:
            result = results[0]
            result.metadata["batch_size"] = 1
            result.metadata["organization_llm_calls"] = llm_calls
            return result
        return OperationResult(
            success=all(result.success for result in results),
            message=" ".join(result.message for result in results),
            warnings=[warning for result in results for warning in result.warnings],
            errors=[error for result in results for error in result.errors],
            metadata={
                **_base_metadata(context),
                "memory_ids": [
                    result.metadata.get("memory_id")
                    for result in results
                    if result.metadata.get("memory_id")
                ],
                "operations": [result.metadata for result in results],
                "batch_size": len(results),
                "organization_llm_calls": llm_calls,
            },
        )

    @staticmethod
    def _success_result(
        saved: MemoryRecord,
        *,
        action: str,
        semantic_decision: str,
        context: AgentOperationContext | None,
        relation: str | None = None,
        superseded_ids: list[str] | None = None,
        organization_relations: list[MemOSCandidateRelation] | None = None,
        organization_llm_calls: int = 0,
    ) -> OperationResult:
        return OperationResult(
            success=True,
            message=f"Memory {action}: {saved.content}",
            metadata={
                **_base_metadata(context),
                "memory_id": saved.memory_id,
                "type": saved.type,
                "action": action,
                "semantic_decision": semantic_decision,
                "relation": relation,
                "version": saved.version,
                "supersedes_id": saved.supersedes_id,
                "superseded_memory_ids": superseded_ids or [],
                "organization_relations": [
                    {
                        "memory_id": item.memory_id,
                        "relation": item.relation,
                        "confidence": item.confidence,
                        "reason": item.reason,
                    }
                    for item in organization_relations or []
                ],
                "organization_llm_calls": organization_llm_calls,
            },
        )


def _partition_preference_review_candidates(
    candidates: list[MemoryCandidate],
) -> tuple[list[MemoryCandidate], list[MemoryCandidate]]:
    """Keep unapproved preference slots out of all persistence paths."""

    accepted: list[MemoryCandidate] = []
    review: list[MemoryCandidate] = []
    for candidate in candidates:
        if (
            candidate.type == PREFERENCE_MEMORY
            and candidate.metadata.get("preference_schema_status") == "needs_review"
        ):
            review.append(candidate)
        else:
            accepted.append(candidate)
    return accepted, review


def _preference_review_result(
    candidates: list[MemoryCandidate],
    context: AgentOperationContext | None,
    *,
    layer: str | None = None,
) -> OperationResult:
    result = OperationResult(
        success=True,
        message="Preference candidate requires ontology review; not persisted.",
        warnings=["preference_schema_review_required"],
        metadata={
            **_base_metadata(context),
            "operation": "preference_review_pending",
            "review_candidates": _review_candidate_metadata(candidates),
        },
    )
    if layer:
        result.metadata["layer"] = layer
    return result


def _attach_preference_review_metadata(
    result: OperationResult,
    candidates: list[MemoryCandidate],
) -> OperationResult:
    if not candidates:
        return result
    if "preference_schema_review_required" not in result.warnings:
        result.warnings.append("preference_schema_review_required")
    result.metadata["review_candidates"] = _review_candidate_metadata(candidates)
    return result


def _review_candidate_metadata(
    candidates: list[MemoryCandidate],
) -> list[dict[str, str]]:
    return [
        {
            "subject": candidate.subject,
            "attribute": candidate.attribute,
            "value": candidate.value,
            "polarity": candidate.polarity,
            "scope": candidate.scope,
        }
        for candidate in candidates
    ]


def _dominant_relation(
    relations: list[MemOSCandidateRelation],
    related_ids: set[str],
) -> str:
    selected = [item for item in relations if item.memory_id in related_ids]
    if any(item.relation == MEMOS_CONTRADICTORY for item in selected):
        return MEMOS_CONTRADICTORY
    if any(item.relation == MEMOS_REDUNDANT for item in selected):
        return MEMOS_REDUNDANT
    return MEMOS_INDEPENDENT


def _latest_slot_source(
    current: MemoryRecord,
    related: list[MemoryRecord],
) -> MemoryRecord:
    """Use the latest effective value for every same-slot conflict."""

    if not current.subject.strip() or not current.attribute.strip():
        return current
    same_slot = [
        item
        for item in [current, *related]
        if item.subject.strip().lower() == current.subject.strip().lower()
        and item.attribute.strip().lower() == current.attribute.strip().lower()
        and item.scope.strip().lower() == current.scope.strip().lower()
    ]
    return max(
        same_slot,
        key=lambda item: (
            item.effective_at or item.created_at,
            item.created_at,
            item.version,
        ),
    )


def extract_memory_content(user_input: str) -> str:
    text = " ".join(user_input.strip().split())
    if not text:
        return ""
    patterns = (
        r"^(?:please\s+)?remember(?:\s+that)?[:：,，\s]*(.+)$",
        r"^(?:记住|请记住)[:：,，\s]*(.+)$",
        r"^(?:帮我记住|替我记住)[:：,，\s]*(.+)$",
        r"^(?:以后|之后|下次)[:：,，\s]*(.+)$",
        r"^(?:以后帮我|下次帮我)[:：,，\s]*(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_content(match.group(1))
    return _clean_content(text)


def classify_memory_type(content: str) -> str:
    lowered = content.lower()
    if any(
        word in content
        for word in ("偏好", "偏爱", "喜欢", "不喜欢", "讨厌", "习惯", "风格", "格式")
    ):
        return "preference"
    if any(word in content for word in ("任务", "待办", "截止", "完成", "进度")):
        return TASK_MEMORY
    if any(word in lowered for word in ("task", "todo", "deadline", "completed")):
        return TASK_MEMORY
    if any(word in content for word in ("必须", "不要", "不能", "约束", "限制")):
        return PROJECT_MEMORY
    if any(word in content for word in ("决定", "方案", "路线", "采用")):
        return PROJECT_MEMORY
    if any(word in lowered for word in ("prefer", "style", "format")):
        return "preference"
    return DEFAULT_MEMORY_TYPE


def _clean_content(content: str) -> str:
    return content.strip(" \t\r\n。？！?!,，")


def _base_metadata(context: AgentOperationContext | None) -> dict[str, object]:
    return {
        "memory_query": context.rewritten_queries.memory_query if context else None,
    }


def _tool_messages(
    context: AgentOperationContext | None,
) -> list[tuple[str, str]]:
    if context is None:
        return []
    messages: list[tuple[str, str]] = []
    for result in context.tool_results:
        if not result.success:
            continue
        payload = {
            "message": result.message,
            "metadata": result.metadata,
        }
        messages.append(
            ("tool", json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
        )
    return messages
