"""Factory for creating RAG components with sensible defaults."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from src.rag_module.embedding.base import EmbeddingProvider
from src.rag_module.pipeline.retrieval import RetrievalPipeline
from src.rag_module.rerank.base import Reranker
from src.rag_module.rerank.noop_reranker import NoopReranker
from src.rag_module.retrieval.query_expansion import MultiQueryExpander
from src.rag_module.retrieval.rejection import load_rejection_config
from src.rag_module.storage.chunk_repo import ChunkRepository
from src.rag_module.storage.document_repo import DocumentRepository
from src.rag_module.storage.schema import init_database
from src.rag_module.storage.vector.factory import create_vector_store

DEFAULT_REJECTION_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "evaluation"
    / "rag_rejection_kylin_r14.json"
)


def _resolve_embedding_provider() -> EmbeddingProvider:
    """Return the configured provider, defaulting to the Kylin embedding SDK."""

    from src.rag_module.embedding.registry import create_default_embedding_provider

    return create_default_embedding_provider()


def _resolve_reranker(use_reranker: bool) -> Reranker:
    """Return the default OpenVINO mMiniLM reranker, else NoopReranker."""
    if not use_reranker:
        return NoopReranker()
    try:
        from src.rag_module.rerank.openvino_mminilm_reranker import (
            DEFAULT_OPENVINO_MMINILM_RERANKER_MODEL_PATH,
            OpenVINOMMiniLMReranker,
        )

        return OpenVINOMMiniLMReranker(
            model_id=os.environ.get(
                "RAG_RERANKER_MODEL",
                DEFAULT_OPENVINO_MMINILM_RERANKER_MODEL_PATH,
            ),
            batch_size=1,
            max_length=512,
        )
    except ImportError:
        return NoopReranker()


def create_retrieval_pipeline(
    db_path: str | Path = ".rag_module.sqlite3",
    embedding_provider: EmbeddingProvider | None = None,
    use_reranker: bool = True,
    use_query_expansion: bool = True,
    use_rejection: bool = True,
    rejection_config_path: str | Path | None = None,
    vector_store_backend: str | None = None,
    vector_store_fallback: str | None = None,
) -> RetrievalPipeline:
    """Create a RetrievalPipeline backed by SQLite.

    This is the recommended way to get a RagRetriever for the orchestrator.
    Defaults to the Kylin embedding SDK, OpenVINO mMiniLM (falls back to
    NoopReranker), and
    MultiQueryExpander (no LLM by default, returns original query only).
    """
    db_str = str(db_path)
    conn = sqlite3.connect(db_str)
    init_database(conn)

    provider = embedding_provider or _resolve_embedding_provider()
    reranker = _resolve_reranker(use_reranker)
    vector_store = create_vector_store(
        conn,
        dimension=provider.info.dimension,
        backend=vector_store_backend,
        fallback=vector_store_fallback,
    )
    chunk_repo = ChunkRepository(conn)
    doc_repo = DocumentRepository(conn)

    query_expander = MultiQueryExpander() if use_query_expansion else None
    configured_path = Path(
        rejection_config_path
        or os.environ.get("RAG_REJECTION_CONFIG", DEFAULT_REJECTION_CONFIG_PATH)
    )
    rejection_config = (
        load_rejection_config(configured_path)
        if use_rejection and configured_path.is_file()
        else None
    )

    return RetrievalPipeline(
        embedding_provider=provider,
        vector_store=vector_store,
        chunk_repo=chunk_repo,
        doc_repo=doc_repo,
        conn=conn,
        reranker=reranker,
        query_expander=query_expander,
        rejection_config=rejection_config,
        rerank_candidate_k=15,
    )
