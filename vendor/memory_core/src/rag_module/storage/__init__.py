"""Storage layer for RAG module."""

from src.rag_module.storage.chunk_repo import ChunkRecord, ChunkRepository
from src.rag_module.storage.document_repo import DocumentRecord, DocumentRepository
from src.rag_module.storage.schema import init_database
from src.rag_module.storage.vector.base import VectorHit, VectorStore
from src.rag_module.storage.vector.factory import close_vector_store, create_vector_store
from src.rag_module.storage.vector.kylin_engine import KylinVectorStore
from src.rag_module.storage.vector.sqlite_vec import SqliteVecStore

__all__ = [
    "ChunkRecord",
    "ChunkRepository",
    "DocumentRecord",
    "DocumentRepository",
    "SqliteVecStore",
    "KylinVectorStore",
    "VectorHit",
    "VectorStore",
    "init_database",
    "create_vector_store",
    "close_vector_store",
]
