"""RAG module: document parsing, chunking, embedding, storage, and retrieval."""

from src.rag_module.factory import create_retrieval_pipeline
from src.rag_module.pipeline.ingestion import IngestionPipeline
from src.rag_module.pipeline.retrieval import RetrievalPipeline
from src.rag_module.retriever.fake import FakeRagRetriever
from src.rag_module.retriever.interface import RagRetriever

__all__ = [
    "FakeRagRetriever",
    "IngestionPipeline",
    "RagRetriever",
    "RetrievalPipeline",
    "create_retrieval_pipeline",
]
