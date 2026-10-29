"""Ingestion pipeline: parse → chunk → embed → store with versioning."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from src.rag_module.chunking.base import Chunker
from src.rag_module.embedding.base import EmbeddingProvider
from src.rag_module.parser.base import DocumentParser
from src.rag_module.storage.chunk_repo import ChunkRepository
from src.rag_module.storage.document_repo import DocumentRepository
from src.rag_module.storage.vector.base import VectorStore


class IngestionPipeline:
    """Orchestrate document ingestion with state machine and versioning."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        parsers: list[DocumentParser],
        chunker: Chunker,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._conn = conn
        self._parsers = {ext: p for p in parsers for ext in p.supported_extensions}
        self._chunker = chunker
        self._embedding = embedding_provider
        self._vectors = vector_store
        self._doc_repo = DocumentRepository(conn)
        self._chunk_repo = ChunkRepository(conn)

    def ingest(self, file_path: Path, project_id: str) -> int:
        """Ingest a file with versioning support. Returns doc_pk."""
        ext = file_path.suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            raise ValueError(f"No parser for extension: {ext}")

        content_hash = self._compute_file_hash(file_path)
        existing = self._doc_repo.get_by_filename(project_id, file_path.name)

        if existing and existing.status == "ready":
            if existing.content_hash == content_hash:
                return existing.doc_pk
            return self._rebuild(file_path, project_id, existing, parser, content_hash)

        if existing and existing.status == "processing":
            self._doc_repo.deactivate(existing.doc_pk)
            self._conn.commit()

        return self._create_new(file_path, project_id, parser, content_hash)

    def _create_new(
        self,
        file_path: Path,
        project_id: str,
        parser: DocumentParser,
        content_hash: str,
    ) -> int:
        doc_record = self._doc_repo.create(
            project_id=project_id,
            filename=file_path.name,
            content_hash=content_hash,
            source_uri=str(file_path),
            file_type=file_path.suffix.lower(),
            parser_version=type(parser).__name__,
            chunker_version=self._get_chunker_version(),
        )
        doc_pk = doc_record.doc_pk

        try:
            parsed = parser.parse(file_path)
            chunks = self._chunker.chunk(parsed, doc_record.doc_id, project_id)
            chunk_records = self._chunk_repo.create_batch(chunks, doc_pk)
            self._doc_repo.update_total_chunks(doc_pk, len(chunks))
            self._conn.commit()

            texts = [
                f"{c.heading_path}\n{c.content}" if c.heading_path else c.content
                for c in chunks
            ]
            embeddings = self._embedding.embed_documents(texts)

            entries = [
                (cr.chunk_pk, emb, project_id, doc_pk, True)
                for cr, emb in zip(chunk_records, embeddings)
            ]
            self._vectors.upsert_batch(entries)
            self._chunk_repo.update_embedding_status_batch(
                [cr.chunk_pk for cr in chunk_records], "ready"
            )
            self._doc_repo.update_status(doc_pk, "ready")
            self._conn.commit()

        except Exception as exc:
            self._doc_repo.update_status(doc_pk, "failed", str(exc))
            self._conn.commit()
            raise

        return doc_pk

    def _rebuild(
        self,
        file_path: Path,
        project_id: str,
        old_doc,
        parser: DocumentParser,
        new_content_hash: str,
    ) -> int:
        """Rebuild document: create new version, atomically switch."""
        old_doc_pk = old_doc.doc_pk
        old_chunks = self._chunk_repo.list_by_doc(old_doc_pk)
        old_hashes = {c.content_hash: c for c in old_chunks}

        new_doc_record = self._doc_repo.create(
            project_id=project_id,
            filename=file_path.name,
            content_hash=new_content_hash,
            source_uri=str(file_path),
            file_type=file_path.suffix.lower(),
            parser_version=type(parser).__name__,
            chunker_version=self._get_chunker_version(),
            version=old_doc.version + 1,
        )
        new_doc_pk = new_doc_record.doc_pk

        try:
            parsed = parser.parse(file_path)
            new_chunks = self._chunker.chunk(parsed, new_doc_record.doc_id, project_id)

            reuse_count = 0
            new_count = 0
            for chunk in new_chunks:
                if chunk.content_hash in old_hashes:
                    reuse_count += 1
                else:
                    new_count += 1

            chunk_records = self._chunk_repo.create_batch(new_chunks, new_doc_pk)
            self._doc_repo.update_total_chunks(new_doc_pk, len(new_chunks))
            self._conn.commit()

            chunks_to_embed = [
                (cr, chunk)
                for cr, chunk in zip(chunk_records, new_chunks)
                if chunk.content_hash not in old_hashes
            ]

            if chunks_to_embed:
                texts = [
                    f"{c.heading_path}\n{c.content}" if c.heading_path else c.content
                    for _, c in chunks_to_embed
                ]
                embeddings = self._embedding.embed_documents(texts)
                entries = [
                    (cr.chunk_pk, emb, project_id, new_doc_pk, True)
                    for cr, emb in zip([cr for cr, _ in chunks_to_embed], embeddings)
                ]
                self._vectors.upsert_batch(entries)

            self._chunk_repo.update_embedding_status_batch(
                [cr.chunk_pk for cr in chunk_records], "ready"
            )

            self._vectors.deactivate_by_doc(old_doc_pk)
            self._doc_repo.deactivate(old_doc_pk)
            self._doc_repo.update_status(new_doc_pk, "ready")
            self._conn.commit()

        except Exception as exc:
            self._doc_repo.update_status(new_doc_pk, "failed", str(exc))
            self._doc_repo.update_status(old_doc_pk, "ready")
            self._conn.commit()
            raise

        return new_doc_pk

    def _compute_file_hash(self, file_path: Path) -> str:
        h = hashlib.sha256()
        h.update(file_path.read_bytes())
        return h.hexdigest()

    def _get_chunker_version(self) -> str:
        config = getattr(self._chunker, "config", None)
        return config.chunker_version if config else "unknown"
