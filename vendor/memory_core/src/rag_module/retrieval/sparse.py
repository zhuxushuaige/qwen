"""Sparse retriever using SQLite FTS5 full-text search."""

from __future__ import annotations

import json
import re
import sqlite3

from src.rag_module.retrieval.models import RagCandidate

_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "什么", "如何", "分别", "是否", "哪些", "哪个", "怎么", "怎样", "为什么", "多少",
    "什么", "这个", "那个", "这些", "那些", "吗", "呢", "吧", "啊", "呀", "嘛",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "not", "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some", "such",
    "than", "too", "very", "just", "because", "if", "when", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those", "it",
    "its", "he", "she", "they", "them", "their", "his", "her", "my", "your",
    "our", "we", "you", "i", "me", "us",
})

_MAX_QUERY_TERMS = 48

_CJK_RANGE = r"\u4e00-\u9fff"
_CJK_PATTERN = re.compile(f"[{_CJK_RANGE}]+")
_WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")
_CLAUSE_PATTERN = re.compile(
    r"[，,。！？!?；;：:\n]+|(?:以及|并且|同时|另外|此外|还有|然后|分别)"
)
_QUESTION_NOISE = re.compile(r"(?:请问|请说明|请解释|是什么|有哪些|哪些|如何|怎么|怎样|为什么)")


class SparseRetriever:
    """Retrieve candidates using FTS5 full-text search."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def retrieve(
        self,
        query: str,
        project_id: str,
        candidate_k: int = 20,
    ) -> list[RagCandidate]:
        fts_query = self._build_fts_query(query)
        if not fts_query:
            return []

        rows = self._conn.execute(
            """
            SELECT
                c.chunk_pk,
                c.chunk_id,
                c.doc_pk,
                c.project_id,
                c.content,
                c.heading_path,
                c.page_start,
                c.page_end,
                c.chunk_index,
                c.metadata,
                bm25(chunks_fts) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_pk = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
              AND c.project_id = ?
              AND c.status = 'active'
              AND c.embedding_status = 'ready'
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, project_id, candidate_k),
        ).fetchall()

        candidates: list[RagCandidate] = []
        for row in rows:
            candidates.append(
                RagCandidate(
                    chunk_pk=row["chunk_pk"],
                    chunk_id=row["chunk_id"],
                    doc_pk=row["doc_pk"],
                    doc_id="",
                    project_id=row["project_id"],
                    content=row["content"],
                    heading_path=row["heading_path"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    chunk_index=row["chunk_index"],
                    sparse_score=-row["rank"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )
        return candidates

    def _build_fts_query(self, query: str) -> str:
        tokens = self._tokenize(query)
        if not tokens:
            return ""
        return " OR ".join(f'"{t}"' for t in tokens)

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        clauses = [part.strip() for part in _CLAUSE_PATTERN.split(query) if part.strip()]
        if not clauses:
            clauses = [query]
        clause_tokens = [SparseRetriever._tokenize_clause(clause) for clause in clauses]
        return SparseRetriever._round_robin(clause_tokens, _MAX_QUERY_TERMS)

    @staticmethod
    def _tokenize_clause(query: str) -> list[str]:
        tokens: list[str] = []
        seen: set[str] = set()

        for word in _WORD_PATTERN.findall(query):
            lower = word.lower()
            if lower not in _STOP_WORDS and len(lower) > 1:
                if lower not in seen:
                    seen.add(lower)
                    tokens.append(lower)

        cleaned = _QUESTION_NOISE.sub("", query)
        for match in _CJK_PATTERN.finditer(cleaned):
            segment = match.group()
            ngrams = [segment[i : i + 3] for i in range(len(segment) - 2)]
            # Alternate from the beginning and end so a single long clause does
            # not lose its final condition when the global term cap is reached.
            for ngram in SparseRetriever._edge_balanced(ngrams):
                if ngram not in _STOP_WORDS and ngram not in seen:
                    seen.add(ngram)
                    tokens.append(ngram)
        return tokens

    @staticmethod
    def _edge_balanced(tokens: list[str]) -> list[str]:
        result: list[str] = []
        left, right = 0, len(tokens) - 1
        while left <= right:
            result.append(tokens[left])
            left += 1
            if left <= right:
                result.append(tokens[right])
                right -= 1
        return result

    @staticmethod
    def _round_robin(groups: list[list[str]], limit: int) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        positions = [0] * len(groups)
        while len(result) < limit:
            progressed = False
            for index, group in enumerate(groups):
                while positions[index] < len(group) and group[positions[index]] in seen:
                    positions[index] += 1
                if positions[index] >= len(group):
                    continue
                value = group[positions[index]]
                positions[index] += 1
                seen.add(value)
                result.append(value)
                progressed = True
                if len(result) >= limit:
                    break
            if not progressed:
                break
        return result
