"""Query expansion for improved RAG retrieval."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class LLMProvider(Protocol):
    """Protocol for LLM providers used in query expansion."""

    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        ...


class QueryExpander(ABC):
    """Base protocol for query expanders."""

    @abstractmethod
    def expand(self, query: str, max_variants: int = 5) -> list[str]:
        """Expand a query into multiple variants.

        Args:
            query: The original query.
            max_variants: Maximum number of query variants to return.

        Returns:
            List of query variants including the original query.
        """
        ...


class HyDEExpander(QueryExpander):
    """Hypothetical Document Embedding expander.

    Generates a hypothetical document that would answer the query,
    then uses that document for retrieval.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    def expand(self, query: str, max_variants: int = 5) -> list[str]:
        """Expand query using HyDE.

        If no LLM is available, returns only the original query.
        """
        variants = [query]

        if self._llm is None:
            return variants

        prompt = (
            f"Write a short passage that would be a perfect answer to the following question. "
            f"Focus on providing factual information that would be found in a document.\n\n"
            f"Question: {query}\n\n"
            f"Answer passage:"
        )

        try:
            hypothetical_doc = self._llm.generate(prompt)
            if hypothetical_doc and hypothetical_doc.strip():
                variants.append(hypothetical_doc.strip())
        except Exception:
            pass

        return variants[:max_variants]


class MultiQueryExpander(QueryExpander):
    """Multi-query expander.

    Generates multiple query variants by rewriting the original query
    in different ways to improve recall.
    """

    def __init__(self, llm: LLMProvider | None = None) -> None:
        self._llm = llm

    def expand(self, query: str, max_variants: int = 5) -> list[str]:
        """Expand query into multiple variants.

        If no LLM is available, returns only the original query.
        """
        variants = [query]

        if self._llm is None:
            return variants

        prompt = (
            f"You are an AI assistant helping with information retrieval. "
            f"Generate {max_variants - 1} different versions of the following query "
            f"that would help find the same information. Each version should use "
            f"different wording or perspective.\n\n"
            f"Original query: {query}\n\n"
            f"Return the variants as a numbered list, one per line:"
        )

        try:
            response = self._llm.generate(prompt)
            if response:
                for line in response.strip().split("\n"):
                    line = line.strip()
                    # Remove numbering like "1." or "1)"
                    if line and line[0].isdigit():
                        line = line.split(".", 1)[-1].strip() if "." in line else line.split(")", 1)[-1].strip()
                    if line and line != query:
                        variants.append(line)
        except Exception:
            pass

        return variants[:max_variants]
