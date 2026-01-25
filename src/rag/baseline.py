"""
Baseline RAG: Direct single-query retrieval.

This is the simplest RAG method - just use the query directly
to retrieve evidence. No query transformation or expansion.

Workflow:
    query → retriever.retrieve(query) → evidence
"""

from typing import TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


class BaselineRAG(RAGMethod):
    """Baseline RAG: Direct single-query retrieval.
    
    Example:
        retriever = ThreeStageAdapter(index_dir="...")
        method = BaselineRAG(retriever, top_k=10)
        evidence = method.retrieve("Does aspirin prevent heart attacks?")
    """
    
    @property
    def name(self) -> str:
        return f"baseline_top{self.top_k}"
    
    def retrieve(self, query: str) -> list["Evidence"]:
        """Direct retrieval with no transformation."""
        return self.retriever.retrieve(query, top_k=self.top_k)
    
    def retrieve_batch(self, queries: list[str]) -> list[list["Evidence"]]:
        """Batch retrieval using underlying retriever's batch method."""
        return self.retriever.retrieve_batch(queries, top_k=self.top_k)