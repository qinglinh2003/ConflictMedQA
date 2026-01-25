"""
Query Rewrite RAG: Use LLM to rewrite query before retrieval.

Workflow:
    query → LLM rewrite → retriever.retrieve(rewritten) → evidence
"""

from typing import Callable, TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


class QueryRewriteRAG(RAGMethod):
    """Query Rewrite RAG.
    
    Example:
        method = QueryRewriteRAG(
            retriever=retriever,
            rewrite_fn=lambda q: llm(f"Rewrite for medical search: {q}"),
            top_k=10,
        )
    """
    
    def __init__(
        self, 
        retriever: "BaseRetriever", 
        rewrite_fn: Callable[[str], str],
        top_k: int = 10,
    ):
        """Initialize query rewrite RAG.
        
        Args:
            retriever: Underlying retriever.
            rewrite_fn: Function to rewrite query (e.g., LLM call).
            top_k: Number of results.
        """
        super().__init__(retriever, top_k)
        self.rewrite_fn = rewrite_fn
    
    @property
    def name(self) -> str:
        return f"query_rewrite_top{self.top_k}"
    
    def retrieve(self, query: str) -> list["Evidence"]:
        """Rewrite query then retrieve."""
        rewritten = self.rewrite_fn(query)
        evidence = self.retriever.retrieve(rewritten, top_k=self.top_k)
        
        # Add rewrite info to metadata
        for e in evidence:
            e.metadata["original_query"] = query
            e.metadata["rewritten_query"] = rewritten
        
        return evidence