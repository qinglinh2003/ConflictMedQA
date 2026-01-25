"""
HyDE RAG: Hypothetical Document Embeddings.

Generate a hypothetical answer document, then use it as the query.
This helps bridge the gap between question and document embeddings.

Workflow:
    query → LLM generate hypothetical doc → retriever.retrieve(hyde_doc) → evidence

Reference: https://arxiv.org/abs/2212.10496
"""

from typing import Callable, TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


class HyDERAG(RAGMethod):
    """HyDE RAG: Hypothetical Document Embeddings.
    
    Example:
        method = HyDERAG(
            retriever=retriever,
            generate_fn=lambda q: llm(f"Write a passage answering: {q}"),
            top_k=10,
        )
    """
    
    def __init__(
        self,
        retriever: "BaseRetriever",
        generate_fn: Callable[[str], str],
        top_k: int = 10,
    ):
        """Initialize HyDE RAG.
        
        Args:
            retriever: Underlying retriever.
            generate_fn: Function to generate hypothetical document.
            top_k: Number of results.
        """
        super().__init__(retriever, top_k)
        self.generate_fn = generate_fn
    
    @property
    def name(self) -> str:
        return f"hyde_top{self.top_k}"
    
    def retrieve(self, query: str) -> list["Evidence"]:
        """Generate hypothetical doc then retrieve."""
        hyde_doc = self.generate_fn(query)
        evidence = self.retriever.retrieve(hyde_doc, top_k=self.top_k)
        
        # Add HyDE info to metadata
        for e in evidence:
            e.metadata["original_query"] = query
            e.metadata["hyde_doc"] = hyde_doc[:500]  # Truncate for storage
        
        return evidence