"""
Iterative RAG: Multi-round retrieval with query refinement.

Retrieve, analyze results, refine query, retrieve again.
Useful for complex queries that benefit from iterative exploration.

Workflow:
    query → retrieve → analyze → refine query → retrieve → ... → merge → evidence
"""

from typing import Callable, TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


class IterativeRAG(RAGMethod):
    """Iterative RAG with multi-round retrieval.
    
    Example:
        method = IterativeRAG(
            retriever=retriever,
            refine_fn=lambda q, docs: llm(f"Given docs, refine query: {q}"),
            num_rounds=2,
            top_k=10,
        )
    """
    
    def __init__(
        self,
        retriever: "BaseRetriever",
        refine_fn: Callable[[str, list[str]], str],
        num_rounds: int = 2,
        top_k: int = 10,
        per_round_k: int = 5,
    ):
        """Initialize Iterative RAG.
        
        Args:
            retriever: Underlying retriever.
            refine_fn: Function to refine query given current docs.
            num_rounds: Number of retrieval rounds.
            top_k: Final number of results.
            per_round_k: Results to retrieve per round.
        """
        super().__init__(retriever, top_k)
        self.refine_fn = refine_fn
        self.num_rounds = num_rounds
        self.per_round_k = per_round_k
    
    @property
    def name(self) -> str:
        return f"iterative_r{self.num_rounds}_top{self.top_k}"
    
    def retrieve(self, query: str) -> list["Evidence"]:
        """Multi-round retrieval with refinement."""
        all_evidence: list = []
        current_query = query
        queries_used = [query]
        
        for round_idx in range(self.num_rounds):
            # Retrieve
            results = self.retriever.retrieve(current_query, top_k=self.per_round_k)
            
            # Add round info
            for e in results:
                e.metadata["retrieval_round"] = round_idx
                e.metadata["round_query"] = current_query
            
            all_evidence.extend(results)
            
            # Refine query for next round (if not last)
            if round_idx < self.num_rounds - 1:
                doc_texts = [e.text for e in results]
                current_query = self.refine_fn(query, doc_texts)
                queries_used.append(current_query)
        
        # Deduplicate by text
        seen = set()
        unique_evidence = []
        for e in all_evidence:
            key = e.text[:500]
            if key not in seen:
                seen.add(key)
                e.metadata["original_query"] = query
                e.metadata["queries_used"] = queries_used
                unique_evidence.append(e)
        
        return unique_evidence[:self.top_k]