"""
Multi-Query RAG: Expand query into multiple variants, merge results.

Generate multiple query variants, retrieve for each, then merge
and deduplicate results using reciprocal rank fusion or similar.

Workflow:
    query → LLM expand to [q1, q2, ...] → retrieve each → merge/dedup → evidence
"""

from typing import Callable, TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


class MultiQueryRAG(RAGMethod):
    """Multi-Query RAG with query expansion and fusion.
    
    Example:
        method = MultiQueryRAG(
            retriever=retriever,
            expand_fn=lambda q: llm(f"Generate 3 search queries for: {q}").split("\\n"),
            top_k=10,
            fusion="rrf",
        )
    """
    
    def __init__(
        self,
        retriever: "BaseRetriever",
        expand_fn: Callable[[str], list[str]],
        top_k: int = 10,
        fusion: str = "rrf",  # "rrf" or "score"
        include_original: bool = True,
    ):
        """Initialize Multi-Query RAG.
        
        Args:
            retriever: Underlying retriever.
            expand_fn: Function to expand query into multiple variants.
            top_k: Number of final results after fusion.
            fusion: Fusion method ("rrf" for reciprocal rank fusion, "score" for max score).
            include_original: Whether to include original query in retrieval.
        """
        super().__init__(retriever, top_k)
        self.expand_fn = expand_fn
        self.fusion = fusion
        self.include_original = include_original
    
    @property
    def name(self) -> str:
        return f"multi_query_{self.fusion}_top{self.top_k}"
    
    def retrieve(self, query: str) -> list["Evidence"]:
        """Expand query, retrieve for each variant, merge results."""
        # Generate query variants
        variants = self.expand_fn(query)
        if self.include_original:
            variants = [query] + variants
        
        # Retrieve for each variant
        all_results: list[list] = []
        for v in variants:
            results = self.retriever.retrieve(v, top_k=self.top_k * 2)
            all_results.append(results)
        
        # Merge using fusion method
        if self.fusion == "rrf":
            merged = self._reciprocal_rank_fusion(all_results)
        else:
            merged = self._max_score_fusion(all_results)
        
        # Add expansion info to metadata
        for e in merged[:self.top_k]:
            e.metadata["original_query"] = query
            e.metadata["query_variants"] = variants
        
        return merged[:self.top_k]
    
    def _reciprocal_rank_fusion(
        self, 
        result_lists: list[list["Evidence"]], 
        k: int = 60
    ) -> list["Evidence"]:
        """Merge results using Reciprocal Rank Fusion."""
        from ..retrieve.interface import Evidence
        
        scores: dict[str, float] = {}
        evidence_map: dict[str, Evidence] = {}
        
        for results in result_lists:
            for rank, e in enumerate(results):
                key = e.text[:500]
                if key not in evidence_map:
                    evidence_map[key] = e
                    scores[key] = 0.0
                scores[key] += 1.0 / (k + rank + 1)
        
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        result = []
        for key in sorted_keys:
            e = evidence_map[key]
            e.metadata["rrf_score"] = scores[key]
            result.append(e)
        
        return result
    
    def _max_score_fusion(self, result_lists: list[list["Evidence"]]) -> list["Evidence"]:
        """Merge results by taking max score for each document."""
        from ..retrieve.interface import Evidence
        
        scores: dict[str, float] = {}
        evidence_map: dict[str, Evidence] = {}
        
        for results in result_lists:
            for e in results:
                key = e.text[:500]
                score = e.metadata.get("score", 0.0)
                if key not in evidence_map or score > scores[key]:
                    evidence_map[key] = e
                    scores[key] = score
        
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [evidence_map[key] for key in sorted_keys]