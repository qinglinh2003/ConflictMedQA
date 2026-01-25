"""
Adapters to wrap existing retrievers with the BaseRetriever interface.

These adapters allow using the existing ThreeStageRetriever and PyseriniRetriever
with the evaluation pipeline while preserving all metadata.

Example:
    from src.retrieve import ThreeStageAdapter
    from src.eval import RAGSetting
    
    retriever = ThreeStageAdapter(
        index_dir="data/pyserini/index_full",
        stages="full",
    )
    
    setting = RAGSetting(retriever, top_k=10)
"""

from typing import Literal

from .interface import BaseRetriever, Evidence


class ThreeStageAdapter(BaseRetriever):
    """Adapter for ThreeStageRetriever (BM25 → Dense → Cross-encoder).
    
    This wraps the existing ThreeStageRetriever to provide the standard
    BaseRetriever interface with full metadata preservation.
    
    Attributes:
        index_dir: Path to the Pyserini index.
        stages: Retrieval stages to use ("bm25", "bm25+dense", "full").
        bm25_k: Number of BM25 candidates.
        dense_k: Number of dense rerank candidates.
    """
    
    def __init__(
        self,
        index_dir: str,
        stages: Literal["bm25", "bm25+dense", "full"] = "full",
        bm25_k: int = 500,
        dense_k: int = 100,
        device: str = "auto",
    ):
        """Initialize the adapter.
        
        Args:
            index_dir: Path to Pyserini index directory.
            stages: Which stages to run:
                - "bm25": BM25 only (fast, no GPU)
                - "bm25+dense": BM25 + Qwen embedding rerank
                - "full": BM25 + Dense + MedCPT cross-encoder
            bm25_k: Number of BM25 candidates to retrieve.
            dense_k: Number to keep after dense reranking.
            device: Device for neural models ("auto", "cuda", "cpu").
        """
        self.index_dir = index_dir
        self.stages = stages
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.device = device
        
        self._retriever = None
    
    def _ensure_loaded(self):
        """Lazy load the retriever."""
        if self._retriever is not None:
            return
        
        from .retriever import ThreeStageRetriever, RetrieverConfig
        
        config = RetrieverConfig(
            bm25_k=self.bm25_k,
            dense_k=self.dense_k,
            top_k=100,  # We'll limit in retrieve()
        )
        
        self._retriever = ThreeStageRetriever(
            index_dir=self.index_dir,
            config=config,
            device=self.device,
        )
    
    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Retrieve evidence using the three-stage pipeline.
        
        Args:
            query: Search query.
            top_k: Maximum results to return.
        
        Returns:
            List of Evidence objects with full metadata.
        """
        self._ensure_loaded()
        
        # Update config for this query
        self._retriever.config.top_k = top_k
        
        results = self._retriever.search(query, stages=self.stages)
        
        return [
            Evidence(
                text=r.text,
                metadata={
                    "chunk_id": r.chunk_id,
                    "pmcid": r.pmcid,
                    "section": r.section,
                    "score": r.score,
                    "bm25_rank": r.bm25_rank,
                    "dense_rank": r.dense_rank,
                    "retriever": "three_stage",
                    "stages": self.stages,
                }
            )
            for r in results[:top_k]
        ]
    
    def __repr__(self) -> str:
        return f"ThreeStageAdapter(index_dir='{self.index_dir}', stages='{self.stages}')"


class PrecomputedRetriever(BaseRetriever):
    """Retriever that uses pre-computed retrieval results.
    
    Useful when retrieval has already been done offline and results
    are stored in a lookup table (e.g., loaded from JSONL).
    
    Example:
        # Load pre-computed results
        results_map = {}
        with open("retrieved.jsonl") as f:
            for line in f:
                data = json.loads(line)
                results_map[data["Prompt"]] = data["retrieved"]
        
        retriever = PrecomputedRetriever(results_map)
    """
    
    def __init__(self, results_map: dict[str, list[dict]]):
        """Initialize with pre-computed results.
        
        Args:
            results_map: Dict mapping query strings to lists of result dicts.
                Each result dict should have at least "text" key,
                optionally other metadata fields.
        """
        self._results_map = results_map
    
    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Look up pre-computed results for a query.
        
        Args:
            query: Search query (must match exactly).
            top_k: Maximum results to return.
        
        Returns:
            List of Evidence objects, or empty list if query not found.
        """
        results = self._results_map.get(query, [])
        
        evidence_list = []
        for r in results[:top_k]:
            if isinstance(r, str):
                evidence_list.append(Evidence(text=r))
            elif isinstance(r, dict):
                text = r.get("text", "")
                metadata = {k: v for k, v in r.items() if k != "text"}
                evidence_list.append(Evidence(text=text, metadata=metadata))
            else:
                evidence_list.append(Evidence(text=str(r)))
        
        return evidence_list
    
    @classmethod
    def from_jsonl(cls, path: str, query_field: str = "Prompt", results_field: str = "retrieved") -> "PrecomputedRetriever":
        """Load from a JSONL file.
        
        Args:
            path: Path to JSONL file.
            query_field: Field name for the query.
            results_field: Field name for the results list.
        
        Returns:
            PrecomputedRetriever instance.
        """
        import json
        
        results_map = {}
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                query = data.get(query_field, "")
                results = data.get(results_field, [])
                if query:
                    results_map[query] = results
        
        return cls(results_map)
    
    def __repr__(self) -> str:
        return f"PrecomputedRetriever(num_queries={len(self._results_map)})"