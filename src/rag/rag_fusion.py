"""
RAG Fusion: Multi-query retrieval with reciprocal rank fusion.

Paper: https://arxiv.org/abs/2402.03367

Algorithm:
1. Generate multiple search queries from the original query using LLM
2. Retrieve documents for EACH generated query
3. Apply reciprocal rank fusion to combine and rerank results
4. Return fused evidence list

Adapted from rag_experiments/workflows/rag_fusion.py
"""

from typing import Callable, TYPE_CHECKING

from .base import RAGMethod

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


# Default prompt for query generation
QUERY_GENERATION_PROMPT = """You are a helpful assistant that generates multiple search queries based on a single input query.

Generate multiple search queries related to: {query}

OUTPUT ({num_queries} queries):"""


class RAGFusionRAG(RAGMethod):
    """RAG Fusion: Multi-query generation + reciprocal rank fusion.

    This method generates multiple search queries from the original query,
    retrieves documents for each, then fuses results using RRF scoring.

    Example:
        def generate_queries(query: str, num: int) -> list[str]:
            prompt = f"Generate {num} search queries for: {query}"
            response = llm.generate(prompt)
            return response.strip().split("\\n")

        method = RAGFusionRAG(
            retriever=retriever,
            generate_fn=generate_queries,
            num_queries=4,
            top_k=10,
        )
        evidence = method.retrieve("Does aspirin prevent heart attacks?")
    """

    def __init__(
        self,
        retriever: "BaseRetriever",
        generate_fn: Callable[[str, int], list[str]],
        num_queries: int = 4,
        top_k: int = 10,
        per_query_k: int = 5,
        rrf_k: int = 60,
        include_original: bool = False,
    ):
        """Initialize RAG Fusion.

        Args:
            retriever: Underlying retriever.
            generate_fn: Function to generate queries. Signature: (query, num_queries) -> list[str]
            num_queries: Number of queries to generate.
            top_k: Number of final results after fusion.
            per_query_k: Documents to retrieve per generated query.
            rrf_k: RRF constant (default 60). Mitigates impact of high rankings.
            include_original: Whether to also retrieve for the original query.
        """
        super().__init__(retriever, top_k)
        self.generate_fn = generate_fn
        self.num_queries = num_queries
        self.per_query_k = per_query_k
        self.rrf_k = rrf_k
        self.include_original = include_original

    @property
    def name(self) -> str:
        return f"rag_fusion_q{self.num_queries}_top{self.top_k}"

    def retrieve(self, query: str) -> list["Evidence"]:
        """Generate queries, retrieve for each, and fuse with RRF."""
        # Step 1: Generate multiple search queries
        generated_queries = self._generate_queries(query)

        # Optionally include original query
        queries_to_search = generated_queries.copy()
        if self.include_original:
            queries_to_search.insert(0, query)

        # Step 2: Retrieve documents for each query
        all_doc_lists: list[list] = []
        for gen_query in queries_to_search:
            docs = self.retriever.retrieve(gen_query, top_k=self.per_query_k)
            all_doc_lists.append(docs)

        # Step 3: Apply reciprocal rank fusion
        fused_docs = self._reciprocal_rank_fusion(all_doc_lists)

        # Add metadata about the fusion process
        for e in fused_docs[:self.top_k]:
            e.metadata["original_query"] = query
            e.metadata["generated_queries"] = generated_queries
            e.metadata["rag_method"] = self.name

        return fused_docs[:self.top_k]

    def _generate_queries(self, original_query: str) -> list[str]:
        """Generate multiple search queries from the original query."""
        raw_queries = self.generate_fn(original_query, self.num_queries)

        # Clean up the queries (remove numbering, empty lines, etc.)
        queries = []
        for line in raw_queries:
            line = line.strip()
            if line and len(line) > 2:
                # Strip leading numbers, dots, dashes, parentheses
                cleaned = line.lstrip("0123456789.-) ").strip()
                if cleaned:
                    queries.append(cleaned)

        return queries[:self.num_queries]

    def _reciprocal_rank_fusion(
        self,
        doc_lists: list[list["Evidence"]]
    ) -> list["Evidence"]:
        """
        Combine multiple ranked lists using reciprocal rank fusion.

        RRF score: score(d) = Σ 1/(k + rank(d)) across all ranking lists
        where rank is 0-indexed position in each list.

        Args:
            doc_lists: List of evidence lists from different queries.

        Returns:
            Fused and re-ranked evidence list.
        """
        from ..retrieve.interface import Evidence

        fused_scores: dict[str, float] = {}
        doc_map: dict[str, Evidence] = {}

        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list):
                # Use first 500 chars of text as unique identifier
                doc_id = doc.text[:500]

                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                    fused_scores[doc_id] = 0.0

                # RRF formula: 1 / (rank + k)
                fused_scores[doc_id] += 1.0 / (rank + self.rrf_k)

        # Sort by fused score (descending)
        sorted_doc_ids = sorted(
            fused_scores.keys(),
            key=lambda x: fused_scores[x],
            reverse=True
        )

        # Build result list with RRF scores
        result = []
        for doc_id in sorted_doc_ids:
            doc = doc_map[doc_id]
            # Create new Evidence with updated metadata
            fused_doc = Evidence(
                text=doc.text,
                metadata={
                    **doc.metadata,
                    "rrf_score": fused_scores[doc_id],
                },
            )
            result.append(fused_doc)

        return result