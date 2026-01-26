"""
RAG Methods module for ConflictMedQA.

This module provides different RAG (Retrieval-Augmented Generation) strategies
that define HOW to use a retriever to get evidence for a query.

Architecture:
    BaseRetriever (src/retrieve): Low-level retrieval (query → evidence)
    RAGMethod (src/rag): Retrieval strategy (query + retriever → final evidence)
    RAGSetting (src/eval): Eval integration (instance + method → prepared instance)

Available Methods:
    - BaselineRAG: Direct single-query retrieval (simplest)
    - QueryRewriteRAG: LLM rewrites query before retrieval
    - MultiQueryRAG: Expand query to variants, merge with RRF
    - IterativeRAG: Multi-round retrieval with query refinement
    - RAGFusionRAG: Generate multiple queries + reciprocal rank fusion
    - GARRAG: Generation-Augmented Retrieval (answer/title/sentence/context modes)
    - GARContextRAG: HyDE-style hypothetical document generation
    - MADAMRetrieveRAG: Multi-agent document assignment (retrieval only)
    - MADAMDebateWorkflow: Full multi-agent debate workflow (standalone)

Example:
    from src.retrieve import ThreeStageAdapter
    from src.rag import BaselineRAG, GARContextRAG, RAGFusionRAG
    from src.eval import RAGSetting

    # Create retriever
    retriever = ThreeStageAdapter(index_dir="...")

    # Option 1: Baseline RAG
    method = BaselineRAG(retriever, top_k=10)
    evidence = method.retrieve("Does aspirin prevent heart attacks?")

    # Option 2: HyDE-style with GARContextRAG
    method = GARContextRAG(
        retriever=retriever,
        generate_fn=lambda prompt: llm.generate(prompt),
        top_k=10,
    )
    evidence = method.retrieve("Does aspirin prevent heart attacks?")

    # Option 3: RAG Fusion with multi-query
    method = RAGFusionRAG(
        retriever=retriever,
        generate_fn=lambda q, n: generate_queries(q, n),
        num_queries=4,
        top_k=10,
    )
    evidence = method.retrieve("Does aspirin prevent heart attacks?")

    # Use with evaluation
    setting = RAGSetting(method)
    prepared = setting.prepare(instance)
"""

from .base import RAGMethod
from .baseline import BaselineRAG
from .query_rewrite import QueryRewriteRAG
from .multi_query import MultiQueryRAG
from .iterative import IterativeRAG
from .rag_fusion import RAGFusionRAG
from .gar import GARRAG, GARAnswerRAG, GARContextRAG
from .madam import MADAMRetrieveRAG, MADAMDebateWorkflow

__all__ = [
    # Base
    "RAGMethod",
    # Simple methods
    "BaselineRAG",
    "QueryRewriteRAG",
    "MultiQueryRAG",
    "IterativeRAG",
    # Advanced methods (from rag_experiments)
    "RAGFusionRAG",
    "GARRAG",
    "GARAnswerRAG",
    "GARContextRAG",
    "MADAMRetrieveRAG",
    "MADAMDebateWorkflow",
]