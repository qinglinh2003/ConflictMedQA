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
    - HyDERAG: Generate hypothetical document, use as query
    - MultiQueryRAG: Expand query to variants, merge with RRF
    - IterativeRAG: Multi-round retrieval with query refinement

Example:
    from src.retrieve import ThreeStageAdapter
    from src.rag import BaselineRAG, HyDERAG
    from src.eval import RAGSetting
    
    # Create retriever
    retriever = ThreeStageAdapter(index_dir="...")
    
    # Option 1: Baseline RAG
    method = BaselineRAG(retriever, top_k=10)
    evidence = method.retrieve("Does aspirin prevent heart attacks?")
    
    # Option 2: HyDE RAG with LLM
    method = HyDERAG(
        retriever=retriever,
        generate_fn=lambda q: llm(f"Write a passage answering: {q}"),
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
from .hyde import HyDERAG
from .multi_query import MultiQueryRAG
from .iterative import IterativeRAG

__all__ = [
    "RAGMethod",
    "BaselineRAG",
    "QueryRewriteRAG",
    "HyDERAG",
    "MultiQueryRAG",
    "IterativeRAG",
]