"""
Retrieval module for ConflictMedQA.

This module provides low-level retrieval functionality with a unified interface.

Components:
- BaseRetriever: Abstract interface for all retrievers
- Evidence: Data class for retrieved evidence with metadata
- Adapters: Wrappers for existing retrievers (ThreeStage, Pyserini)
- Utilities: Batch retrieval, JSONL I/O

For RAG strategies (how to USE retrievers), see src/rag/ module.

Quick Start:
    from src.retrieve import ThreeStageAdapter
    
    retriever = ThreeStageAdapter(
        index_dir="data/pyserini/index_full",
        stages="full",
    )
    evidence = retriever.retrieve("Does aspirin prevent heart attacks?")
    
    # For RAG methods, use src/rag:
    from src.rag import BaselineRAG
    method = BaselineRAG(retriever, top_k=10)
"""

from .interface import (
    Evidence,
    BaseRetriever,
    FunctionRetriever,
)

from .adapters import (
    ThreeStageAdapter,
    PrecomputedRetriever,
)

from .utils import (
    batch_retrieve_to_jsonl,
    load_retrieved_jsonl,
    convert_retrieved_to_instances,
    merge_shards,
)

__all__ = [
    # Core interface
    "Evidence",
    "BaseRetriever",
    "FunctionRetriever",
    # Adapters
    "ThreeStageAdapter",
    "PrecomputedRetriever",
    # Utilities
    "batch_retrieve_to_jsonl",
    "load_retrieved_jsonl",
    "convert_retrieved_to_instances",
    "merge_shards",
]