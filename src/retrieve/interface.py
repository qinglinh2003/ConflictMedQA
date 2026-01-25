"""
Base retriever interface for ConflictMedQA.

This module defines the abstract interface that all retrievers should implement,
enabling easy integration with the evaluation pipeline while preserving metadata.

Example:
    class MyRetriever(BaseRetriever):
        def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
            # Your retrieval logic here
            return [Evidence(text="...", metadata={...}), ...]
    
    # Use with RAGSetting
    setting = RAGSetting(MyRetriever(), top_k=5)
    prepared = setting.prepare(instance)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Evidence:
    """A single piece of retrieved evidence with metadata.
    
    Attributes:
        text: The evidence text content.
        metadata: Optional metadata (e.g., pmcid, section, score, rank).
    """
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}
    
    @classmethod
    def from_dict(cls, data: dict | str) -> "Evidence":
        if isinstance(data, str):
            return cls(text=data)
        return cls(text=data.get("text", ""), metadata=data.get("metadata", {}))


class BaseRetriever(ABC):
    """Abstract base class for retrievers.
    
    All custom retrievers should inherit from this class and implement
    the `retrieve` method. This ensures compatibility with RAGSetting
    and other evaluation components.
    
    The interface is designed to:
    1. Return Evidence objects with full metadata (not just strings)
    2. Support batch retrieval for efficiency
    3. Provide backward compatibility with simple callable interface
    """
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Retrieve relevant evidence for a query.
        
        Args:
            query: The search query.
            top_k: Maximum number of results to return.
        
        Returns:
            List of Evidence objects, ordered by relevance (best first).
        """
        pass
    
    def retrieve_batch(
        self, 
        queries: list[str], 
        top_k: int = 10
    ) -> list[list[Evidence]]:
        """Retrieve evidence for multiple queries.
        
        Default implementation calls retrieve() for each query.
        Subclasses can override for more efficient batch processing.
        
        Args:
            queries: List of search queries.
            top_k: Maximum results per query.
        
        Returns:
            List of evidence lists, one per query.
        """
        return [self.retrieve(q, top_k) for q in queries]
    
    def as_callable(self) -> Callable[[str], list[str]]:
        """Return a simple callable for backward compatibility.
        
        This allows using the retriever with code that expects
        a function signature of (query: str) -> list[str].
        
        Note: This loses metadata. Use retrieve() directly when possible.
        
        Returns:
            A function that takes a query and returns list of text strings.
        """
        def fn(query: str) -> list[str]:
            return [e.text for e in self.retrieve(query)]
        return fn
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class FunctionRetriever(BaseRetriever):
    """Wrapper to use a simple function as a retriever.
    
    This allows backward compatibility with existing code that uses
    plain functions for retrieval.
    
    Example:
        def my_search(query: str) -> list[str]:
            return ["result 1", "result 2"]
        
        retriever = FunctionRetriever(my_search)
        evidence = retriever.retrieve("test query")
    """
    
    def __init__(self, fn: Callable[[str], list[str]], top_k: int = 10):
        """Initialize with a retrieval function.
        
        Args:
            fn: Function that takes query string and returns list of text strings.
            top_k: Default top_k (the function may ignore this).
        """
        self._fn = fn
        self._default_top_k = top_k
    
    def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
        """Retrieve using the wrapped function."""
        results = self._fn(query)[:top_k]
        return [Evidence(text=text) for text in results]
    
    def __repr__(self) -> str:
        return f"FunctionRetriever(fn={self._fn.__name__ if hasattr(self._fn, '__name__') else '...'})"