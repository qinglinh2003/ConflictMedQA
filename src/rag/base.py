"""
RAG Method base interface.

A RAGMethod defines HOW to use a retriever to get evidence for a query.
Different methods implement different retrieval strategies/workflows.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever, Evidence


class RAGMethod(ABC):
    """Abstract base class for RAG methods.
    
    A RAGMethod wraps a retriever and implements a specific retrieval
    strategy. The key method is `retrieve(query) -> list[Evidence]`.
    
    Subclasses implement different strategies:
    - BaselineRAG: Direct single-query retrieval
    - HyDERAG: Hypothetical document expansion
    - MultiQueryRAG: Query expansion with merging
    - IterativeRAG: Multi-round retrieval
    
    Example:
        class MyRAGMethod(RAGMethod):
            def retrieve(self, query: str) -> list[Evidence]:
                # Custom retrieval logic
                return self.retriever.retrieve(query, self.top_k)
    """
    
    def __init__(self, retriever: "BaseRetriever", top_k: int = 10):
        """Initialize RAG method.
        
        Args:
            retriever: Underlying retriever to use.
            top_k: Number of final evidence passages to return.
        """
        self.retriever = retriever
        self.top_k = top_k
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Method name for logging/config."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str) -> list["Evidence"]:
        """Retrieve evidence using this method's strategy.
        
        Args:
            query: The input query/question.
        
        Returns:
            List of Evidence objects (up to top_k).
        """
        pass
    
    def retrieve_batch(self, queries: list[str]) -> list[list["Evidence"]]:
        """Batch retrieval. Override for efficiency if possible."""
        return [self.retrieve(q) for q in queries]
    
    def as_retriever(self) -> "BaseRetriever":
        """Return a BaseRetriever-compatible interface.
        
        This allows RAGMethod to be used anywhere a BaseRetriever is expected.
        """
        from ..retrieve.interface import BaseRetriever, Evidence
        
        method = self
        
        class _RAGMethodAsRetriever(BaseRetriever):
            def retrieve(self, query: str, top_k: int = 10) -> list[Evidence]:
                return method.retrieve(query)
            
            def retrieve_batch(self, queries: list[str], top_k: int = 10) -> list[list[Evidence]]:
                return method.retrieve_batch(queries)
            
            def __repr__(self) -> str:
                return f"AsRetriever({method})"
        
        return _RAGMethodAsRetriever()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(retriever={self.retriever}, top_k={self.top_k})"