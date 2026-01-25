"""Base class for response extractors."""

from abc import ABC, abstractmethod
from typing import Any


class BaseExtractor(ABC):
    """Abstract base class for extracting structured data from LLM responses.
    
    Extractors are responsible for parsing raw LLM responses and extracting
    relevant information into a structured dictionary format.
    """
    
    @abstractmethod
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract structured data from a response.
        
        Args:
            response: Raw text response from the LLM.
            context: Optional context information (e.g., original prompt, instance).
        
        Returns:
            Dictionary containing extracted information. Keys depend on the
            specific extractor implementation.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
