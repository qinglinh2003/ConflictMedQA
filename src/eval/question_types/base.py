"""Base class for question types."""

from abc import ABC, abstractmethod
from typing import Any

from ..types import Instance
from ..extractors.base import BaseExtractor
from ..metrics.base import BaseMetric


class BaseQuestionType(ABC):
    """Abstract base class for question types.
    
    A QuestionType defines:
    1. How to format an instance into a prompt
    2. What extractor to use for parsing responses
    3. What metrics to compute for evaluation
    
    Attributes:
        name: Unique identifier for this question type.
        extractor: The extractor to use for parsing responses.
    """
    
    def __init__(
        self,
        name: str | None = None,
        extractor: BaseExtractor | None = None
    ):
        """Initialize the question type.
        
        Args:
            name: Optional custom name. Defaults to class name.
            extractor: Optional custom extractor. Uses default if None.
        """
        self._name = name
        self._extractor = extractor
    
    @property
    def name(self) -> str:
        """Return the question type name."""
        return self._name or self.__class__.__name__
    
    @property
    def extractor(self) -> BaseExtractor:
        """Return the extractor for this question type."""
        if self._extractor is None:
            self._extractor = self._get_default_extractor()
        return self._extractor
    
    @extractor.setter
    def extractor(self, value: BaseExtractor):
        """Set a custom extractor."""
        self._extractor = value
    
    @abstractmethod
    def _get_default_extractor(self) -> BaseExtractor:
        """Return the default extractor for this question type."""
        pass
    
    @abstractmethod
    def format(
        self, 
        instance: Instance, 
        include_evidence: bool = True
    ) -> str:
        """Format an instance into a prompt.
        
        Args:
            instance: The evaluation instance.
            include_evidence: Whether to include evidence in the prompt.
        
        Returns:
            Formatted prompt string.
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> list[BaseMetric]:
        """Return the metrics to compute for this question type.
        
        Returns:
            List of metric instances.
        """
        pass
    
    def extract(
        self, 
        response: str, 
        context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Extract structured data from a response.
        
        Args:
            response: Raw LLM response text.
            context: Optional context information.
        
        Returns:
            Dictionary of extracted values.
        """
        result = self.extractor.extract(response, context)
        # Store raw response for metrics that need it
        result["_raw_response"] = response
        return result
    
    def _format_evidence(self, evidence: list[str]) -> str:
        """Format evidence list into a string.
        
        Args:
            evidence: List of evidence passages.
        
        Returns:
            Formatted evidence string with numbered citations.
        """
        if not evidence:
            return ""
        
        lines = ["Evidence:"]
        for i, ev in enumerate(evidence, 1):
            lines.append(f"[{i}] {ev}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
