"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from typing import Any

from ..types import Instance, MetricResult


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.
    
    Metrics compute specific measurements from extracted LLM responses
    compared against ground truth data from instances.
    
    Attributes:
        name: Unique identifier for this metric.
    """
    
    def __init__(self, name: str | None = None):
        """Initialize the metric.
        
        Args:
            name: Optional custom name. Defaults to class name.
        """
        self._name = name
    
    @property
    def name(self) -> str:
        """Return the metric name."""
        return self._name or self.__class__.__name__
    
    @abstractmethod
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute the metric value.
        
        Args:
            extracted: Dictionary of extracted values from LLM response.
            instance: The original instance with ground truth.
        
        Returns:
            MetricResult containing the computed value and details.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
