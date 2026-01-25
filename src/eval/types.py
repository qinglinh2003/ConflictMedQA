"""Core data types for the ConflictMedQA evaluation module."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Label(str, Enum):
    """Labels for evidence conflict status."""
    SUPPORT = "support"
    CONTRADICT = "contradict"
    CONFLICT = "conflict"  # Mixed evidence
    NEUTRAL = "neutral"


@dataclass
class Instance:
    """A single evaluation instance containing a claim and associated evidence.
    
    Attributes:
        id: Unique identifier for this instance.
        claim: The medical claim to be evaluated.
        evidence: List of evidence passages (may contain conflicting information).
        label: Ground truth label indicating the relationship between evidence.
        ground_truth: Additional ground truth information for specific metrics.
        metadata: Optional additional metadata.
    """
    id: str
    claim: str
    evidence: list[str]
    label: Label
    ground_truth: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        """Create an Instance from a dictionary."""
        return cls(
            id=data["id"],
            claim=data["claim"],
            evidence=data.get("evidence", []),
            label=Label(data["label"]) if isinstance(data["label"], str) else data["label"],
            ground_truth=data.get("ground_truth", {}),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert instance to dictionary."""
        return {
            "id": self.id,
            "claim": self.claim,
            "evidence": self.evidence,
            "label": self.label.value if isinstance(self.label, Label) else self.label,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }


@dataclass
class MetricResult:
    """Result from computing a single metric.
    
    Attributes:
        name: Name of the metric.
        value: Computed metric value (typically 0-1 or percentage).
        details: Additional details about the computation.
    """
    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "details": self.details,
        }


@dataclass
class EvalResult:
    """Complete evaluation result for a single instance.
    
    Attributes:
        instance_id: ID of the evaluated instance.
        question_type: Name of the question type used.
        prompt: The formatted prompt sent to the LLM.
        response: Raw response from the LLM.
        extracted: Extracted structured data from the response.
        metrics: List of computed metric results.
        error: Error message if evaluation failed.
    """
    instance_id: str
    question_type: str
    prompt: str
    response: str
    extracted: dict[str, Any]
    metrics: list[MetricResult]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "instance_id": self.instance_id,
            "question_type": self.question_type,
            "prompt": self.prompt,
            "response": self.response,
            "extracted": self.extracted,
            "metrics": [m.to_dict() for m in self.metrics],
            "error": self.error,
        }


@dataclass
class AggregatedResults:
    """Aggregated results across multiple instances.
    
    Attributes:
        question_type: Name of the question type.
        num_instances: Total number of instances evaluated.
        num_errors: Number of instances with errors.
        metrics: Aggregated metric values (mean, std, etc.).
        per_instance: Optional per-instance results.
    """
    question_type: str
    num_instances: int
    num_errors: int
    metrics: dict[str, dict[str, float]]
    per_instance: list[EvalResult] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        result = {
            "question_type": self.question_type,
            "num_instances": self.num_instances,
            "num_errors": self.num_errors,
            "metrics": self.metrics,
        }
        if self.per_instance:
            result["per_instance"] = [r.to_dict() for r in self.per_instance]
        return result