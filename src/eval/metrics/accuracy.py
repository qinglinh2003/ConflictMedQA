"""Accuracy-based metrics for evaluation."""

from typing import Any

from .base import BaseMetric
from ..types import Instance, MetricResult, Label


class ForcedStanceAccuracy(BaseMetric):
    """Accuracy metric for forced stance questions.
    
    Measures whether the model's forced choice matches the expected
    stance based on the evidence (support/contradict).
    
    For conflict instances, there may not be a single correct answer,
    so this metric can be configured to handle conflicts differently.
    """
    
    def __init__(
        self,
        name: str = "forced_stance_accuracy",
        conflict_handling: str = "exclude"
    ):
        """Initialize the metric.
        
        Args:
            name: Metric name.
            conflict_handling: How to handle conflict instances:
                - "exclude": Skip conflict instances (return None)
                - "any_valid": Accept either stance for conflicts
                - "strict": Require a specific answer even for conflicts
        """
        super().__init__(name)
        self.conflict_handling = conflict_handling
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute forced stance accuracy.
        
        Args:
            extracted: Must contain "choice" key with the selected option.
            instance: Must have label and optionally ground_truth["expected_choice"].
        
        Returns:
            MetricResult with value 1.0 (correct), 0.0 (incorrect), or None (skipped).
        """
        choice = extracted.get("choice")
        expected = instance.ground_truth.get("expected_choice")
        
        details = {
            "predicted": choice,
            "expected": expected,
            "label": instance.label.value if isinstance(instance.label, Label) else instance.label,
            "skipped": False,
        }
        
        # Handle missing data
        if choice is None:
            details["error"] = "No choice extracted"
            return MetricResult(name=self.name, value=0.0, details=details)
        
        # Handle conflict instances
        if instance.label == Label.CONFLICT:
            if self.conflict_handling == "exclude":
                details["skipped"] = True
                details["reason"] = "Conflict instance excluded"
                return MetricResult(name=self.name, value=float("nan"), details=details)
            elif self.conflict_handling == "any_valid":
                # For conflicts, any valid choice is acceptable
                valid_choices = instance.ground_truth.get("valid_choices", [])
                is_correct = choice in valid_choices if valid_choices else True
                details["valid_choices"] = valid_choices
                return MetricResult(
                    name=self.name, 
                    value=1.0 if is_correct else 0.0, 
                    details=details
                )
        
        # Standard accuracy computation
        if expected is None:
            # Infer expected choice from label
            label_to_choice = {
                Label.SUPPORT: "A",  # Typically A = support
                Label.CONTRADICT: "B",  # Typically B = contradict
            }
            expected = label_to_choice.get(instance.label)
            details["expected"] = expected
        
        if expected is None:
            details["error"] = "Could not determine expected choice"
            return MetricResult(name=self.name, value=float("nan"), details=details)
        
        is_correct = choice.upper() == expected.upper()
        return MetricResult(
            name=self.name,
            value=1.0 if is_correct else 0.0,
            details=details
        )


class ConsensusAccuracy(BaseMetric):
    """Accuracy metric for consensus detection questions.
    
    Measures whether the model correctly identifies the level of
    scientific consensus on a claim.
    """
    
    def __init__(self, name: str = "consensus_accuracy"):
        """Initialize the metric.
        
        Args:
            name: Metric name.
        """
        super().__init__(name)
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute consensus accuracy.
        
        Args:
            extracted: Must contain "choice" or "consensus_level" key.
            instance: Must have ground_truth["consensus_level"] or similar.
        
        Returns:
            MetricResult with value 1.0 (correct) or 0.0 (incorrect).
        """
        predicted = extracted.get("choice") or extracted.get("consensus_level")
        expected = instance.ground_truth.get("expected_choice") or instance.ground_truth.get("consensus_level")
        
        details = {
            "predicted": predicted,
            "expected": expected,
            "label": instance.label.value if isinstance(instance.label, Label) else instance.label,
        }
        
        if predicted is None:
            details["error"] = "No prediction extracted"
            return MetricResult(name=self.name, value=0.0, details=details)
        
        if expected is None:
            # For conflict instances, the expected answer is typically "no consensus" or "C"
            if instance.label == Label.CONFLICT:
                expected = instance.ground_truth.get("conflict_choice", "C")
                details["expected"] = expected
                details["note"] = "Inferred expected from conflict label"
            else:
                details["error"] = "No expected value available"
                return MetricResult(name=self.name, value=float("nan"), details=details)
        
        # Normalize for comparison
        predicted_norm = str(predicted).upper().strip()
        expected_norm = str(expected).upper().strip()
        
        is_correct = predicted_norm == expected_norm
        return MetricResult(
            name=self.name,
            value=1.0 if is_correct else 0.0,
            details=details
        )
