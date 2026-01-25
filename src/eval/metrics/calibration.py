"""Calibration metrics for confidence evaluation."""

from typing import Any
import math

from .base import BaseMetric
from ..types import Instance, MetricResult, Label


class CalibrationScoreMetric(BaseMetric):
    """Metric for measuring confidence calibration.
    
    Evaluates whether the model's confidence appropriately reflects
    the actual uncertainty in the evidence.
    
    For conflicting evidence, lower confidence is expected (better calibrated).
    For consistent evidence, higher confidence is expected.
    """
    
    def __init__(
        self,
        name: str = "calibration_score",
        conflict_expected_max: float = 0.6,
        consistent_expected_min: float = 0.7
    ):
        """Initialize the metric.
        
        Args:
            name: Metric name.
            conflict_expected_max: Maximum expected confidence for conflicts.
            consistent_expected_min: Minimum expected confidence for consistent evidence.
        """
        super().__init__(name)
        self.conflict_expected_max = conflict_expected_max
        self.consistent_expected_min = consistent_expected_min
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute calibration score.
        
        Args:
            extracted: Should contain "confidence" key (0-1 scale).
            instance: Instance with label.
        
        Returns:
            MetricResult with calibration score (higher = better calibrated).
        """
        confidence = extracted.get("confidence")
        
        details = {
            "confidence": confidence,
            "label": instance.label.value if isinstance(instance.label, Label) else instance.label,
        }
        
        if confidence is None:
            details["error"] = "No confidence extracted"
            return MetricResult(name=self.name, value=float("nan"), details=details)
        
        is_conflict = instance.label == Label.CONFLICT
        
        if is_conflict:
            # For conflicts: lower confidence is better calibrated
            # Score 1.0 if confidence <= expected_max
            # Score decreases as confidence exceeds expected_max
            if confidence <= self.conflict_expected_max:
                score = 1.0
            else:
                # Linear decrease: score = 0 when confidence = 1.0
                excess = confidence - self.conflict_expected_max
                max_excess = 1.0 - self.conflict_expected_max
                score = max(0.0, 1.0 - (excess / max_excess))
            
            details["expected_range"] = f"<= {self.conflict_expected_max}"
            details["calibration_type"] = "conflict"
        else:
            # For consistent evidence: higher confidence is better calibrated
            if confidence >= self.consistent_expected_min:
                score = 1.0
            else:
                # Linear decrease: score approaches 0 as confidence approaches 0
                score = confidence / self.consistent_expected_min
            
            details["expected_range"] = f">= {self.consistent_expected_min}"
            details["calibration_type"] = "consistent"
        
        details["is_well_calibrated"] = score >= 0.8
        
        return MetricResult(name=self.name, value=round(score, 3), details=details)


class CalibrationGapMetric(BaseMetric):
    """Metric for measuring the calibration gap.
    
    Computes the difference between expected and actual confidence,
    useful for understanding systematic over/under-confidence.
    """
    
    def __init__(
        self,
        name: str = "calibration_gap",
        conflict_target: float = 0.4,
        consistent_target: float = 0.85
    ):
        """Initialize the metric.
        
        Args:
            name: Metric name.
            conflict_target: Target confidence for conflict instances.
            consistent_target: Target confidence for consistent instances.
        """
        super().__init__(name)
        self.conflict_target = conflict_target
        self.consistent_target = consistent_target
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute calibration gap.
        
        Args:
            extracted: Should contain "confidence" key (0-1 scale).
            instance: Instance with label.
        
        Returns:
            MetricResult with calibration gap (0 = perfectly calibrated,
            positive = overconfident, negative = underconfident).
        """
        confidence = extracted.get("confidence")
        
        details = {
            "confidence": confidence,
            "label": instance.label.value if isinstance(instance.label, Label) else instance.label,
        }
        
        if confidence is None:
            details["error"] = "No confidence extracted"
            return MetricResult(name=self.name, value=float("nan"), details=details)
        
        is_conflict = instance.label == Label.CONFLICT
        target = self.conflict_target if is_conflict else self.consistent_target
        
        gap = confidence - target
        
        details["target"] = target
        details["gap"] = round(gap, 3)
        details["is_overconfident"] = gap > 0.1
        details["is_underconfident"] = gap < -0.1
        details["is_calibrated"] = abs(gap) <= 0.1
        
        # Return absolute gap as the metric value (lower is better)
        return MetricResult(name=self.name, value=round(abs(gap), 3), details=details)


class ExpectedCalibrationError(BaseMetric):
    """Expected Calibration Error (ECE) metric.
    
    This is typically computed over a batch of instances, but we provide
    per-instance contributions that can be aggregated later.
    """
    
    def __init__(
        self,
        name: str = "ece_contribution",
        num_bins: int = 10
    ):
        """Initialize the metric.
        
        Args:
            name: Metric name.
            num_bins: Number of confidence bins for ECE calculation.
        """
        super().__init__(name)
        self.num_bins = num_bins
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute per-instance ECE contribution.
        
        For full ECE, aggregate these results across instances.
        
        Args:
            extracted: Should contain "confidence" and correctness indicator.
            instance: Instance with label.
        
        Returns:
            MetricResult with per-instance calibration data for aggregation.
        """
        confidence = extracted.get("confidence")
        is_correct = extracted.get("is_correct", None)
        
        details = {
            "confidence": confidence,
            "is_correct": is_correct,
        }
        
        if confidence is None or is_correct is None:
            details["error"] = "Missing confidence or correctness"
            return MetricResult(name=self.name, value=float("nan"), details=details)
        
        # Determine bin
        bin_idx = min(int(confidence * self.num_bins), self.num_bins - 1)
        bin_lower = bin_idx / self.num_bins
        bin_upper = (bin_idx + 1) / self.num_bins
        
        details["bin_idx"] = bin_idx
        details["bin_range"] = (bin_lower, bin_upper)
        
        # Per-instance contribution to ECE
        # |confidence - accuracy| for this instance
        accuracy_proxy = 1.0 if is_correct else 0.0
        contribution = abs(confidence - accuracy_proxy)
        
        return MetricResult(name=self.name, value=round(contribution, 3), details=details)
