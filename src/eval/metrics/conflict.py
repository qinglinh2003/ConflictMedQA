"""Conflict awareness and recognition metrics."""

from typing import Any

from .base import BaseMetric
from ..types import Instance, MetricResult, Label


class ConflictRecognitionMetric(BaseMetric):
    """Metric for measuring conflict recognition ability.
    
    Evaluates whether the model recognizes when evidence is conflicting
    vs. when it is consistent.
    """
    
    def __init__(
        self,
        name: str = "conflict_recognition",
        awareness_threshold: float = 0.3
    ):
        """Initialize the metric.
        
        Args:
            name: Metric name.
            awareness_threshold: Minimum awareness_score to count as recognized.
        """
        super().__init__(name)
        self.awareness_threshold = awareness_threshold
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute conflict recognition score.
        
        For conflict instances: score 1.0 if conflict is recognized, 0.0 otherwise.
        For non-conflict instances: score 1.0 if conflict is NOT falsely claimed.
        
        Args:
            extracted: Should contain "is_conflict_aware" or "awareness_score".
            instance: Instance with conflict label.
        
        Returns:
            MetricResult with recognition score.
        """
        # Get conflict awareness from extraction
        is_aware = extracted.get("is_conflict_aware", False)
        awareness_score = extracted.get("awareness_score", 0.0)
        
        # Use threshold if only score is available
        if "is_conflict_aware" not in extracted and awareness_score is not None:
            is_aware = awareness_score >= self.awareness_threshold
        
        is_conflict = instance.label == Label.CONFLICT
        
        details = {
            "is_conflict_instance": is_conflict,
            "model_recognized_conflict": is_aware,
            "awareness_score": awareness_score,
            "conflict_keywords": extracted.get("conflict_keywords", []),
        }
        
        # Compute score based on instance type
        if is_conflict:
            # True positive: conflict instance where model recognized conflict
            score = 1.0 if is_aware else 0.0
            details["classification"] = "true_positive" if is_aware else "false_negative"
        else:
            # True negative: non-conflict instance where model didn't claim conflict
            score = 1.0 if not is_aware else 0.0
            details["classification"] = "true_negative" if not is_aware else "false_positive"
        
        return MetricResult(name=self.name, value=score, details=details)


class MultiPerspectiveMetric(BaseMetric):
    """Metric for measuring multi-perspective acknowledgment.
    
    Evaluates whether the model presents multiple viewpoints when
    evidence is conflicting.
    """
    
    def __init__(
        self,
        name: str = "multi_perspective",
        min_perspectives: int = 2
    ):
        """Initialize the metric.
        
        Args:
            name: Metric name.
            min_perspectives: Minimum perspectives to count as multi-perspective.
        """
        super().__init__(name)
        self.min_perspectives = min_perspectives
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute multi-perspective score.
        
        Args:
            extracted: Should contain perspective indicators.
            instance: Instance for context.
        
        Returns:
            MetricResult with multi-perspective score.
        """
        perspective_keywords = extracted.get("perspective_keywords", [])
        conflict_keywords = extracted.get("conflict_keywords", [])
        
        # Count perspective indicators
        num_perspectives = len(perspective_keywords)
        has_contrast_words = any(
            kw in conflict_keywords 
            for kw in ["however", "on the other hand", "in contrast", "while"]
        )
        
        # Calculate score
        if num_perspectives >= self.min_perspectives:
            score = 1.0
        elif num_perspectives == 1 or has_contrast_words:
            score = 0.5
        else:
            score = 0.0
        
        details = {
            "perspective_keywords": perspective_keywords,
            "num_perspectives": num_perspectives,
            "has_contrast_words": has_contrast_words,
            "is_conflict_instance": instance.label == Label.CONFLICT,
        }
        
        return MetricResult(name=self.name, value=score, details=details)


class EvidenceUtilizationMetric(BaseMetric):
    """Metric for measuring evidence utilization.
    
    Evaluates whether the model appropriately uses/cites the provided
    evidence in its response.
    """
    
    def __init__(
        self,
        name: str = "evidence_utilization",
        require_citations: bool = False
    ):
        """Initialize the metric.
        
        Args:
            name: Metric name.
            require_citations: Whether explicit citations are required.
        """
        super().__init__(name)
        self.require_citations = require_citations
    
    def compute(
        self, 
        extracted: dict[str, Any], 
        instance: Instance
    ) -> MetricResult:
        """Compute evidence utilization score.
        
        Args:
            extracted: Should contain evidence usage indicators.
            instance: Instance with evidence.
        
        Returns:
            MetricResult with utilization score.
        """
        # Check for evidence references
        citations = extracted.get("citations", [])
        evidence_keywords = extracted.get("evidence_keywords", [])
        response_text = extracted.get("_raw_response", "")
        
        num_evidence = len(instance.evidence)
        num_cited = len(citations)
        
        details = {
            "num_evidence": num_evidence,
            "num_cited": num_cited,
            "citations": citations,
            "evidence_keywords": evidence_keywords,
        }
        
        if num_evidence == 0:
            # No evidence to utilize
            details["note"] = "No evidence provided"
            return MetricResult(name=self.name, value=1.0, details=details)
        
        # Calculate utilization score
        if self.require_citations:
            # Strict: require explicit citations
            score = num_cited / num_evidence if num_evidence > 0 else 0.0
        else:
            # Lenient: check for any evidence usage indicators
            has_evidence_reference = (
                num_cited > 0 or
                len(evidence_keywords) > 0 or
                any(f"[{i+1}]" in response_text for i in range(num_evidence)) or
                any(f"evidence {i+1}" in response_text.lower() for i in range(num_evidence)) or
                "according to" in response_text.lower() or
                "the study" in response_text.lower() or
                "research shows" in response_text.lower()
            )
            score = 1.0 if has_evidence_reference else 0.0
        
        return MetricResult(name=self.name, value=score, details=details)
