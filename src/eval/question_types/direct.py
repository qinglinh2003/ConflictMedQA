"""Direct question type for open-ended conflict assessment."""

from typing import Any

from .base import BaseQuestionType
from ..types import Instance
from ..extractors.base import BaseExtractor
from ..extractors.keyword import ConflictAwarenessExtractor
from ..metrics.base import BaseMetric
from ..metrics.conflict import (
    ConflictRecognitionMetric,
    MultiPerspectiveMetric,
    EvidenceUtilizationMetric,
)


class DirectQuestion(BaseQuestionType):
    """Direct open-ended question about a medical claim.
    
    This question type presents the claim and evidence without constraining
    the response format, allowing assessment of natural conflict recognition.
    
    Example prompt:
        Claim: Aspirin reduces the risk of heart attack.
        
        Evidence:
        [1] A large clinical trial showed 25% reduction in heart attacks...
        [2] A meta-analysis found no significant benefit...
        
        Based on the evidence provided, what can you conclude about this claim?
        Please discuss the strength of evidence and any limitations.
    """
    
    PROMPT_TEMPLATE = """Claim: {claim}

{evidence}

Based on the evidence provided, what can you conclude about this claim?
Please discuss the strength of evidence and any limitations or conflicting findings."""

    PROMPT_TEMPLATE_NO_EVIDENCE = """Claim: {claim}

Based on your knowledge, what can you conclude about this claim?
Please discuss the strength of evidence and any limitations or conflicting findings."""
    
    def __init__(
        self,
        name: str = "direct",
        extractor: BaseExtractor | None = None,
        include_hedging: bool = True
    ):
        """Initialize the DirectQuestion type.
        
        Args:
            name: Question type name.
            extractor: Optional custom extractor.
            include_hedging: Whether to include hedging in conflict detection.
        """
        super().__init__(name, extractor)
        self.include_hedging = include_hedging
    
    def _get_default_extractor(self) -> BaseExtractor:
        """Return ConflictAwarenessExtractor as default."""
        return ConflictAwarenessExtractor(include_hedging=self.include_hedging)
    
    def format(
        self, 
        instance: Instance, 
        include_evidence: bool = True
    ) -> str:
        """Format instance into a direct question prompt.
        
        Args:
            instance: The evaluation instance.
            include_evidence: Whether to include evidence.
        
        Returns:
            Formatted prompt string.
        """
        if include_evidence and instance.evidence:
            evidence_str = self._format_evidence(instance.evidence)
            return self.PROMPT_TEMPLATE.format(
                claim=instance.claim,
                evidence=evidence_str
            )
        else:
            return self.PROMPT_TEMPLATE_NO_EVIDENCE.format(claim=instance.claim)
    
    def get_metrics(self) -> list[BaseMetric]:
        """Return metrics for direct question evaluation.
        
        Returns:
            List containing ConflictRecognition, MultiPerspective, and
            EvidenceUtilization metrics.
        """
        return [
            ConflictRecognitionMetric(),
            MultiPerspectiveMetric(),
            EvidenceUtilizationMetric(),
        ]
