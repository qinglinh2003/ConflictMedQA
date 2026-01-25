"""Confidence question type for calibration evaluation."""

from typing import Any

from .base import BaseQuestionType
from ..types import Instance
from ..extractors.base import BaseExtractor
from ..extractors.number import ConfidenceExtractor
from ..metrics.base import BaseMetric
from ..metrics.calibration import CalibrationScoreMetric, CalibrationGapMetric


class ConfidenceQuestion(BaseQuestionType):
    """Question type that asks for a confidence-rated response.
    
    This question type is designed to evaluate model calibration by
    requiring explicit confidence scores alongside answers.
    
    Example prompt:
        Claim: Aspirin reduces the risk of heart attack.
        
        Evidence:
        [1] A large clinical trial showed 25% reduction...
        [2] A meta-analysis found no significant benefit...
        
        Based on the evidence, does this claim appear to be true?
        
        Please provide:
        1. Your answer (Yes/No/Uncertain)
        2. Your confidence level (0-100%)
        3. Brief reasoning
        
        Format your response as:
        Answer: [Yes/No/Uncertain]
        Confidence: [0-100]%
        Reasoning: [Your explanation]
    """
    
    PROMPT_TEMPLATE = """Claim: {claim}

{evidence}

Based on the evidence, does this claim appear to be true?

Please provide:
1. Your answer (Yes/No/Uncertain)
2. Your confidence level (0-100%)
3. Brief reasoning for your confidence level

Format your response as:
Answer: [Yes/No/Uncertain]
Confidence: [0-100]%
Reasoning: [Your explanation]"""

    PROMPT_TEMPLATE_NO_EVIDENCE = """Claim: {claim}

Based on your knowledge, does this claim appear to be true?

Please provide:
1. Your answer (Yes/No/Uncertain)
2. Your confidence level (0-100%)
3. Brief reasoning for your confidence level

Format your response as:
Answer: [Yes/No/Uncertain]
Confidence: [0-100]%
Reasoning: [Your explanation]"""

    PROMPT_TEMPLATE_NUMERIC = """Claim: {claim}

{evidence}

On a scale from 0 to 1, how confident are you that this claim is true based on the evidence?

- 0.0 = Definitely false
- 0.5 = Uncertain / Mixed evidence
- 1.0 = Definitely true

Please respond with just a number between 0 and 1, followed by a brief explanation.

Confidence:"""
    
    def __init__(
        self,
        name: str = "confidence",
        extractor: BaseExtractor | None = None,
        format_style: str = "structured",
        conflict_expected_max: float = 0.6,
        consistent_expected_min: float = 0.7
    ):
        """Initialize the ConfidenceQuestion type.
        
        Args:
            name: Question type name.
            extractor: Optional custom extractor.
            format_style: "structured" for full format, "numeric" for simple.
            conflict_expected_max: Max expected confidence for conflicts.
            consistent_expected_min: Min expected confidence for consistent evidence.
        """
        super().__init__(name, extractor)
        self.format_style = format_style
        self.conflict_expected_max = conflict_expected_max
        self.consistent_expected_min = consistent_expected_min
    
    def _get_default_extractor(self) -> BaseExtractor:
        """Return ConfidenceExtractor as default."""
        return ConfidenceExtractor()
    
    def format(
        self, 
        instance: Instance, 
        include_evidence: bool = True
    ) -> str:
        """Format instance into a confidence question prompt.
        
        Args:
            instance: The evaluation instance.
            include_evidence: Whether to include evidence.
        
        Returns:
            Formatted prompt string.
        """
        if self.format_style == "numeric":
            if include_evidence and instance.evidence:
                evidence_str = self._format_evidence(instance.evidence)
                return self.PROMPT_TEMPLATE_NUMERIC.format(
                    claim=instance.claim,
                    evidence=evidence_str
                )
            else:
                return self.PROMPT_TEMPLATE_NUMERIC.format(
                    claim=instance.claim,
                    evidence=""
                ).replace("\n\n\n", "\n\n")
        else:
            if include_evidence and instance.evidence:
                evidence_str = self._format_evidence(instance.evidence)
                return self.PROMPT_TEMPLATE.format(
                    claim=instance.claim,
                    evidence=evidence_str
                )
            else:
                return self.PROMPT_TEMPLATE_NO_EVIDENCE.format(
                    claim=instance.claim
                )
    
    def get_metrics(self) -> list[BaseMetric]:
        """Return metrics for confidence evaluation.
        
        Returns:
            List containing CalibrationScore and CalibrationGap metrics.
        """
        return [
            CalibrationScoreMetric(
                conflict_expected_max=self.conflict_expected_max,
                consistent_expected_min=self.consistent_expected_min
            ),
            CalibrationGapMetric(),
        ]
