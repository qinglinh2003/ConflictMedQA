"""Consensus question type for scientific consensus detection."""

from typing import Any

from .base import BaseQuestionType
from ..types import Instance
from ..extractors.base import BaseExtractor
from ..extractors.choice import ChoiceExtractor
from ..metrics.base import BaseMetric
from ..metrics.accuracy import ConsensusAccuracy


class ConsensusQuestion(BaseQuestionType):
    """Question type for detecting scientific consensus level.
    
    This question type asks the model to assess the level of scientific
    consensus on a medical claim based on the evidence.
    
    Example prompt:
        Claim: Aspirin reduces the risk of heart attack.
        
        Evidence:
        [1] A large clinical trial showed 25% reduction...
        [2] A meta-analysis found no significant benefit...
        
        Based on the evidence, what is the level of scientific consensus 
        on this claim?
        
        Choose one:
        (A) Strong consensus supporting the claim
        (B) Strong consensus against the claim
        (C) Mixed evidence / No clear consensus
        (D) Insufficient evidence to determine
        
        Answer:
    """
    
    PROMPT_TEMPLATE = """Claim: {claim}

{evidence}

Based on the evidence, what is the level of scientific consensus on this claim?

Choose one:
(A) Strong consensus supporting the claim
(B) Strong consensus against the claim
(C) Mixed evidence / No clear consensus
(D) Insufficient evidence to determine

Answer:"""

    PROMPT_TEMPLATE_NO_EVIDENCE = """Claim: {claim}

Based on your knowledge of the medical literature, what is the level of scientific consensus on this claim?

Choose one:
(A) Strong consensus supporting the claim
(B) Strong consensus against the claim
(C) Mixed evidence / No clear consensus
(D) Insufficient evidence to determine

Answer:"""

    PROMPT_TEMPLATE_DETAILED = """Claim: {claim}

{evidence}

Analyze the evidence and determine the level of scientific consensus.

Consider:
- Do the studies agree or disagree?
- Are there methodological concerns?
- Is the evidence sufficient to draw conclusions?

Choose the most appropriate option:
(A) Strong consensus supporting the claim - Most/all evidence supports it
(B) Strong consensus against the claim - Most/all evidence refutes it
(C) Mixed evidence / No clear consensus - Evidence is conflicting
(D) Insufficient evidence to determine - Not enough quality evidence

First, briefly explain your reasoning, then state your answer.

Reasoning:"""
    
    def __init__(
        self,
        name: str = "consensus",
        extractor: BaseExtractor | None = None,
        detailed_format: bool = False
    ):
        """Initialize the ConsensusQuestion type.
        
        Args:
            name: Question type name.
            extractor: Optional custom extractor.
            detailed_format: Whether to use detailed prompt with reasoning.
        """
        super().__init__(name, extractor)
        self.detailed_format = detailed_format
    
    def _get_default_extractor(self) -> BaseExtractor:
        """Return ChoiceExtractor as default."""
        return ChoiceExtractor(choices=["A", "B", "C", "D"])
    
    def format(
        self, 
        instance: Instance, 
        include_evidence: bool = True
    ) -> str:
        """Format instance into a consensus question prompt.
        
        Args:
            instance: The evaluation instance.
            include_evidence: Whether to include evidence.
        
        Returns:
            Formatted prompt string.
        """
        if include_evidence and instance.evidence:
            evidence_str = self._format_evidence(instance.evidence)
            template = (
                self.PROMPT_TEMPLATE_DETAILED 
                if self.detailed_format 
                else self.PROMPT_TEMPLATE
            )
            return template.format(
                claim=instance.claim,
                evidence=evidence_str
            )
        else:
            return self.PROMPT_TEMPLATE_NO_EVIDENCE.format(
                claim=instance.claim
            )
    
    def get_metrics(self) -> list[BaseMetric]:
        """Return metrics for consensus evaluation.
        
        Returns:
            List containing ConsensusAccuracy metric.
        """
        return [
            ConsensusAccuracy(),
        ]
