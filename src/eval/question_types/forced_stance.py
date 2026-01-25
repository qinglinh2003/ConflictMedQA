"""Forced stance question type for multiple-choice evaluation."""

from typing import Any

from .base import BaseQuestionType
from ..types import Instance
from ..extractors.base import BaseExtractor
from ..extractors.choice import ChoiceExtractor
from ..metrics.base import BaseMetric
from ..metrics.accuracy import ForcedStanceAccuracy


class ForcedStanceQuestion(BaseQuestionType):
    """Forced choice question requiring a definitive stance.
    
    This question type forces the model to take a position, which is useful
    for measuring how models handle uncertainty and conflicting evidence
    when they cannot express nuance.
    
    Example prompt:
        Claim: Aspirin reduces the risk of heart attack.
        
        Evidence:
        [1] A large clinical trial showed 25% reduction...
        [2] A meta-analysis found no significant benefit...
        
        Based on the evidence, does aspirin reduce the risk of heart attack?
        
        Choose one:
        (A) Yes, the evidence supports this claim
        (B) No, the evidence does not support this claim
        
        Answer:
    """
    
    PROMPT_TEMPLATE = """Claim: {claim}

{evidence}

Based on the evidence, {question}

Choose one:
(A) Yes, the evidence supports this claim
(B) No, the evidence does not support this claim

Answer:"""

    PROMPT_TEMPLATE_NO_EVIDENCE = """Claim: {claim}

Based on your knowledge, {question}

Choose one:
(A) Yes, this claim is supported
(B) No, this claim is not supported

Answer:"""

    PROMPT_TEMPLATE_WITH_UNCERTAIN = """Claim: {claim}

{evidence}

Based on the evidence, {question}

Choose one:
(A) Yes, the evidence supports this claim
(B) No, the evidence does not support this claim
(C) The evidence is mixed or inconclusive

Answer:"""
    
    def __init__(
        self,
        name: str = "forced_stance",
        extractor: BaseExtractor | None = None,
        allow_uncertain: bool = False,
        conflict_handling: str = "exclude"
    ):
        """Initialize the ForcedStanceQuestion type.
        
        Args:
            name: Question type name.
            extractor: Optional custom extractor.
            allow_uncertain: Whether to include an "uncertain" option.
            conflict_handling: How to handle conflicts in accuracy metric.
        """
        super().__init__(name, extractor)
        self.allow_uncertain = allow_uncertain
        self.conflict_handling = conflict_handling
    
    def _get_default_extractor(self) -> BaseExtractor:
        """Return ChoiceExtractor as default."""
        choices = ["A", "B", "C"] if self.allow_uncertain else ["A", "B"]
        return ChoiceExtractor(choices=choices)
    
    def format(
        self, 
        instance: Instance, 
        include_evidence: bool = True
    ) -> str:
        """Format instance into a forced stance prompt.
        
        Args:
            instance: The evaluation instance.
            include_evidence: Whether to include evidence.
        
        Returns:
            Formatted prompt string.
        """
        # Generate question from claim
        question = self._generate_question(instance.claim)
        
        if include_evidence and instance.evidence:
            evidence_str = self._format_evidence(instance.evidence)
            template = (
                self.PROMPT_TEMPLATE_WITH_UNCERTAIN 
                if self.allow_uncertain 
                else self.PROMPT_TEMPLATE
            )
            return template.format(
                claim=instance.claim,
                evidence=evidence_str,
                question=question
            )
        else:
            return self.PROMPT_TEMPLATE_NO_EVIDENCE.format(
                claim=instance.claim,
                question=question
            )
    
    def _generate_question(self, claim: str) -> str:
        """Generate a yes/no question from the claim.
        
        Args:
            claim: The medical claim.
        
        Returns:
            Question string.
        """
        # Simple transformation: add "is this claim true?"
        # More sophisticated methods could use templates or LLM
        claim_lower = claim.lower().strip()
        
        if claim_lower.startswith(("does ", "do ", "is ", "are ", "can ", "will ")):
            return claim + "?"
        else:
            return f"is it true that {claim.rstrip('.')}?"
    
    def get_metrics(self) -> list[BaseMetric]:
        """Return metrics for forced stance evaluation.
        
        Returns:
            List containing ForcedStanceAccuracy metric.
        """
        return [
            ForcedStanceAccuracy(conflict_handling=self.conflict_handling),
        ]
