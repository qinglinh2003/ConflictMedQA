#!/usr/bin/env python
"""
Prompt templates for LLM labeling.
"""
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
import re


@dataclass
class LabelInput:
    """Input for labeling a single (query, document) pair."""
    query: str
    document: str  # Single document
    label_choices: list[str]  # e.g., ["positive", "negative", "no_significant_difference", "irrelevant"]


@dataclass
class LabelOutput:
    """Structured output from labeling."""
    label: str  # one of label_choices, or "UNKNOWN" if parse fails
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    raw_response: str = ""


class BasePrompt(ABC):
    """Abstract base class for prompts."""
    
    name: str = "base"
    system: str = ""
    
    @abstractmethod
    def format(self, input: LabelInput) -> str:
        """Format the prompt with inputs."""
        pass
    
    def parse(self, response: str, label_choices: list[str]) -> LabelOutput:
        """
        Parse LLM response into structured output.
        
        Default implementation: find first occurrence of any label_choice.
        Subclasses can override for custom parsing.
        """
        response_lower = response.lower()
        
        # Try to find a label from choices
        for label in label_choices:
            # Match as whole word
            pattern = r'\b' + re.escape(label.lower()) + r'\b'
            if re.search(pattern, response_lower):
                return LabelOutput(
                    label=label,
                    raw_response=response,
                )
        
        return LabelOutput(
            label="UNKNOWN",
            raw_response=response,
        )


# ==================== Registry ====================

PROMPT_REGISTRY: dict[str, type[BasePrompt]] = {}


def register_prompt(name: str):
    """Decorator to register a prompt class."""
    def decorator(cls: type[BasePrompt]):
        PROMPT_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_prompt(name: str) -> BasePrompt:
    """Get prompt instance by name."""
    if name not in PROMPT_REGISTRY:
        available = list(PROMPT_REGISTRY.keys())
        raise ValueError(f"Unknown prompt: {name}. Available: {available}")
    return PROMPT_REGISTRY[name]()


def list_prompts() -> list[str]:
    """List available prompt names."""
    return list(PROMPT_REGISTRY.keys())


# ==================== Prompt Implementations ====================

@register_prompt("simple")
class SimplePrompt(BasePrompt):
    """Simple prompt: read document and choose a label."""
    
    system = "You are a helpful assistant."
    
    def format(self, input: LabelInput) -> str:
        choices = ", ".join(input.label_choices)
        
        return f"""Read the following query and document, then choose a label.

<query>
{input.query}
</query>

<document>
{input.document}
</document>

Based on the document, choose one label from: {choices}

Answer with just the label."""


@register_prompt("medical_evidence")
class MedicalEvidencePrompt(BasePrompt):
    """Prompt for classifying medical evidence."""
    
    system = "You are a medical expert analyzing clinical trial evidence."
    
    def format(self, input: LabelInput) -> str:
        choices = ", ".join(input.label_choices)
        
        return f"""Your task is to classify the effect of an intervention on a specific outcome based on a medical document.

<query>
{input.query}
</query>

<document>
{input.document}
</document>

Instructions:
1. Focus ONLY on the specific outcome mentioned in the query
2. Ignore findings about other outcomes (e.g., if query asks about "pain score", ignore findings about "blood pressure")
3. Compare the intervention group vs the comparator group

Label definitions:
- positive: The outcome value is HIGHER/INCREASED in the intervention group compared to the comparator
- negative: The outcome value is LOWER/DECREASED in the intervention group compared to the comparator
- no_significant_difference: No statistically significant difference between groups for this outcome
- irrelevant: The document does not contain information about the specific outcome in the query

Choose one label from: {choices}

Respond with only the label, nothing else."""