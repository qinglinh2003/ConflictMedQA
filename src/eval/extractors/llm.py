"""LLM-based extractors for complex extraction tasks.

These extractors use an LLM to extract structured information from responses,
which can be more accurate than rule-based methods for complex cases.
"""

import json
import re
import logging
from typing import Any, TYPE_CHECKING

from .base import BaseExtractor

if TYPE_CHECKING:
    from ..llm.base import BaseLLM, GenerationConfig

logger = logging.getLogger(__name__)


class LLMExtractorBase(BaseExtractor):
    """Base class for LLM-based extractors.
    
    Provides common functionality for extractors that use an LLM
    to parse and extract information from responses.
    
    Attributes:
        llm: LLM backend for extraction.
        generation_config: Generation config for extraction calls.
        max_retries: Maximum retry attempts on parse failure.
    """
    
    def __init__(
        self,
        llm: "BaseLLM",
        generation_config: "GenerationConfig | None" = None,
        max_retries: int = 2,
    ):
        """Initialize the LLM extractor.
        
        Args:
            llm: LLM backend to use for extraction.
            generation_config: Override generation config for extraction.
            max_retries: Maximum retries on JSON parse failure.
        """
        self.llm = llm
        self.generation_config = generation_config
        self.max_retries = max_retries
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the extraction prompt."""
        return self.llm.generate(prompt, self.generation_config)
    
    def _parse_json(self, text: str) -> dict[str, Any] | None:
        """Try to parse JSON from LLM response.
        
        Handles common issues like markdown code blocks.
        """
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code block
        patterns = [
            r"```json\s*(.*?)\s*```",
            r"```\s*(.*?)\s*```",
            r"\{.*\}",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if "```" in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue
        
        return None


class LLMChoiceExtractor(LLMExtractorBase):
    """LLM-based extractor for multiple choice answers.
    
    Uses an LLM to identify the selected choice from a response,
    which can handle more complex response formats than regex.
    """
    
    EXTRACTION_PROMPT = """Analyze the following response and extract the selected choice.

Response to analyze:
\"\"\"
{response}
\"\"\"

The valid choices are: {choices}

Return a JSON object with:
- "choice": The selected choice letter (one of {choices}), or null if no clear choice
- "confidence": Your confidence in this extraction (0.0 to 1.0)
- "reasoning": Brief explanation of why you selected this choice

Return ONLY the JSON object, no other text."""

    def __init__(
        self,
        llm: "BaseLLM",
        choices: list[str] | None = None,
        generation_config: "GenerationConfig | None" = None,
    ):
        """Initialize the LLM choice extractor.
        
        Args:
            llm: LLM backend for extraction.
            choices: Valid choice labels. Defaults to ["A", "B", "C", "D"].
            generation_config: Override generation config.
        """
        super().__init__(llm, generation_config)
        self.choices = choices or ["A", "B", "C", "D"]
    
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract choice using LLM.
        
        Args:
            response: Raw LLM response to analyze.
            context: Optional context (unused).
        
        Returns:
            Dictionary with choice, confidence, and reasoning.
        """
        prompt = self.EXTRACTION_PROMPT.format(
            response=response,
            choices=", ".join(self.choices),
        )
        
        for attempt in range(self.max_retries + 1):
            try:
                llm_response = self._call_llm(prompt)
                parsed = self._parse_json(llm_response)
                
                if parsed and "choice" in parsed:
                    # Validate choice
                    choice = parsed.get("choice")
                    if choice and choice.upper() in [c.upper() for c in self.choices]:
                        return {
                            "choice": choice.upper(),
                            "confidence": float(parsed.get("confidence", 0.8)),
                            "reasoning": parsed.get("reasoning", ""),
                            "extraction_method": "llm",
                        }
                
            except Exception as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
        
        # Fallback: return empty result
        return {
            "choice": None,
            "confidence": 0.0,
            "reasoning": "LLM extraction failed",
            "extraction_method": "llm_failed",
        }


class LLMConflictExtractor(LLMExtractorBase):
    """LLM-based extractor for conflict awareness detection.
    
    Uses an LLM to analyze whether a response recognizes conflicting
    evidence, which can be more nuanced than keyword matching.
    """
    
    EXTRACTION_PROMPT = """Analyze the following response and determine if it recognizes conflicting or mixed evidence.

Response to analyze:
\"\"\"
{response}
\"\"\"

Evaluate whether the response:
1. Acknowledges that evidence is conflicting, mixed, or inconsistent
2. Presents multiple perspectives or viewpoints
3. Expresses appropriate uncertainty given conflicting information
4. Cites or references multiple pieces of evidence

Return a JSON object with:
- "is_conflict_aware": Boolean - does the response recognize conflict/uncertainty?
- "awareness_score": Float 0.0-1.0 - how strongly does it acknowledge conflict?
- "recognizes_multiple_perspectives": Boolean - does it present multiple viewpoints?
- "expresses_uncertainty": Boolean - does it express appropriate uncertainty?
- "evidence_cited": Integer - how many pieces of evidence are referenced?
- "key_phrases": List of phrases that indicate conflict awareness
- "reasoning": Brief explanation of your assessment

Return ONLY the JSON object, no other text."""

    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract conflict awareness using LLM.
        
        Args:
            response: Raw LLM response to analyze.
            context: Optional context (unused).
        
        Returns:
            Dictionary with conflict awareness indicators.
        """
        prompt = self.EXTRACTION_PROMPT.format(response=response)
        
        for attempt in range(self.max_retries + 1):
            try:
                llm_response = self._call_llm(prompt)
                parsed = self._parse_json(llm_response)
                
                if parsed and "is_conflict_aware" in parsed:
                    return {
                        "is_conflict_aware": bool(parsed.get("is_conflict_aware", False)),
                        "awareness_score": float(parsed.get("awareness_score", 0.0)),
                        "recognizes_multiple_perspectives": bool(parsed.get("recognizes_multiple_perspectives", False)),
                        "expresses_uncertainty": bool(parsed.get("expresses_uncertainty", False)),
                        "evidence_cited": int(parsed.get("evidence_cited", 0)),
                        "conflict_keywords": parsed.get("key_phrases", []),
                        "perspective_keywords": [],
                        "reasoning": parsed.get("reasoning", ""),
                        "extraction_method": "llm",
                    }
                
            except Exception as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
        
        # Fallback
        return {
            "is_conflict_aware": False,
            "awareness_score": 0.0,
            "conflict_keywords": [],
            "perspective_keywords": [],
            "extraction_method": "llm_failed",
        }


class LLMConfidenceExtractor(LLMExtractorBase):
    """LLM-based extractor for confidence scores.
    
    Uses an LLM to extract confidence levels from responses,
    handling both explicit numerical and implicit qualitative expressions.
    """
    
    EXTRACTION_PROMPT = """Analyze the following response and extract the confidence level expressed.

Response to analyze:
\"\"\"
{response}
\"\"\"

Look for:
1. Explicit confidence scores (e.g., "80% confident", "confidence: 0.7")
2. Qualitative expressions (e.g., "very confident", "uncertain", "likely")
3. Hedging language that implies lower confidence
4. Strong assertions that imply higher confidence

Return a JSON object with:
- "confidence": Float 0.0-1.0 representing the overall confidence level
- "confidence_type": "explicit" if a number was given, "implicit" if inferred from language
- "raw_expression": The exact phrase/number that indicates confidence (if any)
- "reasoning": Brief explanation of how you determined the confidence level

Return ONLY the JSON object, no other text."""

    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract confidence score using LLM.
        
        Args:
            response: Raw LLM response to analyze.
            context: Optional context (unused).
        
        Returns:
            Dictionary with confidence score and metadata.
        """
        prompt = self.EXTRACTION_PROMPT.format(response=response)
        
        for attempt in range(self.max_retries + 1):
            try:
                llm_response = self._call_llm(prompt)
                parsed = self._parse_json(llm_response)
                
                if parsed and "confidence" in parsed:
                    confidence = float(parsed.get("confidence", 0.5))
                    # Clamp to valid range
                    confidence = max(0.0, min(1.0, confidence))
                    
                    return {
                        "confidence": round(confidence, 3),
                        "source": parsed.get("confidence_type", "llm_inferred"),
                        "raw_expression": parsed.get("raw_expression"),
                        "reasoning": parsed.get("reasoning", ""),
                        "extraction_method": "llm",
                    }
                
            except Exception as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
        
        # Fallback
        return {
            "confidence": None,
            "source": "llm_failed",
            "raw_expression": None,
            "extraction_method": "llm_failed",
        }


class LLMStructuredExtractor(LLMExtractorBase):
    """Generic LLM-based extractor for custom structured output.
    
    Allows defining a custom extraction schema and prompt template.
    """
    
    DEFAULT_PROMPT = """Analyze the following response and extract the requested information.

Response to analyze:
\"\"\"
{response}
\"\"\"

{additional_instructions}

Extract the following fields:
{schema_description}

Return ONLY a JSON object with these fields, no other text."""

    def __init__(
        self,
        llm: "BaseLLM",
        schema: dict[str, str],
        additional_instructions: str = "",
        generation_config: "GenerationConfig | None" = None,
    ):
        """Initialize the structured extractor.
        
        Args:
            llm: LLM backend for extraction.
            schema: Dictionary mapping field names to descriptions.
            additional_instructions: Extra instructions for the LLM.
            generation_config: Override generation config.
        """
        super().__init__(llm, generation_config)
        self.schema = schema
        self.additional_instructions = additional_instructions
    
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract structured data using LLM.
        
        Args:
            response: Raw LLM response to analyze.
            context: Optional context (can include additional instructions).
        
        Returns:
            Dictionary with extracted fields.
        """
        # Build schema description
        schema_lines = []
        for field, description in self.schema.items():
            schema_lines.append(f"- {field}: {description}")
        schema_description = "\n".join(schema_lines)
        
        # Get additional instructions from context if provided
        instructions = self.additional_instructions
        if context and "instructions" in context:
            instructions = context["instructions"]
        
        prompt = self.DEFAULT_PROMPT.format(
            response=response,
            additional_instructions=instructions,
            schema_description=schema_description,
        )
        
        for attempt in range(self.max_retries + 1):
            try:
                llm_response = self._call_llm(prompt)
                parsed = self._parse_json(llm_response)
                
                if parsed:
                    # Ensure all schema fields are present
                    result = {"extraction_method": "llm"}
                    for field in self.schema:
                        result[field] = parsed.get(field)
                    return result
                
            except Exception as e:
                logger.warning(f"LLM extraction attempt {attempt + 1} failed: {e}")
        
        # Fallback: return None for all fields
        result = {"extraction_method": "llm_failed"}
        for field in self.schema:
            result[field] = None
        return result
