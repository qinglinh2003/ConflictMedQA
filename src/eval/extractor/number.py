"""Number extractors for extracting numerical values from responses."""

import re
from typing import Any

from .base import BaseExtractor


class NumberExtractor(BaseExtractor):
    """Extract numerical values from responses.
    
    This extractor finds numbers in various formats (integers, decimals,
    percentages) and can optionally normalize them to a specific range.
    
    Attributes:
        min_value: Minimum valid value.
        max_value: Maximum valid value.
        normalize_to: If set, normalize values to this range (e.g., (0, 1)).
    """
    
    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        normalize_to: tuple[float, float] | None = None
    ):
        """Initialize the NumberExtractor.
        
        Args:
            min_value: Minimum valid value (values below are clamped or rejected).
            max_value: Maximum valid value (values above are clamped or rejected).
            normalize_to: Target range for normalization, e.g., (0, 1).
        """
        self.min_value = min_value
        self.max_value = max_value
        self.normalize_to = normalize_to
    
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract numerical values from a response.
        
        Args:
            response: Raw LLM response text.
            context: Optional context (unused).
        
        Returns:
            Dictionary with:
                - "value": The primary extracted value (or None)
                - "raw_value": The original value before normalization
                - "is_percentage": Whether the value was expressed as a percentage
                - "all_values": List of all numerical values found
        """
        result = {
            "value": None,
            "raw_value": None,
            "is_percentage": False,
            "all_values": [],
        }
        
        # Find all numbers (including percentages)
        # Pattern matches: 0.5, .5, 50%, 50.5%, 50, etc.
        number_pattern = r"(\d+\.?\d*|\.\d+)\s*(%?)"
        matches = re.findall(number_pattern, response)
        
        all_values = []
        for num_str, pct in matches:
            try:
                value = float(num_str)
                is_pct = pct == "%"
                
                # Convert percentage to decimal if needed
                if is_pct and self.normalize_to == (0, 1):
                    value = value / 100
                
                # Apply bounds
                if self.min_value is not None and value < self.min_value:
                    continue
                if self.max_value is not None and value > self.max_value:
                    continue
                
                all_values.append({
                    "value": value,
                    "raw_value": float(num_str),
                    "is_percentage": is_pct
                })
            except ValueError:
                continue
        
        result["all_values"] = all_values
        
        # Select primary value using heuristics
        if all_values:
            # Prefer values that look like confidence scores (0-1 or 0-100%)
            for v in all_values:
                raw = v["raw_value"]
                if 0 <= raw <= 1 or (v["is_percentage"] and 0 <= raw <= 100):
                    result["value"] = v["value"]
                    result["raw_value"] = v["raw_value"]
                    result["is_percentage"] = v["is_percentage"]
                    break
            else:
                # Fall back to first valid value
                result["value"] = all_values[0]["value"]
                result["raw_value"] = all_values[0]["raw_value"]
                result["is_percentage"] = all_values[0]["is_percentage"]
        
        return result


class ConfidenceExtractor(BaseExtractor):
    """Specialized extractor for confidence scores.
    
    This extractor looks for confidence expressions and extracts
    a numerical confidence value on a 0-1 scale.
    """
    
    # Qualitative confidence mappings
    CONFIDENCE_QUALITATIVE = {
        # High confidence
        "very confident": 0.95,
        "highly confident": 0.95,
        "extremely confident": 0.98,
        "certain": 0.95,
        "definitely": 0.95,
        "absolutely": 0.98,
        "sure": 0.90,
        "very sure": 0.95,
        
        # Medium-high confidence
        "confident": 0.80,
        "fairly confident": 0.75,
        "reasonably confident": 0.75,
        "quite confident": 0.80,
        "probably": 0.75,
        "likely": 0.70,
        
        # Medium confidence
        "somewhat confident": 0.60,
        "moderately confident": 0.60,
        "moderate confidence": 0.60,
        
        # Low confidence
        "not very confident": 0.35,
        "slightly confident": 0.40,
        "uncertain": 0.40,
        "unsure": 0.35,
        "not sure": 0.35,
        
        # Very low confidence
        "not confident": 0.20,
        "very uncertain": 0.20,
        "highly uncertain": 0.15,
        "no idea": 0.10,
    }
    
    def __init__(self):
        """Initialize the ConfidenceExtractor."""
        self._number_extractor = NumberExtractor(
            min_value=0,
            max_value=100,
            normalize_to=(0, 1)
        )
    
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract confidence score from a response.
        
        Tries multiple approaches:
        1. Look for explicit confidence statements (e.g., "confidence: 0.8")
        2. Look for percentage confidence (e.g., "80% confident")
        3. Look for qualitative expressions (e.g., "I am fairly confident")
        4. Fall back to numerical extraction
        
        Args:
            response: Raw LLM response text.
            context: Optional context (unused).
        
        Returns:
            Dictionary with:
                - "confidence": Confidence score (0-1 scale)
                - "source": How the confidence was determined
                - "raw_expression": The matched expression
        """
        text = response.lower()
        
        result = {
            "confidence": None,
            "source": None,
            "raw_expression": None,
        }
        
        # Pattern 1: Explicit confidence declaration
        explicit_patterns = [
            r"confidence\s*(?:level|score|rating)?[:\s]+(\d+\.?\d*)\s*(%?)",
            r"(\d+\.?\d*)\s*(%?)\s*(?:confident|confidence)",
            r"(?:my\s+)?confidence\s+is\s+(\d+\.?\d*)\s*(%?)",
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, text)
            if match:
                value = float(match.group(1))
                is_pct = match.group(2) == "%" if len(match.groups()) > 1 else False
                
                if is_pct or value > 1:
                    value = value / 100
                
                value = max(0, min(1, value))
                
                result["confidence"] = round(value, 3)
                result["source"] = "explicit"
                result["raw_expression"] = match.group(0)
                return result
        
        # Pattern 2: Qualitative expressions
        for phrase, value in sorted(
            self.CONFIDENCE_QUALITATIVE.items(),
            key=lambda x: len(x[0]),
            reverse=True
        ):
            if phrase in text:
                result["confidence"] = value
                result["source"] = "qualitative"
                result["raw_expression"] = phrase
                return result
        
        # Pattern 3: Fall back to number extraction
        num_result = self._number_extractor.extract(response)
        if num_result["value"] is not None:
            value = num_result["value"]
            # Normalize if needed
            if num_result["is_percentage"]:
                value = num_result["raw_value"] / 100
            elif value > 1:
                value = value / 100
            
            value = max(0, min(1, value))
            
            result["confidence"] = round(value, 3)
            result["source"] = "numeric"
            result["raw_expression"] = str(num_result["raw_value"])
        
        return result
