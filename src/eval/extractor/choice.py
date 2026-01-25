"""Choice extractor for multiple-choice question responses."""

import re
from typing import Any

from .base import BaseExtractor


class ChoiceExtractor(BaseExtractor):
    """Extract choice selections from multiple-choice responses.
    
    This extractor uses regex patterns to identify selected choices
    from responses to multiple-choice questions.
    
    Attributes:
        choices: List of valid choice labels (e.g., ["A", "B", "C", "D"]).
        case_sensitive: Whether to match choices case-sensitively.
    """
    
    def __init__(
        self,
        choices: list[str] | None = None,
        case_sensitive: bool = False
    ):
        """Initialize the ChoiceExtractor.
        
        Args:
            choices: Valid choice labels. Defaults to ["A", "B", "C", "D"].
            case_sensitive: Whether matching is case-sensitive.
        """
        self.choices = choices or ["A", "B", "C", "D"]
        self.case_sensitive = case_sensitive
    
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract the selected choice from a response.
        
        Tries multiple patterns to find the choice:
        1. "Answer: X" or "Choice: X" patterns
        2. "The answer is X" patterns
        3. "(X)" at the start of a line
        4. Single letter on its own line
        5. First mention of a valid choice
        
        Args:
            response: Raw LLM response text.
            context: Optional context (unused).
        
        Returns:
            Dictionary with:
                - "choice": The extracted choice letter (or None if not found)
                - "confidence": Confidence score (1.0 for explicit patterns, lower for implicit)
                - "raw_match": The raw text that matched
        """
        text = response if self.case_sensitive else response.upper()
        choices_pattern = "|".join(
            c if self.case_sensitive else c.upper() 
            for c in self.choices
        )
        
        result = {
            "choice": None,
            "confidence": 0.0,
            "raw_match": None
        }
        
        # Pattern 1: Explicit answer declaration
        patterns = [
            rf"(?:answer|choice|select|option)\s*(?:is|:)\s*\(?({choices_pattern})\)?",
            rf"(?:i\s+(?:choose|select|pick))\s*\(?({choices_pattern})\)?",
            rf"(?:the\s+(?:correct\s+)?answer\s+is)\s*\(?({choices_pattern})\)?",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                choice = match.group(1).upper() if not self.case_sensitive else match.group(1)
                result["choice"] = choice
                result["confidence"] = 1.0
                result["raw_match"] = match.group(0)
                return result
        
        # Pattern 2: Choice at start of response or on its own line
        line_patterns = [
            rf"^\s*\(?({choices_pattern})\)?[\.\):]",  # "(A)." or "A:" at start
            rf"^\s*({choices_pattern})\s*$",  # Just the letter on a line
        ]
        
        for line in text.split("\n"):
            for pattern in line_patterns:
                match = re.match(pattern, line.strip())
                if match:
                    choice = match.group(1).upper() if not self.case_sensitive else match.group(1)
                    result["choice"] = choice
                    result["confidence"] = 0.9
                    result["raw_match"] = match.group(0)
                    return result
        
        # Pattern 3: First mention of a valid choice in parentheses
        paren_match = re.search(rf"\(({choices_pattern})\)", text)
        if paren_match:
            choice = paren_match.group(1).upper() if not self.case_sensitive else paren_match.group(1)
            result["choice"] = choice
            result["confidence"] = 0.7
            result["raw_match"] = paren_match.group(0)
            return result
        
        # Pattern 4: Last resort - find any mention of a choice
        for choice in (self.choices if self.case_sensitive else [c.upper() for c in self.choices]):
            if re.search(rf"\b{choice}\b", text):
                result["choice"] = choice
                result["confidence"] = 0.3
                result["raw_match"] = choice
                return result
        
        return result
    
    def __repr__(self) -> str:
        return f"ChoiceExtractor(choices={self.choices}, case_sensitive={self.case_sensitive})"
