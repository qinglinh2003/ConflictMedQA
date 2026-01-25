"""Keyword-based extractors for detecting specific patterns in responses."""

import re
from typing import Any

from .base import BaseExtractor


class KeywordExtractor(BaseExtractor):
    """Extract information based on keyword/phrase detection.
    
    This extractor checks for the presence of specified keywords or phrases
    in the response text.
    
    Attributes:
        keywords: Dictionary mapping category names to lists of keywords.
        case_sensitive: Whether matching is case-sensitive.
    """
    
    def __init__(
        self,
        keywords: dict[str, list[str]],
        case_sensitive: bool = False
    ):
        """Initialize the KeywordExtractor.
        
        Args:
            keywords: Dictionary mapping category names to keyword lists.
                Example: {"positive": ["yes", "agree"], "negative": ["no", "disagree"]}
            case_sensitive: Whether matching is case-sensitive.
        """
        self.keywords = keywords
        self.case_sensitive = case_sensitive
    
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract keyword matches from a response.
        
        Args:
            response: Raw LLM response text.
            context: Optional context (unused).
        
        Returns:
            Dictionary with:
                - For each category: list of matched keywords
                - "matched_categories": list of categories with at least one match
                - "primary_category": category with most matches (or None)
        """
        text = response if self.case_sensitive else response.lower()
        
        result = {
            "matched_categories": [],
            "primary_category": None,
        }
        
        max_matches = 0
        
        for category, kw_list in self.keywords.items():
            matches = []
            for keyword in kw_list:
                kw = keyword if self.case_sensitive else keyword.lower()
                # Use word boundaries for whole word matching
                pattern = rf"\b{re.escape(kw)}\b"
                if re.search(pattern, text):
                    matches.append(keyword)
            
            result[category] = matches
            
            if matches:
                result["matched_categories"].append(category)
                if len(matches) > max_matches:
                    max_matches = len(matches)
                    result["primary_category"] = category
        
        return result


class ConflictAwarenessExtractor(BaseExtractor):
    """Specialized extractor for detecting conflict awareness in responses.
    
    This extractor looks for phrases indicating the model recognizes
    conflicting evidence or uncertainty.
    """
    
    # Keywords indicating conflict awareness
    CONFLICT_INDICATORS = [
        "conflicting",
        "conflict",
        "contradictory",
        "contradiction",
        "inconsistent",
        "inconsistency",
        "mixed evidence",
        "mixed results",
        "debate",
        "controversy",
        "controversial",
        "disputed",
        "disagreement",
        "on the other hand",
        "however",
        "in contrast",
        "alternatively",
        "some studies suggest",
        "while other studies",
        "evidence is mixed",
        "results vary",
        "not conclusive",
        "inconclusive",
        "uncertain",
        "no clear consensus",
        "both support and refute",
    ]
    
    # Keywords indicating acknowledgment of multiple perspectives
    PERSPECTIVE_INDICATORS = [
        "one perspective",
        "another perspective",
        "some argue",
        "others argue",
        "proponents",
        "critics",
        "supporters",
        "opponents",
        "one view",
        "another view",
        "different interpretations",
        "various viewpoints",
    ]
    
    # Keywords indicating hedging/uncertainty
    HEDGING_INDICATORS = [
        "may",
        "might",
        "could",
        "possibly",
        "potentially",
        "perhaps",
        "it seems",
        "appears to",
        "suggests",
        "indicates",
        "likely",
        "unlikely",
        "uncertain",
        "unclear",
        "depends on",
    ]
    
    def __init__(self, include_hedging: bool = True):
        """Initialize the ConflictAwarenessExtractor.
        
        Args:
            include_hedging: Whether to include hedging as a conflict indicator.
        """
        self.include_hedging = include_hedging
    
    def extract(self, response: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract conflict awareness indicators from a response.
        
        Args:
            response: Raw LLM response text.
            context: Optional context (unused).
        
        Returns:
            Dictionary with:
                - "is_conflict_aware": Boolean indicating if conflict is recognized
                - "conflict_keywords": List of matched conflict keywords
                - "perspective_keywords": List of matched perspective keywords
                - "hedging_keywords": List of matched hedging keywords (if enabled)
                - "awareness_score": Numeric score (0-1) of conflict awareness
        """
        text = response.lower()
        
        conflict_matches = self._find_matches(text, self.CONFLICT_INDICATORS)
        perspective_matches = self._find_matches(text, self.PERSPECTIVE_INDICATORS)
        hedging_matches = self._find_matches(text, self.HEDGING_INDICATORS) if self.include_hedging else []
        
        # Calculate awareness score
        # Conflict keywords are strongest signal, perspective keywords medium, hedging weak
        score = 0.0
        if conflict_matches:
            score += min(len(conflict_matches) * 0.3, 0.6)
        if perspective_matches:
            score += min(len(perspective_matches) * 0.2, 0.3)
        if hedging_matches and self.include_hedging:
            score += min(len(hedging_matches) * 0.05, 0.1)
        
        score = min(score, 1.0)
        
        result = {
            "is_conflict_aware": score >= 0.3 or len(conflict_matches) >= 1,
            "conflict_keywords": conflict_matches,
            "perspective_keywords": perspective_matches,
            "awareness_score": round(score, 3),
        }
        
        if self.include_hedging:
            result["hedging_keywords"] = hedging_matches
        
        return result
    
    def _find_matches(self, text: str, keywords: list[str]) -> list[str]:
        """Find all matching keywords in text."""
        matches = []
        for keyword in keywords:
            pattern = rf"\b{re.escape(keyword.lower())}\b"
            if re.search(pattern, text):
                matches.append(keyword)
        return matches
