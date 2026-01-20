#!/usr/bin/env python
"""
Voting mechanisms for label aggregation.
"""
from dataclasses import dataclass
from typing import Optional, Callable
from collections import Counter


@dataclass
class Vote:
    """A single vote."""
    label: str
    model: str
    confidence: Optional[float] = None


@dataclass
class VotingResult:
    """Result of vote aggregation."""
    label: str  # Final label, or "NO_CONSENSUS" if no agreement
    votes: list[Vote]
    vote_counts: dict[str, int]
    agreement: float  # Fraction agreeing with final label


# Strategy function type: (votes, weights) -> VotingResult
StrategyFn = Callable[[list[Vote], dict[str, float]], VotingResult]


# ==================== Built-in Strategies ====================

def majority_vote(votes: list[Vote], weights: dict[str, float]) -> VotingResult:
    """Simple majority voting."""
    counts = Counter(v.label for v in votes)
    label, count = counts.most_common(1)[0]
    
    return VotingResult(
        label=label,
        votes=votes,
        vote_counts=dict(counts),
        agreement=count / len(votes),
    )


def unanimous_vote(votes: list[Vote], weights: dict[str, float]) -> VotingResult:
    """Require all votes to agree."""
    counts = Counter(v.label for v in votes)
    
    if len(counts) == 1:
        label = votes[0].label
        agreement = 1.0
    else:
        label = "NO_CONSENSUS"
        agreement = max(counts.values()) / len(votes)
    
    return VotingResult(
        label=label,
        votes=votes,
        vote_counts=dict(counts),
        agreement=agreement,
    )


def weighted_vote(votes: list[Vote], weights: dict[str, float]) -> VotingResult:
    """Weighted voting by model weights."""
    counts = Counter(v.label for v in votes)
    
    # Calculate weighted scores
    scores: dict[str, float] = {}
    for vote in votes:
        weight = weights.get(vote.model, 1.0)
        if vote.label not in scores:
            scores[vote.label] = 0.0
        scores[vote.label] += weight
    
    # Find winner
    label = max(scores, key=scores.get)
    total_weight = sum(scores.values())
    
    return VotingResult(
        label=label,
        votes=votes,
        vote_counts=dict(counts),
        agreement=scores[label] / total_weight if total_weight > 0 else 0.0,
    )


# ==================== Registry ====================

STRATEGY_REGISTRY: dict[str, StrategyFn] = {
    "majority": majority_vote,
    "unanimous": unanimous_vote,
    "weighted": weighted_vote,
}


def register_strategy(name: str):
    """Decorator to register a custom strategy."""
    def decorator(fn: StrategyFn):
        STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


def list_strategies() -> list[str]:
    """List available strategies."""
    return list(STRATEGY_REGISTRY.keys())


# ==================== Aggregator ====================

class VotingAggregator:
    """Aggregate votes using different strategies."""
    
    def __init__(
        self,
        strategy: str = "majority",
        weights: Optional[dict[str, float]] = None,
    ):
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy}. Available: {list_strategies()}")
        
        self.strategy = strategy
        self.weights = weights or {}
        self._fn = STRATEGY_REGISTRY[strategy]
    
    def aggregate(self, votes: list[Vote]) -> VotingResult:
        """Aggregate votes into a final result."""
        if not votes:
            return VotingResult(
                label="NO_CONSENSUS",
                votes=[],
                vote_counts={},
                agreement=0.0,
            )
        
        return self._fn(votes, self.weights)