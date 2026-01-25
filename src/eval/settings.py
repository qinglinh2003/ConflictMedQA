"""Data settings for ConflictMedQA evaluation.

Settings prepare instances by deciding what evidence to include.
They only handle data - no inference logic.

Three settings:
1. NoEvidenceSetting: Remove all evidence (test prior knowledge)
2. WithEvidenceSetting: Use benchmark's ground truth evidence
3. RAGSetting: Use retrieved evidence (plug in your retriever)

Usage:
    setting = WithEvidenceSetting()
    prepared = setting.prepare(instance)
    # Now use prepared instance with any inference method
"""

from abc import ABC, abstractmethod
from typing import Callable

from .types import Instance


class BaseSetting(ABC):
    """Abstract base class for data settings."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Setting name."""
        pass
    
    @abstractmethod
    def prepare(self, instance: Instance) -> Instance:
        """Prepare an instance (modify evidence as needed).
        
        Args:
            instance: Original instance from benchmark.
            
        Returns:
            New instance with appropriate evidence.
        """
        pass
    
    def prepare_batch(self, instances: list[Instance]) -> list[Instance]:
        """Prepare multiple instances."""
        return [self.prepare(inst) for inst in instances]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NoEvidenceSetting(BaseSetting):
    """Remove all evidence - test model's prior knowledge."""
    
    @property
    def name(self) -> str:
        return "no_evidence"
    
    def prepare(self, instance: Instance) -> Instance:
        return Instance(
            id=instance.id,
            claim=instance.claim,
            evidence=[],
            label=instance.label,
            ground_truth=instance.ground_truth.copy(),
            metadata={
                **instance.metadata,
                "setting": self.name,
            },
        )


class WithEvidenceSetting(BaseSetting):
    """Use ground truth evidence from benchmark.
    
    Args:
        max_evidence: Limit number of evidence passages (None = no limit).
        shuffle: Randomize evidence order.
    """
    
    def __init__(self, max_evidence: int | None = None, shuffle: bool = False):
        self.max_evidence = max_evidence
        self.shuffle = shuffle
    
    @property
    def name(self) -> str:
        name = "with_evidence"
        if self.max_evidence:
            name += f"_max{self.max_evidence}"
        if self.shuffle:
            name += "_shuffled"
        return name
    
    def prepare(self, instance: Instance) -> Instance:
        import random
        
        evidence = list(instance.evidence)
        
        if self.shuffle:
            random.shuffle(evidence)
        
        if self.max_evidence:
            evidence = evidence[:self.max_evidence]
        
        return Instance(
            id=instance.id,
            claim=instance.claim,
            evidence=evidence,
            label=instance.label,
            ground_truth=instance.ground_truth.copy(),
            metadata={
                **instance.metadata,
                "setting": self.name,
            },
        )


class RAGSetting(BaseSetting):
    """Use retrieved evidence.
    
    Args:
        retriever: Function (query: str) -> list[str] that returns evidence.
        top_k: Number of passages to retrieve.
    """
    
    def __init__(self, retriever: Callable[[str], list[str]], top_k: int = 5):
        self.retriever = retriever
        self.top_k = top_k
    
    @property
    def name(self) -> str:
        return f"rag_top{self.top_k}"
    
    def prepare(self, instance: Instance) -> Instance:
        retrieved = self.retriever(instance.claim)[:self.top_k]
        
        return Instance(
            id=instance.id,
            claim=instance.claim,
            evidence=retrieved,
            label=instance.label,
            ground_truth={
                **instance.ground_truth,
                "original_evidence": instance.evidence,
            },
            metadata={
                **instance.metadata,
                "setting": self.name,
            },
        )
