"""Data settings for ConflictMedQA evaluation.

Settings prepare instances by deciding what evidence to include.
They only handle data - no inference logic.

Three settings:
1. NoEvidenceSetting: Remove all evidence (test prior knowledge)
2. WithEvidenceSetting: Use benchmark's ground truth evidence
3. RAGSetting: Use retrieved evidence (plug in your retriever)

Usage:
    # Simple function retriever
    setting = RAGSetting(lambda q: ["evidence 1", "evidence 2"], top_k=5)
    
    # BaseRetriever with metadata
    from src.retrieve import ThreeStageAdapter
    retriever = ThreeStageAdapter(index_dir="...")
    setting = RAGSetting(retriever, top_k=10)
    
    prepared = setting.prepare(instance)
"""

from abc import ABC, abstractmethod
from typing import Callable, Any, TYPE_CHECKING

from .types import Instance

if TYPE_CHECKING:
    from ..retrieve.interface import BaseRetriever
    from ..rag.base import RAGMethod


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
            conflict_info=instance.conflict_info,
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
            conflict_info=instance.conflict_info,
            ground_truth=instance.ground_truth.copy(),
            metadata={
                **instance.metadata,
                "setting": self.name,
            },
        )


class RAGSetting(BaseSetting):
    """Use retrieved evidence from a retriever or RAG method.
    
    Supports multiple interfaces:
    1. RAGMethod: Full RAG workflow (e.g., BaselineRAG, HyDERAG)
    2. BaseRetriever: Direct retrieval with metadata preservation
    3. Callable[[str], list[str]]: Simple function for backward compatibility
    
    Args:
        retriever: RAGMethod, BaseRetriever, or callable.
        top_k: Number of passages (only used for BaseRetriever/callable).
    
    Example:
        # With RAGMethod (recommended)
        from src.retrieve import ThreeStageAdapter, BaselineRAG
        retriever = ThreeStageAdapter(index_dir="...")
        method = BaselineRAG(retriever, top_k=10)
        setting = RAGSetting(method)
        
        # With BaseRetriever
        retriever = ThreeStageAdapter(index_dir="...")
        setting = RAGSetting(retriever, top_k=10)
        
        # With simple function (backward compatible)
        def my_search(query: str) -> list[str]:
            return ["result 1", "result 2"]
        setting = RAGSetting(my_search, top_k=5)
    """
    
    def __init__(
        self, 
        retriever: "RAGMethod | BaseRetriever | Callable[[str], list[str]]", 
        top_k: int = 5
    ):
        self._retriever = retriever
        self.top_k = top_k
        
        # Detect interface type
        self._is_rag_method = hasattr(retriever, 'retrieve') and hasattr(retriever, 'name') and hasattr(retriever, 'as_retriever')
        self._has_retrieve_method = hasattr(retriever, 'retrieve') and callable(getattr(retriever, 'retrieve'))
    
    @property
    def name(self) -> str:
        if self._is_rag_method:
            return f"rag_{self._retriever.name}"
        return f"rag_top{self.top_k}"
    
    def prepare(self, instance: Instance) -> Instance:
        """Prepare instance with retrieved evidence.
        
        Args:
            instance: Original instance.
        
        Returns:
            New instance with retrieved evidence.
        """
        if self._is_rag_method:
            # Use RAGMethod interface
            evidence_objects = self._retriever.retrieve(instance.claim)
            evidence = [e.to_dict() for e in evidence_objects]
            retriever_info = self._retriever.name
        elif self._has_retrieve_method:
            # Use BaseRetriever interface
            evidence_objects = self._retriever.retrieve(instance.claim, self.top_k)
            evidence = [e.to_dict() for e in evidence_objects]
            retriever_info = repr(self._retriever)
        else:
            # Use simple callable
            texts = self._retriever(instance.claim)[:self.top_k]
            evidence = [{"text": t, "metadata": {}} for t in texts]
            retriever_info = "callable"
        
        return Instance(
            id=instance.id,
            claim=instance.claim,
            evidence=evidence,
            label=instance.label,
            conflict_info=instance.conflict_info,
            ground_truth={
                **instance.ground_truth,
                "original_evidence": instance.evidence,
            },
            metadata={
                **instance.metadata,
                "setting": self.name,
                "retriever": retriever_info,
            },
        )
    
    def prepare_batch(self, instances: list[Instance]) -> list[Instance]:
        """Prepare multiple instances with retrieved evidence.
        
        Uses batch retrieval if available for efficiency.
        """
        if self._is_rag_method and hasattr(self._retriever, 'retrieve_batch'):
            # Use RAGMethod batch retrieval
            queries = [inst.claim for inst in instances]
            all_evidence = self._retriever.retrieve_batch(queries)
            retriever_info = self._retriever.name
            
            prepared = []
            for inst, evidence_list in zip(instances, all_evidence):
                evidence = [e.to_dict() for e in evidence_list]
                prepared.append(Instance(
                    id=inst.id,
                    claim=inst.claim,
                    evidence=evidence,
                    label=inst.label,
                    conflict_info=inst.conflict_info,
                    ground_truth={
                        **inst.ground_truth,
                        "original_evidence": inst.evidence,
                    },
                    metadata={
                        **inst.metadata,
                        "setting": self.name,
                        "retriever": retriever_info,
                    },
                ))
            return prepared
        elif self._has_retrieve_method and hasattr(self._retriever, 'retrieve_batch'):
            # Use BaseRetriever batch retrieval
            queries = [inst.claim for inst in instances]
            all_evidence = self._retriever.retrieve_batch(queries, self.top_k)
            
            prepared = []
            for inst, evidence_list in zip(instances, all_evidence):
                evidence = [e.to_dict() for e in evidence_list]
                prepared.append(Instance(
                    id=inst.id,
                    claim=inst.claim,
                    evidence=evidence,
                    label=inst.label,
                    conflict_info=inst.conflict_info,
                    ground_truth={
                        **inst.ground_truth,
                        "original_evidence": inst.evidence,
                    },
                    metadata={
                        **inst.metadata,
                        "setting": self.name,
                        "retriever": repr(self._retriever),
                    },
                ))
            return prepared
        else:
            # Fall back to sequential
            return [self.prepare(inst) for inst in instances]
    
    def __repr__(self) -> str:
        if self._is_rag_method:
            return f"RAGSetting(method={self._retriever})"
        return f"RAGSetting(top_k={self.top_k}, retriever={repr(self._retriever)})"