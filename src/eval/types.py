"""Core data types for the ConflictMedQA evaluation module."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class Label(str, Enum):
    """Labels for evidence conflict status."""
    SUPPORT = "support"
    CONTRADICT = "contradict"
    CONFLICT = "conflict"
    NO_CONFLICT = "no_conflict"
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"


class ConflictType(str, Enum):
    """Types of conflict in evidence."""
    OUTCOME_CONFLICT = "outcome_conflict"
    METHODOLOGICAL = "methodological_conflict"
    SUBGROUP = "subgroup_conflict"
    TEMPORAL = "temporal_conflict"
    DOSAGE = "dosage_conflict"
    INTERPRETATION = "interpretation_conflict"
    NO_CONFLICT = "no_conflict"


@dataclass
class Evidence:
    """A single piece of evidence with optional metadata."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {"text": self.text, "metadata": self.metadata}
    
    @classmethod
    def from_dict(cls, data: dict | str) -> "Evidence":
        if isinstance(data, str):
            return cls(text=data)
        return cls(text=data.get("text", ""), metadata=data.get("metadata", {}))


@dataclass
class ConflictInfo:
    """Flexible structure for conflict information."""
    type: ConflictType | str
    description: str = ""
    pairs: list[dict] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "type": self.type.value if isinstance(self.type, ConflictType) else self.type,
            "description": self.description,
            "pairs": self.pairs,
            **self.extra,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ConflictInfo":
        if data is None:
            return cls.no_conflict()
        data = data.copy()
        type_ = data.pop("type", "no_conflict")
        description = data.pop("description", "")
        pairs = data.pop("pairs", [])
        return cls(type=type_, description=description, pairs=pairs, extra=data)
    
    @classmethod
    def no_conflict(cls) -> "ConflictInfo":
        return cls(type=ConflictType.NO_CONFLICT, description="All evidence agrees")
    
    @property
    def has_conflict(self) -> bool:
        t = self.type.value if isinstance(self.type, ConflictType) else self.type
        return t != "no_conflict"


@dataclass
class Instance:
    """A single evaluation instance."""
    id: str
    claim: str
    evidence: list[str]
    label: Label | str
    conflict_info: ConflictInfo | dict | None = None
    ground_truth: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self._evidence_objects = []
        normalized = []
        for e in self.evidence:
            if isinstance(e, Evidence):
                self._evidence_objects.append(e)
                normalized.append(e.text)
            elif isinstance(e, dict) and "text" in e:
                ev = Evidence.from_dict(e)
                self._evidence_objects.append(ev)
                normalized.append(ev.text)
            else:
                self._evidence_objects.append(Evidence(text=str(e)))
                normalized.append(str(e))
        self.evidence = normalized
        
        if isinstance(self.conflict_info, dict):
            self.conflict_info = ConflictInfo.from_dict(self.conflict_info)
    
    @property
    def has_conflict(self) -> bool:
        if self.conflict_info is not None:
            if isinstance(self.conflict_info, ConflictInfo):
                return self.conflict_info.has_conflict
            return self.conflict_info.get("type", "no_conflict") != "no_conflict"
        label = self.label.value if isinstance(self.label, Label) else self.label
        return label in ("conflict", "contradict")
    
    @property
    def conflict_type_str(self) -> str | None:
        if self.conflict_info is None:
            return None
        if isinstance(self.conflict_info, ConflictInfo):
            t = self.conflict_info.type
            return t.value if isinstance(t, ConflictType) else t
        return self.conflict_info.get("type")
    
    def get_evidence_objects(self) -> list[Evidence]:
        return self._evidence_objects
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "claim": self.claim,
            "evidence": [e.to_dict() for e in self._evidence_objects],
            "label": self.label.value if isinstance(self.label, Label) else self.label,
            "conflict_info": self.conflict_info.to_dict() if isinstance(self.conflict_info, ConflictInfo) else self.conflict_info,
            "ground_truth": self.ground_truth,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Instance":
        label = data.get("label", "conflict")
        if isinstance(label, str):
            try:
                label = Label(label)
            except ValueError:
                pass
        
        return cls(
            id=str(data.get("id", "")),
            claim=data.get("claim", data.get("question", data.get("prompt", ""))),
            evidence=data.get("evidence", []),
            label=label,
            conflict_info=data.get("conflict_info"),
            ground_truth=data.get("ground_truth", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MetricResult:
    """Result from computing a single metric."""
    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "details": self.details}


@dataclass
class EvalResult:
    """Complete evaluation result for a single instance."""
    instance_id: str
    question_type: str
    prompt: str
    response: str
    extracted: dict[str, Any]
    metrics: list[MetricResult]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "question_type": self.question_type,
            "prompt": self.prompt,
            "response": self.response,
            "extracted": self.extracted,
            "metrics": [m.to_dict() for m in self.metrics],
            "error": self.error,
        }


@dataclass
class AggregatedResults:
    """Aggregated results across multiple instances."""
    question_type: str
    num_instances: int
    num_errors: int
    metrics: dict[str, dict[str, float]]
    per_instance: list[EvalResult] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "question_type": self.question_type,
            "num_instances": self.num_instances,
            "num_errors": self.num_errors,
            "metrics": self.metrics,
        }
        if self.per_instance:
            result["per_instance"] = [r.to_dict() for r in self.per_instance]
        return result