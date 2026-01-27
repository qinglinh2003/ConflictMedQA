"""Load data files into Instance objects.

This module does ONE thing: file → list[Instance]

Usage:
    from src.eval.loaders import load_instances
    
    instances = load_instances("data/conflicts_final.jsonl")
"""

import json
from pathlib import Path

from .types import Instance, ConflictInfo, ConflictType, Label


def load_instances(path: str | Path) -> list[Instance]:
    """Load JSONL file into Instance list.
    
    Args:
        path: Path to JSONL file.
    
    Returns:
        List of Instance objects.
    """
    instances = []
    
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            instance = _to_instance(item)
            instances.append(instance)
    
    return instances


def _to_instance(item: dict) -> Instance:
    """Convert a dict to Instance."""
    
    # doc_labels lookup
    doc_label_map = {
        dl.get("doc_id", ""): {
            "label": dl.get("label"),
            "agreement": dl.get("agreement"),
            "votes": dl.get("votes", []),
        }
        for dl in item.get("doc_labels", [])
    }
    
    # retrieved → evidence
    evidence = []
    for doc in item.get("retrieved", []):
        chunk_id = doc.get("chunk_id", "")
        metadata = {
            "chunk_id": chunk_id,
            "pmcid": doc.get("pmcid"),
            "section": doc.get("section"),
            "rank": doc.get("rank"),
            "score": doc.get("score"),
        }
        if chunk_id in doc_label_map:
            metadata["doc_label"] = doc_label_map[chunk_id]
        
        evidence.append({"text": doc.get("text", ""), "metadata": metadata})
    
    # conflict info
    label_counts = item.get("label_counts", {})
    num_positive = item.get("num_positive", label_counts.get("positive", 0))
    num_negative = item.get("num_negative", label_counts.get("negative", 0))
    has_conflict = num_positive > 0 and num_negative > 0
    
    conflict_info = ConflictInfo(
        type=ConflictType.OUTCOME_CONFLICT if has_conflict else ConflictType.NO_CONFLICT,
        description=f"+{num_positive} / -{num_negative}" if has_conflict else "No conflict",
        extra={"label_counts": label_counts, "num_positive": num_positive, "num_negative": num_negative},
    )
    
    # ground truth
    ground_truth = {}
    if "Label" in item:
        ground_truth["gold_label"] = item["Label"]
    if "Annotations" in item:
        ground_truth["gold_evidence"] = item["Annotations"]
    
    # metadata
    metadata = {}
    if any(k in item for k in ["Outcome", "Intervention", "Comparator"]):
        metadata["pico"] = {
            "outcome": item.get("Outcome"),
            "intervention": item.get("Intervention"),
            "comparator": item.get("Comparator"),
        }
    
    return Instance(
        id=str(item.get("PromptID", item.get("id", ""))),
        claim=item.get("Prompt", item.get("query", "")),
        evidence=evidence,
        label=Label.CONFLICT if has_conflict else Label.NO_CONFLICT,
        conflict_info=conflict_info,
        ground_truth=ground_truth,
        metadata=metadata,
    )