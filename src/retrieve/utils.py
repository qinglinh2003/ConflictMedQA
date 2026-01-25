"""
Utility functions for batch retrieval operations.

This module provides tools for:
- Batch retrieval to JSONL
- Loading retrieved results
- Converting between formats

Example:
    # Offline batch retrieval
    from src.retrieve import ThreeStageAdapter
    from src.retrieve.utils import batch_retrieve_to_jsonl
    
    retriever = ThreeStageAdapter(index_dir="data/pyserini/index_full")
    
    batch_retrieve_to_jsonl(
        retriever=retriever,
        input_file="data/prompts.csv",
        output_file="data/prompts_retrieved.jsonl",
        query_field="Prompt",
        top_k=10,
    )
"""

import json
import os
import logging
from typing import Any
from dataclasses import asdict

from tqdm import tqdm

from .interface import BaseRetriever, Evidence

logger = logging.getLogger(__name__)


def batch_retrieve_to_jsonl(
    retriever: BaseRetriever,
    input_file: str,
    output_file: str,
    query_field: str = "Prompt",
    id_field: str | None = "PromptID",
    top_k: int = 10,
    resume: bool = True,
    batch_size: int = 1,
) -> int:
    """Run batch retrieval and save results to JSONL.
    
    Args:
        retriever: Retriever to use.
        input_file: Input file (CSV or JSONL).
        output_file: Output JSONL file.
        query_field: Field name for query text.
        id_field: Field name for item ID (optional).
        top_k: Number of results per query.
        resume: Skip already processed items.
        batch_size: Batch size for retrieval (if supported).
    
    Returns:
        Number of items processed.
    """
    import pandas as pd
    
    # Load input
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
        items = df.to_dict("records")
    elif input_file.endswith(".jsonl"):
        items = []
        with open(input_file, "r") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported input format: {input_file}")
    
    logger.info(f"Loaded {len(items)} items from {input_file}")
    
    # Load existing results for resume
    done_ids = set()
    if resume and os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if id_field and id_field in data:
                        done_ids.add(str(data[id_field]))
        logger.info(f"Resume mode: {len(done_ids)} items already done")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # Process items
    processed = 0
    with open(output_file, "a") as f_out:
        for item in tqdm(items, desc="Retrieving"):
            # Skip if done
            if id_field and str(item.get(id_field, "")) in done_ids:
                continue
            
            query = item.get(query_field, "")
            if not query:
                continue
            
            # Retrieve
            evidence_list = retriever.retrieve(query, top_k)
            
            # Build output record
            record = dict(item)
            record["retrieved"] = [
                {
                    "rank": i + 1,
                    "text": e.text,
                    **e.metadata,
                }
                for i, e in enumerate(evidence_list)
            ]
            
            # Handle NaN values for JSON serialization
            record = _sanitize_for_json(record)
            
            # Write
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            
            processed += 1
    
    logger.info(f"Processed {processed} items, saved to {output_file}")
    return processed


def load_retrieved_jsonl(
    path: str,
    query_field: str = "Prompt",
    results_field: str = "retrieved",
) -> dict[str, list[Evidence]]:
    """Load retrieved results from JSONL into a lookup dict.
    
    Args:
        path: Path to JSONL file.
        query_field: Field name for query.
        results_field: Field name for results.
    
    Returns:
        Dict mapping query strings to lists of Evidence objects.
    """
    results_map = {}
    
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            query = data.get(query_field, "")
            results = data.get(results_field, [])
            
            if not query:
                continue
            
            evidence_list = []
            for r in results:
                if isinstance(r, str):
                    evidence_list.append(Evidence(text=r))
                elif isinstance(r, dict):
                    text = r.get("text", "")
                    metadata = {k: v for k, v in r.items() if k != "text"}
                    evidence_list.append(Evidence(text=text, metadata=metadata))
            
            results_map[query] = evidence_list
    
    return results_map


def convert_retrieved_to_instances(
    input_file: str,
    output_file: str,
    query_field: str = "Prompt",
    id_field: str = "PromptID",
    results_field: str = "retrieved",
    label_field: str | None = "Label",
) -> int:
    """Convert retrieved JSONL to evaluation instance format.
    
    This creates instances ready for the evaluation pipeline.
    
    Args:
        input_file: Input JSONL with retrieved results.
        output_file: Output JSONL in Instance format.
        query_field: Field for query/claim.
        id_field: Field for instance ID.
        results_field: Field for retrieved results.
        label_field: Field for label (optional).
    
    Returns:
        Number of instances created.
    """
    count = 0
    
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            
            data = json.loads(line)
            
            # Build instance
            instance = {
                "id": str(data.get(id_field, count)),
                "question": data.get(query_field, ""),
                "evidence": [
                    {"text": r.get("text", ""), "metadata": {k: v for k, v in r.items() if k != "text"}}
                    for r in data.get(results_field, [])
                ],
            }
            
            # Add label if available
            if label_field and label_field in data:
                instance["ground_truth"] = {"gold_label": data[label_field]}
            
            # Add other metadata
            instance["metadata"] = {
                k: v for k, v in data.items()
                if k not in [id_field, query_field, results_field, label_field]
            }
            
            f_out.write(json.dumps(instance, ensure_ascii=False) + "\n")
            count += 1
    
    logger.info(f"Created {count} instances in {output_file}")
    return count


def merge_shards(
    shard_pattern: str,
    output_file: str,
    num_shards: int,
) -> int:
    """Merge sharded retrieval outputs.
    
    Args:
        shard_pattern: Pattern with {shard} placeholder, e.g., "output_shard{shard}.jsonl"
        output_file: Output merged file.
        num_shards: Number of shards.
    
    Returns:
        Total number of items merged.
    """
    count = 0
    
    with open(output_file, "w") as f_out:
        for shard in range(num_shards):
            shard_file = shard_pattern.format(shard=shard)
            if not os.path.exists(shard_file):
                logger.warning(f"Shard file not found: {shard_file}")
                continue
            
            with open(shard_file, "r") as f_in:
                for line in f_in:
                    if line.strip():
                        f_out.write(line)
                        count += 1
    
    logger.info(f"Merged {count} items from {num_shards} shards to {output_file}")
    return count


def _sanitize_for_json(obj: Any) -> Any:
    """Sanitize object for JSON serialization (handle NaN, etc.)."""
    import math
    
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj