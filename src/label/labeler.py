#!/usr/bin/env python
"""
Core labeling module.

Integrates: prompts + llm_factory + voting + I/O
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm

try:
    from .prompts import LabelInput, LabelOutput, BasePrompt, get_prompt
    from .llm_factory import BaseLLM, LLMFactory
    from .voting import Vote, VotingResult, VotingAggregator
except ImportError:
    from prompts import LabelInput, LabelOutput, BasePrompt, get_prompt
    from llm_factory import BaseLLM, LLMFactory
    from voting import Vote, VotingResult, VotingAggregator


@dataclass
class DocLabel:
    """Label result for a single document."""
    doc_id: str
    label: str
    agreement: float
    votes: list[dict]
    vote_counts: dict[str, int]


@dataclass
class LabelResult:
    """Result for a single query with multiple documents."""
    item_id: str
    query: str
    doc_labels: list[DocLabel]


class Labeler:
    """
    Main labeling class.
    
    Usage:
        labeler = Labeler(
            llms=[
                LLMFactory.create("openai", "gpt-4o"),
                LLMFactory.create("anthropic", "claude-sonnet-4-20250514"),
            ],
            prompt=get_prompt("medical_evidence"),
            voting=VotingAggregator("majority"),
        )
        
        result = labeler.label(
            item_id="001",
            query="...",
            documents=[
                {"doc_id": "PMC001_ABSTRACT", "text": "..."},
                {"doc_id": "PMC002_RESULTS", "text": "..."},
            ],
            label_choices=["positive", "negative", "no_significant_difference", "irrelevant"],
        )
    """
    
    def __init__(
        self,
        llms: list[BaseLLM],
        prompt: BasePrompt,
        voting: VotingAggregator,
    ):
        self.llms = llms
        self.prompt = prompt
        self.voting = voting
    
    def label_single_doc(
        self,
        query: str,
        document: str,
        label_choices: list[str],
    ) -> VotingResult:
        """Label a single (query, document) pair with all LLMs."""
        
        # Build input
        input = LabelInput(
            query=query,
            document=document,
            label_choices=label_choices,
        )
        
        # Format prompt
        text = self.prompt.format(input)
        system = self.prompt.system
        
        # Collect votes from all LLMs
        votes = []
        for llm in self.llms:
            try:
                response = llm.generate(text, system=system)
                output = self.prompt.parse(response.text, label_choices)
                
                votes.append(Vote(
                    label=output.label,
                    model=llm.config.model,
                    confidence=output.confidence,
                ))
            except Exception as e:
                votes.append(Vote(
                    label="ERROR",
                    model=llm.config.model,
                    confidence=None,
                ))
        
        # Aggregate votes
        return self.voting.aggregate(votes)
    
    def label(
        self,
        item_id: str,
        query: str,
        documents: list[dict],  # [{"doc_id": "...", "text": "..."}, ...]
        label_choices: list[str],
    ) -> LabelResult:
        """Label all documents for a single query."""
        
        doc_labels = []
        for doc in documents:
            doc_id = doc.get("doc_id", doc.get("chunk_id", ""))
            doc_text = doc.get("text", "")
            
            result = self.label_single_doc(
                query=query,
                document=doc_text,
                label_choices=label_choices,
            )
            
            doc_labels.append(DocLabel(
                doc_id=doc_id,
                label=result.label,
                agreement=result.agreement,
                votes=[asdict(v) for v in result.votes],
                vote_counts=result.vote_counts,
            ))
        
        return LabelResult(
            item_id=item_id,
            query=query,
            doc_labels=doc_labels,
        )
    
    def label_batch(
        self,
        items: list[dict],
        id_field: str = "PromptID",
        query_field: str = "Prompt",
        docs_field: str = "retrieved",
        label_choices: list[str] = None,
        output_path: Optional[str] = None,
        resume: bool = True,
    ) -> list[LabelResult]:
        """
        Label a batch of items.
        
        Args:
            items: List of dicts with query and documents
            id_field: Field name for item ID
            query_field: Field name for query
            docs_field: Field name for documents list
            label_choices: Label options
            output_path: Path to save results (JSONL)
            resume: Skip already processed items
        
        Returns:
            List of LabelResult
        """
        label_choices = label_choices or ["positive", "negative", "no_significant_difference", "irrelevant"]
        
        # Load existing results for resume
        done_ids = set()
        if resume and output_path and os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    obj = json.loads(line)
                    done_ids.add(str(obj.get("item_id", "")))
        
        # Open output file
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            f_out = open(output_path, "a")
        else:
            f_out = None
        
        results = []
        
        try:
            for item in tqdm(items, desc="Labeling"):
                item_id = str(item.get(id_field, ""))
                
                # Skip if done
                if item_id in done_ids:
                    continue
                
                query = item.get(query_field, "")
                
                # Extract documents (keep as list of dicts)
                docs_raw = item.get(docs_field, [])
                documents = []
                for d in docs_raw:
                    if isinstance(d, dict):
                        documents.append({
                            "doc_id": d.get("chunk_id", d.get("doc_id", "")),
                            "text": d.get("text", ""),
                        })
                    elif isinstance(d, str):
                        documents.append({"doc_id": "", "text": d})
                
                # Label
                result = self.label(
                    item_id=item_id,
                    query=query,
                    documents=documents,
                    label_choices=label_choices,
                )
                
                results.append(result)
                
                # Write to file
                if f_out:
                    f_out.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                    f_out.flush()
        
        finally:
            if f_out:
                f_out.close()
        
        return results


# ==================== I/O Utilities ====================

def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file."""
    items = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: str):
    """Save to JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_results(path: str) -> list[LabelResult]:
    """Load LabelResult from JSONL."""
    items = load_jsonl(path)
    results = []
    for item in items:
        doc_labels = [DocLabel(**dl) for dl in item.get("doc_labels", [])]
        results.append(LabelResult(
            item_id=item["item_id"],
            query=item["query"],
            doc_labels=doc_labels,
        ))
    return results