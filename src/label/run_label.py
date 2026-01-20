#!/usr/bin/env python
"""
Main entry point for labeling.

Usage:
    # OpenAI
    python run_label.py \
        --input ../../data/retrieved/prompts_retrieved.jsonl \
        --output ../../data/labels/labels.jsonl \
        --provider openai --model gpt-4o

    # vLLM offline (HPC)
    python run_label.py \
        --input ../../data/retrieved/prompts_retrieved.jsonl \
        --output ../../data/labels/labels.jsonl \
        --provider vllm_offline --model Qwen/Qwen2.5-32B-Instruct

    # Multi-model voting
    python run_label.py \
        --input ../../data/retrieved/prompts_retrieved.jsonl \
        --output ../../data/labels/labels.jsonl \
        --provider openai openai --model gpt-4o gpt-4o-mini \
        --voting majority
"""
import argparse
import logging
import os
from datetime import datetime

from labeler import Labeler, load_jsonl
from llm_factory import LLMFactory
from prompts import get_prompt, list_prompts
from voting import VotingAggregator, list_strategies


def setup_logging(log_path: str = None):
    handlers = [logging.StreamHandler()]
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def main():
    parser = argparse.ArgumentParser(description="Label medical evidence")
    
    # I/O
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    
    # Data fields
    parser.add_argument("--id-field", default="PromptID")
    parser.add_argument("--query-field", default="Prompt")
    parser.add_argument("--docs-field", default="retrieved")
    
    # Labels
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["positive", "negative", "no_significant_difference", "irrelevant"],
        help="Label choices",
    )
    
    # Models (can specify multiple for voting)
    parser.add_argument(
        "--provider",
        nargs="+",
        default=["vllm_offline"],
        help="LLM providers",
    )
    parser.add_argument(
        "--model",
        nargs="+",
        default=["Qwen/Qwen2.5-32B-Instruct"],
        help="Model names (one per provider)",
    )
    parser.add_argument("--tensor-parallel", type=int, default=1, help="GPUs for vllm_offline")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    
    # Prompt
    parser.add_argument(
        "--prompt",
        default="medical_evidence",
        choices=list_prompts(),
        help="Prompt template",
    )
    
    # Voting
    parser.add_argument(
        "--voting",
        default="majority",
        choices=list_strategies(),
        help="Voting strategy",
    )
    
    # Execution
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--max-items", type=int, help="Limit items (for testing)")
    
    args = parser.parse_args()
    
    # Validate
    if len(args.provider) != len(args.model):
        parser.error("Number of providers must match number of models")
    
    # Setup logging
    log_path = args.output.replace(".jsonl", ".log")
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    
    # Log config
    logger.info("=" * 60)
    logger.info("ConflictMedQA Labeling")
    logger.info("=" * 60)
    logger.info(f"Input:    {args.input}")
    logger.info(f"Output:   {args.output}")
    logger.info(f"Models:   {list(zip(args.provider, args.model))}")
    logger.info(f"Prompt:   {args.prompt}")
    logger.info(f"Voting:   {args.voting}")
    logger.info(f"Labels:   {args.labels}")
    logger.info(f"Resume:   {args.resume}")
    
    # Create LLMs
    logger.info("Loading models...")
    llms = []
    for provider, model in zip(args.provider, args.model):
        llm = LLMFactory.create(
            provider=provider,
            model=model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel,
        )
        llms.append(llm)
    
    # Create labeler
    labeler = Labeler(
        llms=llms,
        prompt=get_prompt(args.prompt),
        voting=VotingAggregator(args.voting),
    )
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    items = load_jsonl(args.input)
    logger.info(f"Loaded {len(items)} items")
    
    if args.max_items:
        items = items[:args.max_items]
        logger.info(f"Limited to {len(items)} items")
    
    # Run labeling
    logger.info("Starting labeling...")
    start = datetime.now()
    
    results = labeler.label_batch(
        items=items,
        id_field=args.id_field,
        query_field=args.query_field,
        docs_field=args.docs_field,
        label_choices=args.labels,
        output_path=args.output,
        resume=args.resume,
    )
    
    elapsed = datetime.now() - start
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Processed: {len(results)} items")
    logger.info(f"Time:      {elapsed}")
    logger.info(f"Output:    {args.output}")
    
    # Label distribution
    from collections import Counter
    all_labels = []
    for r in results:
        for dl in r.doc_labels:
            all_labels.append(dl.label)
    
    if all_labels:
        dist = Counter(all_labels)
        logger.info(f"Label distribution: {dict(dist)}")


if __name__ == "__main__":
    main()