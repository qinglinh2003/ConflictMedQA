#!/usr/bin/env python3
"""Command-line interface for ConflictMedQA evaluation."""

import argparse
import json
import logging
import sys
from pathlib import Path

from .types import Instance, Label
from .evaluator import Evaluator, EvaluatorConfig
from .question_types import (
    DirectQuestion,
    ForcedStanceQuestion,
    ConfidenceQuestion,
    ConsensusQuestion,
)
from .llm.base import GenerationConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


QUESTION_TYPES = {
    "direct": DirectQuestion,
    "forced_stance": ForcedStanceQuestion,
    "confidence": ConfidenceQuestion,
    "consensus": ConsensusQuestion,
}


def load_instances(path: str) -> list[Instance]:
    """Load instances from a JSON or JSONL file."""
    path = Path(path)
    instances = []
    
    with open(path, "r") as f:
        if path.suffix == ".jsonl":
            for line in f:
                data = json.loads(line.strip())
                instances.append(Instance.from_dict(data))
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    instances.append(Instance.from_dict(item))
            else:
                instances.append(Instance.from_dict(data))
    
    return instances


def create_llm(args):
    """Create LLM backend based on arguments."""
    if args.backend == "transformers":
        from .llm import TransformersLLM
        
        return TransformersLLM(
            model_name=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            load_in_8bit=args.load_8bit,
            load_in_4bit=args.load_4bit,
        )
    
    elif args.backend == "vllm":
        from .llm import VLLMLlm
        
        return VLLMLlm(
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory,
            trust_remote_code=args.trust_remote_code,
        )
    
    elif args.backend == "api":
        from .llm import APILlm
        
        return APILlm(
            model_name=args.model,
            provider=args.api_provider,
            api_key=args.api_key,
            base_url=args.api_base_url,
        )
    
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


def create_question_type(args):
    """Create question type based on arguments."""
    qt_class = QUESTION_TYPES.get(args.question_type)
    if qt_class is None:
        raise ValueError(f"Unknown question type: {args.question_type}")
    
    kwargs = {}
    
    if args.question_type == "forced_stance":
        kwargs["allow_uncertain"] = args.allow_uncertain
    elif args.question_type == "consensus":
        kwargs["detailed_format"] = args.detailed_format
    elif args.question_type == "confidence":
        kwargs["format_style"] = args.confidence_format
    
    return qt_class(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="ConflictMedQA Evaluation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input file path (JSON or JSONL with instances)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Model name or path",
    )
    
    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results.json",
        help="Output file path",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["json", "jsonl"],
        default="json",
        help="Output format",
    )
    
    # Backend selection
    parser.add_argument(
        "--backend", "-b",
        type=str,
        choices=["transformers", "vllm", "api"],
        default="transformers",
        help="LLM backend to use",
    )
    
    # Question type
    parser.add_argument(
        "--question-type", "-q",
        type=str,
        choices=list(QUESTION_TYPES.keys()),
        default="direct",
        help="Question type for evaluation",
    )
    
    # Generation config
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter",
    )
    
    # Evaluator config
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--no-evidence",
        action="store_true",
        help="Exclude evidence from prompts",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    
    # Transformers-specific
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for transformers backend",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Data type for model weights",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code in model",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit precision",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit precision",
    )
    
    # vLLM-specific
    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM",
    )
    
    # API-specific
    parser.add_argument(
        "--api-provider",
        type=str,
        choices=["openai", "anthropic", "openai_compatible"],
        default="openai",
        help="API provider",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or use environment variable)",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="Base URL for API (for compatible APIs)",
    )
    
    # Question type specific
    parser.add_argument(
        "--allow-uncertain",
        action="store_true",
        help="Allow uncertain option in forced_stance",
    )
    parser.add_argument(
        "--detailed-format",
        action="store_true",
        help="Use detailed format for consensus questions",
    )
    parser.add_argument(
        "--confidence-format",
        type=str,
        choices=["structured", "numeric"],
        default="structured",
        help="Format style for confidence questions",
    )
    
    args = parser.parse_args()
    
    # Load instances
    logger.info(f"Loading instances from {args.input}")
    instances = load_instances(args.input)
    logger.info(f"Loaded {len(instances)} instances")
    
    # Create LLM
    logger.info(f"Creating {args.backend} backend with model {args.model}")
    llm = create_llm(args)
    
    # Set generation config
    llm.generation_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
    )
    
    # Create question type
    question_type = create_question_type(args)
    logger.info(f"Using question type: {question_type.name}")
    
    # Create evaluator
    eval_config = EvaluatorConfig(
        include_evidence=not args.no_evidence,
        batch_size=args.batch_size,
        show_progress=not args.no_progress,
    )
    evaluator = Evaluator(llm, eval_config)
    
    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluator.evaluate(instances, question_type)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Evaluation Results: {question_type.name}")
    print("=" * 60)
    print(f"Total instances: {results.num_instances}")
    print(f"Errors: {results.num_errors}")
    print("\nMetrics:")
    for name, stats in results.metrics.items():
        print(f"  {name}:")
        print(f"    mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"    count: {stats['count']}")
    print("=" * 60)
    
    # Save results
    evaluator.save_results(results, args.output, args.output_format)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
