"""Main Evaluator class for ConflictMedQA evaluation."""

import json
import logging
from pathlib import Path
from typing import Any, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from tqdm import tqdm

from .types import Instance, MetricResult, EvalResult, AggregatedResults, Label
from .question_types.base import BaseQuestionType
from .llm.base import BaseLLM, GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for the Evaluator.
    
    Attributes:
        include_evidence: Whether to include evidence in prompts.
        batch_size: Batch size for generation.
        show_progress: Whether to show progress bar.
        save_individual_results: Whether to save per-instance results.
        num_workers: Number of parallel workers (for API calls).
        retry_on_error: Whether to retry failed generations.
        max_retries: Maximum retry attempts.
    """
    include_evidence: bool = True
    batch_size: int = 8
    show_progress: bool = True
    save_individual_results: bool = True
    num_workers: int = 1
    retry_on_error: bool = True
    max_retries: int = 3


class Evaluator:
    """Main evaluator for ConflictMedQA.
    
    Orchestrates the evaluation pipeline:
    1. Format instances into prompts using QuestionType
    2. Generate responses using LLM backend
    3. Extract structured data from responses
    4. Compute metrics
    5. Aggregate results
    
    Example:
        ```python
        from eval import Evaluator, DirectQuestion, TransformersLLM
        
        llm = TransformersLLM("meta-llama/Llama-2-7b-chat-hf")
        evaluator = Evaluator(llm)
        
        results = evaluator.evaluate(
            instances=dataset,
            question_type=DirectQuestion(),
        )
        print(results.summary())
        ```
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        config: EvaluatorConfig | None = None,
    ):
        """Initialize the Evaluator.
        
        Args:
            llm: LLM backend for generation.
            config: Evaluator configuration.
        """
        self.llm = llm
        self.config = config or EvaluatorConfig()
    
    def evaluate(
        self,
        instances: list[Instance],
        question_type: BaseQuestionType,
        generation_config: GenerationConfig | None = None,
    ) -> AggregatedResults:
        """Run evaluation on a list of instances.
        
        Args:
            instances: List of evaluation instances.
            question_type: Question type to use.
            generation_config: Override LLM generation config.
        
        Returns:
            Aggregated evaluation results.
        """
        logger.info(
            f"Starting evaluation: {len(instances)} instances, "
            f"question_type={question_type.name}"
        )
        
        # Generate prompts
        prompts = [
            question_type.format(inst, include_evidence=self.config.include_evidence)
            for inst in instances
        ]
        
        # Generate responses
        responses = self._generate_responses(prompts, generation_config)
        
        # Evaluate each instance
        individual_results = []
        errors = 0
        
        iterator = zip(instances, prompts, responses)
        if self.config.show_progress:
            iterator = tqdm(
                list(iterator),
                desc="Evaluating",
                unit="instance"
            )
        
        for instance, prompt, response in iterator:
            try:
                result = self._evaluate_single(
                    instance=instance,
                    prompt=prompt,
                    response=response,
                    question_type=question_type,
                )
                individual_results.append(result)
                
                if result.error:
                    errors += 1
                    
            except Exception as e:
                logger.error(f"Error evaluating instance {instance.id}: {e}")
                errors += 1
                individual_results.append(EvalResult(
                    instance_id=instance.id,
                    question_type=question_type.name,
                    prompt=prompt,
                    response=response,
                    extracted={},
                    metrics=[],
                    error=str(e),
                ))
        
        # Aggregate results
        aggregated = self._aggregate_results(
            results=individual_results,
            question_type=question_type,
        )
        
        logger.info(
            f"Evaluation complete: {len(instances)} instances, "
            f"{errors} errors"
        )
        
        return aggregated
    
    def evaluate_streaming(
        self,
        instances: Iterator[Instance],
        question_type: BaseQuestionType,
        generation_config: GenerationConfig | None = None,
    ) -> Iterator[EvalResult]:
        """Evaluate instances in streaming fashion.
        
        Useful for very large datasets that don't fit in memory.
        
        Args:
            instances: Iterator of evaluation instances.
            question_type: Question type to use.
            generation_config: Override LLM generation config.
        
        Yields:
            Individual evaluation results.
        """
        batch = []
        
        for instance in instances:
            batch.append(instance)
            
            if len(batch) >= self.config.batch_size:
                yield from self._evaluate_batch(
                    batch, question_type, generation_config
                )
                batch = []
        
        # Process remaining
        if batch:
            yield from self._evaluate_batch(
                batch, question_type, generation_config
            )
    
    def _evaluate_batch(
        self,
        instances: list[Instance],
        question_type: BaseQuestionType,
        generation_config: GenerationConfig | None,
    ) -> list[EvalResult]:
        """Evaluate a batch of instances."""
        prompts = [
            question_type.format(inst, include_evidence=self.config.include_evidence)
            for inst in instances
        ]
        
        responses = self._generate_responses(prompts, generation_config)
        
        results = []
        for instance, prompt, response in zip(instances, prompts, responses):
            result = self._evaluate_single(
                instance, prompt, response, question_type
            )
            results.append(result)
        
        return results
    
    def _generate_responses(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None,
    ) -> list[str]:
        """Generate responses for all prompts."""
        config = generation_config or self.llm.generation_config
        
        all_responses = []
        
        # Process in batches
        num_batches = math.ceil(len(prompts) / self.config.batch_size)
        
        iterator = range(0, len(prompts), self.config.batch_size)
        if self.config.show_progress:
            iterator = tqdm(
                iterator,
                desc="Generating",
                total=num_batches,
                unit="batch"
            )
        
        for i in iterator:
            batch_prompts = prompts[i:i + self.config.batch_size]
            
            try:
                batch_responses = self.llm.generate_batch(
                    batch_prompts, config
                )
                all_responses.extend(batch_responses)
                
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                
                if self.config.retry_on_error:
                    # Fall back to individual generation
                    for prompt in batch_prompts:
                        try:
                            response = self.llm.generate(prompt, config)
                            all_responses.append(response)
                        except Exception as e2:
                            logger.error(f"Individual generation failed: {e2}")
                            all_responses.append("")
                else:
                    all_responses.extend([""] * len(batch_prompts))
        
        return all_responses
    
    def _evaluate_single(
        self,
        instance: Instance,
        prompt: str,
        response: str,
        question_type: BaseQuestionType,
    ) -> EvalResult:
        """Evaluate a single instance."""
        error = None
        
        # Extract structured data
        try:
            extracted = question_type.extract(response)
        except Exception as e:
            logger.warning(f"Extraction failed for {instance.id}: {e}")
            extracted = {"_raw_response": response}
            error = f"Extraction error: {e}"
        
        # Compute metrics
        metrics = []
        for metric in question_type.get_metrics():
            try:
                result = metric.compute(extracted, instance)
                metrics.append(result)
            except Exception as e:
                logger.warning(
                    f"Metric {metric.name} failed for {instance.id}: {e}"
                )
                metrics.append(MetricResult(
                    name=metric.name,
                    value=float("nan"),
                    details={"error": str(e)}
                ))
        
        return EvalResult(
            instance_id=instance.id,
            question_type=question_type.name,
            prompt=prompt,
            response=response,
            extracted={k: v for k, v in extracted.items() if k != "_raw_response"},
            metrics=metrics,
            error=error,
        )
    
    def _aggregate_results(
        self,
        results: list[EvalResult],
        question_type: BaseQuestionType,
    ) -> AggregatedResults:
        """Aggregate individual results into summary statistics."""
        if not results:
            return AggregatedResults(
                question_type=question_type.name,
                num_instances=0,
                num_errors=0,
                metrics={},
                per_instance=[] if self.config.save_individual_results else None,
            )
        
        # Collect metric values
        metric_values: dict[str, list[float]] = {}
        num_errors = 0
        
        for result in results:
            if result.error:
                num_errors += 1
            
            for metric_result in result.metrics:
                if metric_result.name not in metric_values:
                    metric_values[metric_result.name] = []
                
                # Skip NaN values for aggregation
                if not math.isnan(metric_result.value):
                    metric_values[metric_result.name].append(metric_result.value)
        
        # Compute statistics
        aggregated_metrics = {}
        for name, values in metric_values.items():
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)
                
                aggregated_metrics[name] = {
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "count": len(values),
                }
            else:
                aggregated_metrics[name] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "count": 0,
                }
        
        return AggregatedResults(
            question_type=question_type.name,
            num_instances=len(results),
            num_errors=num_errors,
            metrics=aggregated_metrics,
            per_instance=results if self.config.save_individual_results else None,
        )
    
    def save_results(
        self,
        results: AggregatedResults,
        output_path: str | Path,
        format: str = "json",
    ):
        """Save evaluation results to file.
        
        Args:
            results: Evaluation results to save.
            output_path: Output file path.
            format: Output format ('json', 'jsonl').
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(results.to_dict(), f, indent=2, ensure_ascii=False)
        
        elif format == "jsonl":
            with open(output_path, "w") as f:
                # Write summary first
                summary = {
                    "type": "summary",
                    "question_type": results.question_type,
                    "num_instances": results.num_instances,
                    "num_errors": results.num_errors,
                    "metrics": results.metrics,
                }
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
                
                # Write individual results
                if results.per_instance:
                    for result in results.per_instance:
                        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Results saved to {output_path}")
    
    @staticmethod
    def load_results(path: str | Path) -> AggregatedResults:
        """Load evaluation results from file.
        
        Args:
            path: Path to results file.
        
        Returns:
            Loaded AggregatedResults.
        """
        path = Path(path)
        
        with open(path, "r") as f:
            if path.suffix == ".jsonl":
                lines = f.readlines()
                summary = json.loads(lines[0])
                per_instance = [json.loads(line) for line in lines[1:]]
                
                # Reconstruct EvalResults
                eval_results = []
                for data in per_instance:
                    metrics = [
                        MetricResult(**m) for m in data.get("metrics", [])
                    ]
                    eval_results.append(EvalResult(
                        instance_id=data["instance_id"],
                        question_type=data["question_type"],
                        prompt=data["prompt"],
                        response=data["response"],
                        extracted=data["extracted"],
                        metrics=metrics,
                        error=data.get("error"),
                    ))
                
                return AggregatedResults(
                    question_type=summary["question_type"],
                    num_instances=summary["num_instances"],
                    num_errors=summary["num_errors"],
                    metrics=summary["metrics"],
                    per_instance=eval_results,
                )
            else:
                data = json.load(f)
                
                per_instance = None
                if data.get("per_instance"):
                    per_instance = []
                    for item in data["per_instance"]:
                        metrics = [
                            MetricResult(**m) for m in item.get("metrics", [])
                        ]
                        per_instance.append(EvalResult(
                            instance_id=item["instance_id"],
                            question_type=item["question_type"],
                            prompt=item["prompt"],
                            response=item["response"],
                            extracted=item["extracted"],
                            metrics=metrics,
                            error=item.get("error"),
                        ))
                
                return AggregatedResults(
                    question_type=data["question_type"],
                    num_instances=data["num_instances"],
                    num_errors=data["num_errors"],
                    metrics=data["metrics"],
                    per_instance=per_instance,
                )
