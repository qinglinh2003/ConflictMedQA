"""
ConflictMedQA Evaluation Module

This module provides a comprehensive framework for evaluating LLM responses
to medical claims with potentially conflicting evidence.

Main components:
- Types: Core data structures (Instance, MetricResult, EvalResult)
- QuestionTypes: Different question formats for evaluation
- Extractors: Parse structured data from LLM responses  
- Metrics: Compute evaluation metrics
- LLM: LLM backends (Transformers, vLLM, API)
- Evaluator: Main evaluation orchestrator

Example usage:
    ```python
    from eval import (
        Instance, Label, DirectQuestion,
        Evaluator, EvaluatorConfig,
        TransformersLLM, GenerationConfig,
    )
    
    # Create LLM backend
    llm = TransformersLLM("meta-llama/Llama-2-7b-chat-hf")
    
    # Create instances
    instances = [
        Instance(
            id="1",
            claim="Aspirin reduces heart attack risk",
            evidence=["Study A shows benefit...", "Study B shows no effect..."],
            label=Label.CONFLICT
        )
    ]
    
    # Run evaluation
    evaluator = Evaluator(llm)
    results = evaluator.evaluate(instances, DirectQuestion())
    
    # Print results
    print(results.metrics)
    ```

CLI usage:
    ```bash
    python -m eval.run_eval data.json --model meta-llama/Llama-2-7b-chat-hf \\
        --backend transformers --question-type direct --output results.json
    ```
"""

from .types import (
    Label,
    ConflictType,
    Evidence,
    ConflictInfo,
    Instance,
    MetricResult,
    EvalResult,
    AggregatedResults,
)

from .extractors import (
    BaseExtractor,
    ChoiceExtractor,
    KeywordExtractor,
    ConflictAwarenessExtractor,
    NumberExtractor,
    ConfidenceExtractor,
    # LLM-based extractors
    LLMExtractorBase,
    LLMChoiceExtractor,
    LLMConflictExtractor,
    LLMConfidenceExtractor,
    LLMStructuredExtractor,
)

from .metrics import (
    BaseMetric,
    ForcedStanceAccuracy,
    ConsensusAccuracy,
    ConflictRecognitionMetric,
    MultiPerspectiveMetric,
    EvidenceUtilizationMetric,
    CalibrationScoreMetric,
    CalibrationGapMetric,
    ExpectedCalibrationError,
    # LLM-as-judge metrics
    LLMJudgeMetric,
    ResponseQualityMetric,
    ReasoningQualityMetric,
    SafetyMetric,
    PairwiseComparisonMetric,
)

from .question_types import (
    BaseQuestionType,
    DirectQuestion,
    ForcedStanceQuestion,
    ConfidenceQuestion,
    ConsensusQuestion,
)

from .llm import (
    BaseLLM,
    TransformersLLM,
    VLLMLlm,
    APILlm,
)

from .llm.base import GenerationConfig

from .evaluator import Evaluator, EvaluatorConfig

from .settings import (
    BaseSetting,
    NoEvidenceSetting,
    WithEvidenceSetting,
    RAGSetting,
)

from .loaders import load_instances

__all__ = [
    # Types
    "Label",
    "Instance",
    "MetricResult",
    "EvalResult",
    "AggregatedResults",
    # Extractors (rule-based)
    "BaseExtractor",
    "ChoiceExtractor",
    "KeywordExtractor",
    "ConflictAwarenessExtractor",
    "NumberExtractor",
    "ConfidenceExtractor",
    # Extractors (LLM-based)
    "LLMExtractorBase",
    "LLMChoiceExtractor",
    "LLMConflictExtractor",
    "LLMConfidenceExtractor",
    "LLMStructuredExtractor",
    # Metrics (rule-based)
    "BaseMetric",
    "ForcedStanceAccuracy",
    "ConsensusAccuracy",
    "ConflictRecognitionMetric",
    "MultiPerspectiveMetric",
    "EvidenceUtilizationMetric",
    "CalibrationScoreMetric",
    "CalibrationGapMetric",
    "ExpectedCalibrationError",
    # Metrics (LLM-as-judge)
    "LLMJudgeMetric",
    "ResponseQualityMetric",
    "ReasoningQualityMetric",
    "SafetyMetric",
    "PairwiseComparisonMetric",
    # Question Types
    "BaseQuestionType",
    "DirectQuestion",
    "ForcedStanceQuestion",
    "ConfidenceQuestion",
    "ConsensusQuestion",
    # LLM Backends
    "BaseLLM",
    "TransformersLLM",
    "VLLMLlm",
    "APILlm",
    "GenerationConfig",
    # Evaluator
    "Evaluator",
    "EvaluatorConfig",
    # Settings (data preparation)
    "BaseSetting",
    "NoEvidenceSetting",
    "WithEvidenceSetting",
    "RAGSetting",
    # Loaders
    "load_instances",
]