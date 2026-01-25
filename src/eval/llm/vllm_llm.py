"""vLLM backend for high-throughput inference."""

from typing import Any
import logging

from .base import BaseLLM, GenerationConfig

logger = logging.getLogger(__name__)


class VLLMLlm(BaseLLM):
    """LLM backend using vLLM for high-throughput inference.
    
    vLLM provides significantly faster inference through:
    - PagedAttention for efficient KV cache management
    - Continuous batching
    - Optimized CUDA kernels
    
    Attributes:
        model_name: HuggingFace model name or local path.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory to use.
    """
    
    def __init__(
        self,
        model_name: str,
        generation_config: GenerationConfig | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        max_model_len: int | None = None,
        quantization: str | None = None,
        **kwargs
    ):
        """Initialize the vLLM backend.
        
        Args:
            model_name: HuggingFace model name or path.
            generation_config: Generation configuration.
            tensor_parallel_size: Number of GPUs for tensor parallelism.
            gpu_memory_utilization: Fraction of GPU memory to use (0-1).
            dtype: Model dtype ('auto', 'half', 'float16', 'bfloat16', 'float32').
            trust_remote_code: Allow custom model code.
            max_model_len: Maximum sequence length.
            quantization: Quantization method ('awq', 'gptq', 'squeezellm', etc.).
            **kwargs: Additional vLLM LLM arguments.
        """
        super().__init__(model_name, generation_config)
        
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.max_model_len = max_model_len
        self.quantization = quantization
        self.extra_kwargs = kwargs
        
        self._llm = None
    
    def _load_model(self):
        """Lazy load the vLLM engine."""
        if self._llm is not None:
            return
        
        try:
            from vllm import LLM
        except ImportError:
            raise ImportError(
                "vllm is required for VLLMLlm. "
                "Install with: pip install vllm"
            )
        
        logger.info(f"Loading vLLM model: {self.model_name}")
        
        llm_kwargs = {
            "model": self.model_name,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        
        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len
        if self.quantization:
            llm_kwargs["quantization"] = self.quantization
        
        llm_kwargs.update(self.extra_kwargs)
        
        self._llm = LLM(**llm_kwargs)
        
        logger.info("vLLM model loaded successfully")
    
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        """Generate text from a single prompt.
        
        Args:
            prompt: Input prompt.
            generation_config: Override generation config.
        
        Returns:
            Generated text.
        """
        results = self.generate_batch([prompt], generation_config)
        return results[0]
    
    def generate_batch(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate text for multiple prompts.
        
        vLLM excels at batch generation with continuous batching.
        
        Args:
            prompts: List of prompts.
            generation_config: Override generation config.
        
        Returns:
            List of generated texts.
        """
        self._load_model()
        
        from vllm import SamplingParams
        
        config = generation_config or self.generation_config
        
        # Build sampling params
        sampling_kwargs = {
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature if config.do_sample else 0.0,
            "top_p": config.top_p,
            "top_k": config.top_k if config.top_k > 0 else -1,
            "n": config.num_return_sequences,
        }
        
        if config.stop_sequences:
            sampling_kwargs["stop"] = config.stop_sequences
        
        sampling_params = SamplingParams(**sampling_kwargs)
        
        # Generate
        outputs = self._llm.generate(prompts, sampling_params)
        
        # Extract generated texts
        generated_texts = []
        for output in outputs:
            # Take the first completion (n=1 typically)
            text = output.outputs[0].text
            generated_texts.append(text)
        
        return generated_texts
    
    def unload(self):
        """Unload model to free memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
