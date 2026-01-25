"""Transformers (HuggingFace) LLM backend."""

from typing import Any
import logging

from .base import BaseLLM, GenerationConfig

logger = logging.getLogger(__name__)


class TransformersLLM(BaseLLM):
    """LLM backend using HuggingFace Transformers.
    
    Supports any model compatible with AutoModelForCausalLM.
    
    Attributes:
        model_name: HuggingFace model name or local path.
        device: Device to run on ('cuda', 'cpu', 'auto').
        torch_dtype: Data type for model weights.
        trust_remote_code: Whether to trust remote code.
    """
    
    def __init__(
        self,
        model_name: str,
        generation_config: GenerationConfig | None = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: str | None = None,
        **kwargs
    ):
        """Initialize the Transformers backend.
        
        Args:
            model_name: HuggingFace model name or path.
            generation_config: Generation configuration.
            device: Device placement ('cuda', 'cpu', 'auto').
            torch_dtype: Model dtype ('auto', 'float16', 'bfloat16', 'float32').
            trust_remote_code: Allow custom model code.
            load_in_8bit: Use 8-bit quantization.
            load_in_4bit: Use 4-bit quantization.
            attn_implementation: Attention implementation ('flash_attention_2', etc.).
            **kwargs: Additional model loading arguments.
        """
        super().__init__(model_name, generation_config)
        
        self.device = device
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.attn_implementation = attn_implementation
        self.extra_kwargs = kwargs
        
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._model is not None:
            return
        
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch are required for TransformersLLM. "
                "Install with: pip install transformers torch"
            )
        
        logger.info(f"Loading model: {self.model_name}")
        
        # Determine dtype
        if self.torch_dtype == "auto":
            dtype = "auto"
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif self.torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = "auto"
        
        # Model loading kwargs
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": dtype,
            "device_map": self.device,
        }
        
        if self.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        
        model_kwargs.update(self.extra_kwargs)
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        
        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        logger.info(f"Model loaded on device: {self._model.device}")
    
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
        
        Args:
            prompts: List of prompts.
            generation_config: Override generation config.
        
        Returns:
            List of generated texts.
        """
        self._load_model()
        
        import torch
        
        config = generation_config or self.generation_config
        
        # Tokenize
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._model.device)
        
        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": config.max_new_tokens,
            "temperature": config.temperature if config.do_sample else None,
            "top_p": config.top_p if config.do_sample else None,
            "top_k": config.top_k if config.do_sample else None,
            "do_sample": config.do_sample,
            "num_return_sequences": config.num_return_sequences,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        
        # Handle stop sequences
        if config.stop_sequences:
            from transformers import StoppingCriteria, StoppingCriteriaList
            
            class StopOnTokens(StoppingCriteria):
                def __init__(self, stop_ids):
                    self.stop_ids = stop_ids
                
                def __call__(self, input_ids, scores, **kwargs):
                    for stop_id in self.stop_ids:
                        if input_ids[0][-1] == stop_id:
                            return True
                    return False
            
            stop_ids = []
            for seq in config.stop_sequences:
                ids = self._tokenizer.encode(seq, add_special_tokens=False)
                if ids:
                    stop_ids.append(ids[-1])
            
            if stop_ids:
                gen_kwargs["stopping_criteria"] = StoppingCriteriaList([
                    StopOnTokens(stop_ids)
                ])
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(**inputs, **gen_kwargs)
        
        # Decode, removing the input prompt
        input_length = inputs["input_ids"].shape[1]
        generated_texts = []
        
        for output in outputs:
            generated_tokens = output[input_length:]
            text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(text)
        
        return generated_texts
    
    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
