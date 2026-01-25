"""Base class for LLM backends."""

from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Configuration for text generation.
    
    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy, higher = more random).
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling parameter.
        do_sample: Whether to use sampling (False = greedy decoding).
        stop_sequences: List of sequences that stop generation.
        num_return_sequences: Number of sequences to return.
    """
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    do_sample: bool = False
    stop_sequences: list[str] = field(default_factory=list)
    num_return_sequences: int = 1
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "stop_sequences": self.stop_sequences,
            "num_return_sequences": self.num_return_sequences,
        }


class BaseLLM(ABC):
    """Abstract base class for LLM backends.
    
    Provides a unified interface for generating text using different
    LLM backends (transformers, vLLM, API).
    
    Attributes:
        model_name: Name or path of the model.
        generation_config: Default generation configuration.
    """
    
    def __init__(
        self,
        model_name: str,
        generation_config: GenerationConfig | None = None,
        **kwargs
    ):
        """Initialize the LLM backend.
        
        Args:
            model_name: Model name or path.
            generation_config: Default generation config.
            **kwargs: Backend-specific arguments.
        """
        self.model_name = model_name
        self.generation_config = generation_config or GenerationConfig()
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt text.
            generation_config: Override default generation config.
        
        Returns:
            Generated text (excluding the prompt).
        """
        pass
    
    @abstractmethod
    def generate_batch(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts.
            generation_config: Override default generation config.
        
        Returns:
            List of generated texts.
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
