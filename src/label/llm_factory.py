#!/usr/bin/env python
"""
LLM Factory for multiple providers.

Supports:
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- vLLM (local or server)
- Ollama (local)
"""
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM."""
    model: str
    temperature: float = 0.0
    max_tokens: int = 16
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    tensor_parallel_size: int = 1  # For vllm_offline


@dataclass
class LLMResponse:
    """Standardized LLM response."""
    text: str
    model: str
    usage: Optional[dict] = None
    latency_ms: Optional[float] = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """Generate a response."""
        pass


# ==================== Implementations ====================

class OpenAILLM(BaseLLM):
    """OpenAI API."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = OpenAI(api_key=api_key, base_url=config.api_base)
    
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            latency_ms=latency,
        )


class AnthropicLLM(BaseLLM):
    """Anthropic API."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = anthropic.Anthropic(api_key=api_key, base_url=config.api_base)
    
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        start = time.time()
        
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        
        response = self.client.messages.create(**kwargs)
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            text=response.content[0].text.strip(),
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            },
            latency_ms=latency,
        )


class VLLMClient(BaseLLM):
    """vLLM OpenAI-compatible server."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        
        api_base = config.api_base or "http://localhost:8000/v1"
        self.client = OpenAI(api_key="EMPTY", base_url=api_base)
    
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            text=response.choices[0].message.content.strip(),
            model=self.config.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
            latency_ms=latency,
        )


class OllamaLLM(BaseLLM):
    """Ollama local inference."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import ollama
        except ImportError:
            raise ImportError("pip install ollama")
        
        self._ollama = ollama
    
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        start = time.time()
        response = self._ollama.chat(
            model=self.config.model,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )
        latency = (time.time() - start) * 1000
        
        return LLMResponse(
            text=response["message"]["content"].strip(),
            model=self.config.model,
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
            },
            latency_ms=latency,
        )


class VLLMOffline(BaseLLM):
    """vLLM offline batch inference (no server needed)."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._llm = None
        self._sampling_params = None
    
    def _ensure_loaded(self):
        """Lazy load model."""
        if self._llm is not None:
            return
        
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError("pip install vllm")
        
        tp = self.config.tensor_parallel_size
        logger.info(f"Loading {self.config.model} with tensor_parallel={tp}...")
        
        self._llm = LLM(
            model=self.config.model,
            dtype="float16",
            tensor_parallel_size=tp,
            trust_remote_code=True,
        )
        
        self._sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        
        logger.info(f"Model loaded.")
    
    def generate(self, prompt: str, system: str = "") -> LLMResponse:
        """Generate single response."""
        self._ensure_loaded()
        
        # Format as chat
        if system:
            full_prompt = f"<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        else:
            full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        
        start = time.time()
        outputs = self._llm.generate([full_prompt], self._sampling_params)
        latency = (time.time() - start) * 1000
        
        output = outputs[0]
        text = output.outputs[0].text.strip()
        
        return LLMResponse(
            text=text,
            model=self.config.model,
            usage={
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
            },
            latency_ms=latency,
        )
    
    def generate_batch(self, prompts: list[str], system: str = "") -> list[LLMResponse]:
        """Batch generation (much faster)."""
        self._ensure_loaded()
        
        # Format all prompts
        full_prompts = []
        for prompt in prompts:
            if system:
                full_prompts.append(f"<|system|>\n{system}<|end|>\n<|user|>\n{prompt}<|end|>\n<|assistant|>\n")
            else:
                full_prompts.append(f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n")
        
        start = time.time()
        outputs = self._llm.generate(full_prompts, self._sampling_params)
        total_latency = (time.time() - start) * 1000
        avg_latency = total_latency / len(prompts)
        
        responses = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            responses.append(LLMResponse(
                text=text,
                model=self.config.model,
                usage={
                    "prompt_tokens": len(output.prompt_token_ids),
                    "completion_tokens": len(output.outputs[0].token_ids),
                },
                latency_ms=avg_latency,
            ))
        
        return responses


# ==================== Factory ====================

LLM_REGISTRY: dict[str, type[BaseLLM]] = {
    "openai": OpenAILLM,
    "anthropic": AnthropicLLM,
    "vllm": VLLMClient,
    "ollama": OllamaLLM,
    "vllm_offline": VLLMOffline,
}


class LLMFactory:
    """Factory for creating LLM instances."""
    
    @staticmethod
    def create(
        provider: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 16,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        tensor_parallel_size: int = 1,
    ) -> BaseLLM:
        """
        Create an LLM instance.
        
        Args:
            provider: "openai", "anthropic", "vllm", "ollama", "vllm_offline"
            model: Model name
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            api_key: API key (optional, uses env var if not set)
            api_base: Base URL (for vLLM or custom endpoints)
            tensor_parallel_size: Number of GPUs for vllm_offline
        
        Examples:
            llm = LLMFactory.create("openai", "gpt-4o")
            llm = LLMFactory.create("anthropic", "claude-sonnet-4-20250514")
            llm = LLMFactory.create("vllm", "llama-3.1-8b", api_base="http://localhost:8000/v1")
            llm = LLMFactory.create("vllm_offline", "Qwen/Qwen2.5-32B-Instruct", tensor_parallel_size=2)
        """
        if provider not in LLM_REGISTRY:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(LLM_REGISTRY.keys())}")
        
        config = LLMConfig(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            api_base=api_base,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        return LLM_REGISTRY[provider](config)
    
    @staticmethod
    def list_providers() -> list[str]:
        """List available providers."""
        return list(LLM_REGISTRY.keys())


def register_provider(name: str):
    """Decorator to register a custom provider."""
    def decorator(cls: type[BaseLLM]):
        LLM_REGISTRY[name] = cls
        return cls
    return decorator