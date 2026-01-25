"""API-based LLM backends (OpenAI, Anthropic, OpenAI-compatible)."""

from typing import Any, Literal
import logging
import time
import os

from .base import BaseLLM, GenerationConfig

logger = logging.getLogger(__name__)


class APILlm(BaseLLM):
    """LLM backend using API providers.
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5, etc.)
    - Anthropic (Claude)
    - OpenAI-compatible APIs (vLLM server, LocalAI, Ollama, etc.)
    
    Attributes:
        model_name: Model identifier for the API.
        provider: API provider ('openai', 'anthropic', 'openai_compatible').
        api_key: API key (or set via environment variable).
        base_url: Base URL for API (for compatible APIs).
    """
    
    def __init__(
        self,
        model_name: str,
        generation_config: GenerationConfig | None = None,
        provider: Literal["openai", "anthropic", "openai_compatible"] = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 60.0,
        **kwargs
    ):
        """Initialize the API backend.
        
        Args:
            model_name: Model name for API calls.
            generation_config: Generation configuration.
            provider: API provider type.
            api_key: API key. Falls back to environment variables:
                - OPENAI_API_KEY for OpenAI
                - ANTHROPIC_API_KEY for Anthropic
            base_url: Base URL for OpenAI-compatible APIs.
            max_retries: Maximum retry attempts on failure.
            retry_delay: Delay between retries (seconds).
            timeout: Request timeout (seconds).
            **kwargs: Additional API-specific arguments.
        """
        super().__init__(model_name, generation_config)
        
        self.provider = provider
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.extra_kwargs = kwargs
        
        # Get API key
        if api_key:
            self.api_key = api_key
        elif provider == "openai" or provider == "openai_compatible":
            self.api_key = os.environ.get("OPENAI_API_KEY")
        elif provider == "anthropic":
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            self.api_key = None
        
        self._client = None
    
    def _get_client(self):
        """Get or create the API client."""
        if self._client is not None:
            return self._client
        
        if self.provider in ("openai", "openai_compatible"):
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai is required for OpenAI API. "
                    "Install with: pip install openai"
                )
            
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.timeout:
                client_kwargs["timeout"] = self.timeout
            
            self._client = OpenAI(**client_kwargs)
            
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic is required for Anthropic API. "
                    "Install with: pip install anthropic"
                )
            
            client_kwargs = {"api_key": self.api_key}
            if self.timeout:
                client_kwargs["timeout"] = self.timeout
            
            self._client = Anthropic(**client_kwargs)
        
        return self._client
    
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
        config = generation_config or self.generation_config
        
        for attempt in range(self.max_retries):
            try:
                if self.provider in ("openai", "openai_compatible"):
                    return self._generate_openai(prompt, config)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(prompt, config)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                    
            except Exception as e:
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
    
    def _generate_openai(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using OpenAI API."""
        client = self._get_client()
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build kwargs
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "n": config.num_return_sequences,
        }
        
        if config.stop_sequences:
            kwargs["stop"] = config.stop_sequences
        
        kwargs.update(self.extra_kwargs)
        
        response = client.chat.completions.create(**kwargs)
        
        return response.choices[0].message.content
    
    def _generate_anthropic(self, prompt: str, config: GenerationConfig) -> str:
        """Generate using Anthropic API."""
        client = self._get_client()
        
        # Build messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build kwargs
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }
        
        if config.stop_sequences:
            kwargs["stop_sequences"] = config.stop_sequences
        
        kwargs.update(self.extra_kwargs)
        
        response = client.messages.create(**kwargs)
        
        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
        
        return "".join(text_parts)
    
    def generate_batch(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate text for multiple prompts.
        
        Note: This is a simple sequential implementation.
        For high throughput, consider using async or batch APIs.
        
        Args:
            prompts: List of prompts.
            generation_config: Override generation config.
        
        Returns:
            List of generated texts.
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, generation_config)
            results.append(result)
        return results
    
    async def generate_async(
        self,
        prompt: str,
        generation_config: GenerationConfig | None = None,
    ) -> str:
        """Async version of generate.
        
        Args:
            prompt: Input prompt.
            generation_config: Override generation config.
        
        Returns:
            Generated text.
        """
        config = generation_config or self.generation_config
        
        if self.provider in ("openai", "openai_compatible"):
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("openai is required for async generation")
            
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            
            async with AsyncOpenAI(**client_kwargs) as client:
                messages = [{"role": "user", "content": prompt}]
                
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                }
                
                if config.stop_sequences:
                    kwargs["stop"] = config.stop_sequences
                
                response = await client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
        
        elif self.provider == "anthropic":
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("anthropic is required for async generation")
            
            async with AsyncAnthropic(api_key=self.api_key) as client:
                messages = [{"role": "user", "content": prompt}]
                
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": config.max_new_tokens,
                    "temperature": config.temperature,
                }
                
                if config.stop_sequences:
                    kwargs["stop_sequences"] = config.stop_sequences
                
                response = await client.messages.create(**kwargs)
                return response.content[0].text
    
    async def generate_batch_async(
        self,
        prompts: list[str],
        generation_config: GenerationConfig | None = None,
        max_concurrent: int = 10,
    ) -> list[str]:
        """Async batch generation with concurrency control.
        
        Args:
            prompts: List of prompts.
            generation_config: Override generation config.
            max_concurrent: Maximum concurrent requests.
        
        Returns:
            List of generated texts.
        """
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_generate(prompt: str) -> str:
            async with semaphore:
                return await self.generate_async(prompt, generation_config)
        
        tasks = [bounded_generate(p) for p in prompts]
        results = await asyncio.gather(*tasks)
        
        return list(results)
