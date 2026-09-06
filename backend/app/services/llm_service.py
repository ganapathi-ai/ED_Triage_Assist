"""
LLM Service
Multi-provider cascade: tries providers in priority order, falls back automatically.
Priority: OpenRouter → Groq → Fallback message
This ensures AI responses are always returned even if one API is down/rate-limited.
"""
import logging
import os
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


class LLMProvider:
    """Base class for a single LLM provider."""

    def generate(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float, model: str) -> str:
        raise NotImplementedError


class OpenRouterProvider(LLMProvider):
    """OpenRouter — free tier with multiple models (best quality)."""

    MODELS = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "google/gemma-4-31b-it:free",
        "minimax/minimax-m2.7:free",
    ]

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            import openai
            api_key = os.getenv("OPENROUTER_API_KEY", "")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "https://ed-triage-assist-api.onrender.com",
                    "X-Title": "ED Triage Assist",
                },
            )
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float, model: str) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try each model in order until one works
        target_model = model or self.MODELS[0]
        # If the configured model isn't a known free model, prepend it
        model_list = [target_model] + [m for m in self.MODELS if m != target_model]

        last_error = None
        for m in model_list:
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                content = response.choices[0].message.content
                if content:
                    logger.info(f"OpenRouter OK [{m}]")
                    return content
                else:
                    logger.warning(f"OpenRouter [{m}] returned None content, trying next model")
            except Exception as e:
                last_error = e
                logger.warning(f"OpenRouter [{m}] failed: {e}")
                continue

        raise RuntimeError(f"All OpenRouter models failed. Last error: {last_error}")


class GroqProvider(LLMProvider):
    """Groq — fast inference, free tier fallback."""

    MODELS = [
        "groq/compound-mini",
        "groq/compound",
        "qwen/qwen3.8-27b",
    ]

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY", "")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY not set")
            self._client = Groq(api_key=api_key)
        return self._client

    def generate(self, prompt: str, system_prompt: Optional[str], max_tokens: int, temperature: float, model: str) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        model_list = self.MODELS

        last_error = None
        for m in model_list:
            # Try with normal tokens first, then reduced if 413
            for tokens in [min(max_tokens, 1000), 400]:
                try:
                    response = client.chat.completions.create(
                        model=m,
                        messages=messages,
                        max_tokens=tokens,
                        temperature=temperature,
                    )
                    content = response.choices[0].message.content
                    if content:
                        logger.info(f"Groq OK [{m}] (max_tokens={tokens})")
                        return content
                except Exception as e:
                    err_str = str(e)
                    last_error = e
                    if "413" in err_str or "too_large" in err_str or "request_too_large" in err_str:
                        logger.warning(f"Groq [{m}] 413 too large, retrying with {tokens//2} tokens")
                        continue  # Retry inner loop with fewer tokens
                    else:
                        logger.warning(f"Groq [{m}] failed: {e}")
                        break  # Move to next model

        raise RuntimeError(f"All Groq models failed. Last error: {last_error}")


class LLMService:
    """
    Multi-provider LLM service with automatic cascade fallback.
    Tries providers in order: OpenRouter → Groq → Informative fallback.
    """

    def __init__(self):
        self._providers: List[Tuple[str, LLMProvider]] = [
            ("openrouter", OpenRouterProvider()),
            ("groq", GroqProvider()),
        ]

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        model: str = None,
    ) -> str:
        """
        Generate a response, cascading through providers until one succeeds.
        """
        model = model or os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
        errors = []

        for provider_name, provider in self._providers:
            try:
                result = provider.generate(prompt, system_prompt, max_tokens, temperature, model)
                if result:
                    return result
            except Exception as e:
                logger.warning(f"Provider '{provider_name}' failed: {e}")
                errors.append(f"{provider_name}: {e}")
                continue

        # All providers failed — return informative fallback
        logger.error(f"All LLM providers failed: {errors}")
        return self._generate_fallback(prompt, errors)

    def _generate_fallback(self, prompt: str, errors: List[str] = None) -> str:
        return (
            "⚠️ AI response temporarily unavailable (all LLM providers are busy or rate-limited). "
            "The Triage predictions (ESI level, deterioration risk, wait time) are still fully functional. "
            "Please try the AI Chat again in a moment."
        )


llm_service = LLMService()
