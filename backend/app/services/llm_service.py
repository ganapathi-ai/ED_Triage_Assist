"""
LLM Service
Unified interface for multiple LLM providers including OpenRouter.
"""
import logging
import os
from typing import Optional, List, Dict
import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """Unified LLM service supporting OpenRouter, Groq, OpenAI, Anthropic."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openrouter")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if self.provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY", "")
                self._client = openai.OpenAI(api_key=api_key)
            elif self.provider == "anthropic":
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
                self._client = anthropic.Anthropic(api_key=api_key)
            elif self.provider == "groq":
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY", "")
                self._client = Groq(api_key=api_key)
            elif self.provider == "openrouter":
                # OpenRouter uses OpenAI-compatible API
                import openai
                api_key = os.getenv("OPENROUTER_API_KEY", "")
                self._client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://ed-triage-assist-api.onrender.com",
                        "X-Title": "ED Triage Assist",
                    }
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        model: str = None,
    ) -> str:
        try:
            model = model or os.getenv("LLM_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

            if self.provider == "openai":
                return self._generate_openai_compat(prompt, system_prompt, max_tokens, temperature, model)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, system_prompt, max_tokens, temperature, model)
            elif self.provider == "groq":
                return self._generate_groq(prompt, system_prompt, max_tokens, temperature, model)
            elif self.provider == "openrouter":
                return self._generate_openai_compat(prompt, system_prompt, max_tokens, temperature, model)
            else:
                return self._generate_fallback(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed ({self.provider}): {e}")
            return self._generate_fallback(prompt)

    def _generate_openai_compat(self, prompt, system_prompt, max_tokens, temperature, model):
        """Generate using OpenAI-compatible API (works for OpenAI and OpenRouter)."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        # Some OpenRouter models return None content (streaming only) — handle gracefully
        return content or self._generate_fallback(prompt)

    def _generate_anthropic(self, prompt, system_prompt, max_tokens, temperature, model):
        client = self._get_client()
        kwargs = {
            "model": model or "claude-sonnet-4-20250514",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _generate_groq(self, prompt, system_prompt, max_tokens, temperature, model):
        """Generate using the Groq SDK."""
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _generate_fallback(self, prompt: str) -> str:
        return (
            "I'm currently unable to generate an AI response. "
            "The triage prediction features (ESI level, deterioration risk, wait time) still work. "
            "Please check the AI Chat configuration or try again shortly."
        )


llm_service = LLMService()
