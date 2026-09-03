"""
LLM Service
Unified interface for multiple LLM providers
"""
import logging
import os
from typing import Optional, List, Dict
import httpx

logger = logging.getLogger(__name__)


class LLMService:
    """Unified LLM service supporting multiple providers."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai")
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
                groq_key = os.getenv("GROQ_API_KEY", "")
                self._client = {"api_key": groq_key, "base_url": "https://api.groq.com/openai/v1"}
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
            model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

            if self.provider == "openai":
                return self._generate_openai(prompt, system_prompt, max_tokens, temperature, model)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt, system_prompt, max_tokens, temperature, model)
            elif self.provider == "groq":
                return self._generate_openai_compat(prompt, system_prompt, max_tokens, temperature, model)
            else:
                return self._generate_fallback(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback(prompt)

    def _generate_openai(self, prompt, system_prompt, max_tokens, temperature, model):
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return response.choices[0].message.content

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

    def _generate_openai_compat(self, prompt, system_prompt, max_tokens, temperature, model):
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = httpx.post(
            f"{client['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {client['api_key']}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            timeout=60,
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _generate_fallback(self, prompt: str) -> str:
        return "I'm unable to process that query right now. Please try again or rephrase your question."


llm_service = LLMService()
