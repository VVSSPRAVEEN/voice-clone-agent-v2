"""LLM worker calling a local vLLM OpenAI-compatible server.

The vLLM server runs as a separate process (e.g. `vllm serve` on
http://127.0.0.1:8001). This worker only talks HTTP — it never imports
vLLM. Falls back to rule-based bilingual responses when the server is
unreachable or disabled.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import httpx
from loguru import logger

from .config import SETTINGS


@dataclass
class LLMResult:
    text: str
    latency_ms: float
    source: str  # "vllm" | "fallback"


# Bilingual canned responses used when no LLM is configured.
_TELUGU_ACKS = [
    "అవును, విన్నాను. ఇంకా చెప్పండి.",
    "సరే, కొనసాగించండి.",
    "అర్థమైంది. మీరు ఏమి అనుకుంటున్నారు?",
    "బాగుంది. ఇంకేమైనా చెప్పాలా?",
]
_ENGLISH_ACKS = [
    "Yes, I heard you. Please continue.",
    "Understood. What would you like next?",
    "Okay, go on.",
    "Got it. Anything else?",
]


class LLMWorker:
    def __init__(self):
        self.enabled = SETTINGS.llm_enabled
        self.model_name = SETTINGS.llm_model
        self.system_prompt = SETTINGS.llm_system_prompt
        self.max_tokens = SETTINGS.llm_max_tokens
        self.temperature = SETTINGS.llm_temperature
        self.api_url = SETTINGS.llm_api_url
        self.timeout = 30.0

    async def generate(self, user_text: str, language: str) -> LLMResult:
        if not self.enabled:
            return self._fallback(user_text, language)
        try:
            return await self._call_server(user_text)
        except Exception as e:
            logger.warning(f"vLLM server call failed ({e}); falling back")
            return self._fallback(user_text, language)

    async def _call_server(self, user_text: str) -> LLMResult:
        t0 = time.perf_counter()
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.api_url}/v1/chat/completions", json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"vLLM server returned {resp.status_code}")
            data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        elapsed = (time.perf_counter() - t0) * 1000
        return LLMResult(text=text, latency_ms=elapsed, source="vllm")

    def _fallback(self, user_text: str, language: str) -> LLMResult:
        """Rule-based response when no LLM is configured."""
        lang = (language or "").lower()
        if lang.startswith("te"):
            reply = random.choice(_TELUGU_ACKS)
        elif lang.startswith("en"):
            reply = random.choice(_ENGLISH_ACKS)
        else:
            has_telugu = any("\u0c00" <= ch <= "\u0c7f" for ch in user_text)
            reply = random.choice(_TELUGU_ACKS if has_telugu else _ENGLISH_ACKS)
        return LLMResult(text=reply, latency_ms=50.0, source="fallback")

    async def close(self) -> None:
        # Nothing to clean up: the vLLM server is a separate process.
        logger.debug("LLMWorker closed (remote vLLM server untouched)")