"""
Provider-agnostic LLM client.

Design goal (Section 12 risk: "Agent LLM calls are slow or flaky during a live demo"):
the rest of the codebase calls `call_llm(prompt)` and never knows or cares which
provider answered, or whether one answered at all. Priority order:

  1. Groq          — free tier, fast, OpenAI-compatible API. Set GROQ_API_KEY.
  2. Ollama        — fully local, zero cost, zero API key. Just run `ollama serve`.
  3. Template mode — no network, no key, no server: deterministic canned phrasing
                      so agents still produce usable output. This is what keeps a
                      demo alive if wifi dies or a free-tier quota is hit.

Nothing in the agent logic ever depends on which of these actually ran.
"""
from __future__ import annotations
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env in the working directory, if present; no-op otherwise

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def _call_groq(prompt: str) -> str | None:
    if not GROQ_API_KEY:
        return None
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3,
            },
            timeout=8,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _call_ollama(prompt: str) -> str | None:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception:
        return None


def call_llm(prompt: str, fallback: str) -> str:
    """
    Try Groq, then Ollama, then fall back to a caller-supplied deterministic string.
    `fallback` should already contain the real numbers so the demo output is still
    correct even if no LLM answers — only the *phrasing* degrades, never the facts.
    """
    for fn in (_call_groq, _call_ollama):
        result = fn(prompt)
        if result:
            return result
    return fallback
