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

load_dotenv(override=True)  # reads .env in the working directory, if present; no-op otherwise

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")


def _call_groq(prompt: str) -> str | None:
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", ".env")
    load_dotenv(env_path, override=True)
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    
    if not api_key:
        print(f"\033[91m[AGENT TRACE] No GROQ_API_KEY found at {env_path}\033[0m")
        return None
    try:
        from langchain_groq import ChatGroq
        chat = ChatGroq(temperature=0.3, groq_api_key=api_key, model_name=model, max_tokens=300)
        # We can just call it synchronously since this is running inside resilient_call threadpool anyway
        resp = chat.invoke(prompt)
        return resp.content.strip()
    except Exception as e:
        print(f"\033[91m[AGENT TRACE] Groq Error: {e}\033[0m")
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
    print(f"\n\033[96m[AGENT TRACE] Sending prompt to LLM:\033[0m\n{prompt}\n")
    
    result = _call_groq(prompt)
    if result:
        print(f"\033[92m[AGENT TRACE] Successfully used: Groq\033[0m\n\033[93m[AGENT TRACE] Response:\033[0m\n{result}\n")
        return result
        
    result = _call_ollama(prompt)
    if result:
        print(f"\033[92m[AGENT TRACE] Successfully used: Ollama\033[0m\n\033[93m[AGENT TRACE] Response:\033[0m\n{result}\n")
        return result
        
    print(f"\033[91m[AGENT TRACE] Used: TEMPLATE FALLBACK (Groq/Ollama failed or missing keys)\033[0m\n\033[93m[AGENT TRACE] Response:\033[0m\n{fallback}\n")
    return fallback
