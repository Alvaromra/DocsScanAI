import os
import asyncio
import hashlib
import json
import time
from functools import lru_cache

import requests

# Ajuste via env se quiser
# Aceita tanto base host quanto URL completa /api/generate
_ollama_env = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_URL = (
    _ollama_env
    if "/api/generate" in _ollama_env
    else _ollama_env.rstrip("/") + "/api/generate"
)
MODEL_MAIN = os.getenv("OLLAMA_MODEL_ANALYSIS", "llama3.1:8b")
# modelo leve opcional; se não existir, cai no principal para evitar 404
MODEL_LIGHT = os.getenv("OLLAMA_MODEL_LIGHT") or MODEL_MAIN
RAG_TIMEOUT = 1.5
OLLAMA_TIMEOUT = 60

# Sessão global para keep-alive
session = requests.Session()
session.headers.update({"Connection": "keep-alive"})


def hash_text(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8", errors="ignore")).hexdigest()


def preload_model(model: str = MODEL_MAIN):
    """Dispara um /api/generate vazio para carregar pesos em cache."""
    try:
        session.post(
            OLLAMA_URL,
            json={"model": model, "prompt": "", "stream": False},
            timeout=120,
        )
    except Exception as exc:
        print(f"[PRELOAD] Falha ao carregar {model}: {exc}")


def optimize_prompt(query: str, context: str, max_chars: int = 1200) -> str:
    """Deduplica e trunca contexto para reduzir latência."""
    if not context:
        return query.strip()
    ctx = "\n".join(dict.fromkeys(context.splitlines()))  # dedup por linha
    ctx = ctx.replace("\n\n\n", "\n\n")
    ctx = ctx[:max_chars]
    return (
        f"Pergunta: {query.strip()}\n"
        f"Contexto (trechos locais):\n{ctx}\n"
        "Responda de forma curta, clara e precisa."
    )


def select_model(question: str) -> str:
    words = question.strip().split()
    # Se light == main, evita 404 de modelo inexistente
    light = MODEL_LIGHT or MODEL_MAIN
    if len(words) < 8:
        return light
    if any(k in question.lower() for k in ["analisar", "documento", "pdf", "arquivo"]):
        return MODEL_MAIN
    return light


def call_ollama_stream(prompt: str, model: str):
    """Generator que devolve tokens em streaming."""
    start = time.time()
    try:
        with session.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": True},
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                token = data.get("response", "")
                if token:
                    yield token
    except Exception as exc:
        yield f"[ERRO OLLAMA] {exc}"
    finally:
        dur = (time.time() - start) * 1000
        print(f"[OLLAMA] tempo total: {dur:.0f} ms")


async def rag_search_async(rag_fn, query: str, top_k: int = 3, timeout: float = RAG_TIMEOUT):
    """Executa RAG com timeout; retorna (contexto, ms ou None)."""
    start = time.time()
    try:
        res = await asyncio.wait_for(rag_fn(query, top_k=top_k), timeout=timeout)
        dur = (time.time() - start) * 1000
        return res, dur
    except Exception:
        return "", None


@lru_cache(maxsize=256)
def cache_answer(prompt_hash: str):
    return None
