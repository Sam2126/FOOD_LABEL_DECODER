"""
ollama_client.py
=================

This module is the isolated communication layer that talks to Ollama.
Supports dynamic model selection, listing installed models, and customizable
generation parameters (temperature, max_tokens).
"""

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger("food-label-decoder.ollama_client")

# ---------------------------------------------------------------------------
# Configuration (via environment variables)
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "codellama")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "500"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "15m")


class OllamaError(Exception):
    """Raised when Ollama cannot be reached or returns an error."""


def list_models() -> list[str]:
    """
    Queries Ollama's /api/tags endpoint to return all locally installed models.
    Falls back to inspecting local Ollama model manifests if endpoint returns a partial set.
    """
    models = []
    url = f"{OLLAMA_HOST}/api/tags"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m["name"] for m in data.get("models", [])]
    except Exception as exc:
        logger.warning("Could not fetch models from Ollama at %s: %s", url, exc)

    # Also inspect local manifest directory for any installed model tags
    try:
        from pathlib import Path
        manifest_dir = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
        if manifest_dir.exists():
            for model_folder in manifest_dir.iterdir():
                if model_folder.is_dir():
                    for tag_file in model_folder.iterdir():
                        if tag_file.is_file():
                            tag_name = f"{model_folder.name}:{tag_file.name}"
                            if tag_name not in models and model_folder.name not in models:
                                models.append(tag_name)
    except Exception:
        pass

    if models:
        # Sort so llama3.2 / 1b is first, then llama3, then codellama
        return sorted(list(set(models)), key=lambda x: (0 if "1b" in x else (1 if "llama3.2" in x else (2 if "llama3" in x else 3))))

    return [OLLAMA_MODEL]



def ask_ollama(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Sends a prompt to Ollama and returns the generated text.
    Supports overriding model, max_tokens, and temperature.
    """
    effective_model = (model.strip() if model and model.strip() else OLLAMA_MODEL)
    effective_max_tokens = max_tokens if max_tokens is not None else OLLAMA_MAX_TOKENS
    effective_temp = temperature if temperature is not None else OLLAMA_TEMPERATURE

    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": effective_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": effective_max_tokens,
            "temperature": effective_temp,
            "repeat_last_n": 128,
            "repeat_penalty": 1.18,
        },
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }

    logger.info("Calling Ollama at %s with model=%s, max_tokens=%d, temp=%.2f", url, effective_model, effective_max_tokens, effective_temp)

    try:
        response = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {OLLAMA_HOST}. Is Ollama running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise OllamaError(
            f"Ollama did not respond within {OLLAMA_TIMEOUT_SECONDS} seconds."
        ) from exc
    except requests.exceptions.HTTPError as exc:
        raise OllamaError(f"Ollama returned an HTTP error ({response.status_code}): {response.text}") from exc

    data = response.json()

    if "response" not in data:
        raise OllamaError(f"Unexpected response format from Ollama: {data}")

    return data["response"].strip()


def ask_code_llama(prompt: str) -> str:
    """Backwards-compatible wrapper for existing calls."""
    return ask_ollama(prompt)

