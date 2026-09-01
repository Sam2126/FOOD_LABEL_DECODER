"""
LLM Service
===========

Responsibility: the ONLY service that talks to Ollama/Code Llama.
Takes a fully-formed prompt (built by whoever calls it - the Orchestrator)
and returns the generated text. Knows nothing about ingredients, RAG, or
knowledge bases - pure "given a prompt, generate a response" service.

Endpoints:
    GET  /health
    GET  /models             list all models available in Ollama
    POST /generate   {\"prompt\": \"...\", \"model\": \"...\", \"max_tokens\": N, \"temperature\": F}
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ollama_client import ask_ollama, list_models, OllamaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("llm_service")

app = FastAPI(title="Food Label Decoder - LLM Service", version="0.5.0")


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None          # override the default Ollama model
    max_tokens: Optional[int] = None     # override num_predict
    temperature: Optional[float] = None  # override temperature


class GenerateResponse(BaseModel):
    response: str
    model_used: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "service": "llm_service"}


@app.get("/models")
def models():
    """List all model names currently available in Ollama."""
    return {"models": list_models()}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    logger.info(
        "Generating response  model=%s  max_tokens=%s  prompt_len=%d",
        request.model or "default",
        request.max_tokens or "default",
        len(request.prompt),
    )
    try:
        text = ask_ollama(
            request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except OllamaError as exc:
        logger.error("Ollama call failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return GenerateResponse(response=text, model_used=request.model or "")
