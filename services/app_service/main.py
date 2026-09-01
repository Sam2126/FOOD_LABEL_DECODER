"""
Application Service (Orchestrator)
===================================

Responsibility: the ONLY service the user's browser talks to. Serves the
frontend, exposes the public API, and ORCHESTRATES calls to the RAG
Service and LLM Service to fulfil a request. Contains no LLM logic and no
retrieval logic itself - it delegates and assembles.

    User
     -> Application Service
     -> [calls] RAG Service   -> retrieved context
     -> [calls] LLM Service   -> generated response
     -> back to User

Endpoints:
    GET  /                    - frontend
    GET  /api/health           - liveness + downstream service health
    GET  /api/models           - list available Ollama models (proxy)
    GET  /api/chunks           - proxy to RAG /debug/chunks
    GET  /api/vectors          - proxy to RAG /debug/vectors
    POST /api/query-embed      - proxy to RAG /debug/query-embed
    POST /api/decode           - RAG-powered decode (Exercise 3 + 4 behavior)
    POST /api/decode_no_rag    - baseline decode, no retrieval (Exercise 1 behavior)
"""

import logging
import os
from time import perf_counter
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("app_service")

app = FastAPI(title="Food Label Decoder - Application Service (Orchestrator)", version="0.5.0")

# ---------------------------------------------------------------------------
# Configuration: where are the other services?
# In Docker Compose (Exercise 5) these hostnames become container names.
# ---------------------------------------------------------------------------
RAG_SERVICE_URL = os.getenv("RAG_SERVICE_URL", "http://localhost:8001")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8002")
SERVICE_TIMEOUT_SECONDS = int(os.getenv("SERVICE_TIMEOUT_SECONDS", "120"))


class DecodeRequest(BaseModel):
    ingredient_list: str = Field(min_length=1, max_length=1000)
    model: Optional[str] = None          # e.g. "llama3", "mistral"
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class DecodeResponse(BaseModel):
    ingredient_list: str
    explanation: str
    mode: str
    processing_time_ms: int
    retrieved_chunks: list = Field(default_factory=list)
    unmatched_ingredients: list[str] = Field(default_factory=list)
    context_sent_to_llm: str = ""
    model_used: str = ""


def build_no_rag_prompt(ingredient_list: str) -> str:
    return f"""You are a food label assistant that explains ingredient lists to everyday consumers.

Given the following ingredient list, explain in plain language what each ingredient/additive
is, its likely function, common allergens, and cautious general health considerations (no
diagnoses or definitive disease claims). Only explain ingredients that are explicitly listed
below. Do not add ingredients, do not repeat yourself.

Ingredient list:
{ingredient_list}
"""


def build_rag_prompt(ingredient_list: str, context: str, unmatched_ingredients: list[str]) -> str:
    return f"""SYSTEM INSTRUCTIONS:
You are a food label assistant. Base your answer primarily on the RETRIEVED CONTEXT below,
from a curated food-ingredient knowledge base.
- Use the retrieved context as your main source of facts.
- Explain only ingredients that are literally present in the INGREDIENT LIST. Never infer a
  recipe or product formula, and never add ingredients such as salt, food coloring, or flavoring
  unless they appear in the user's list.
- Do not repeat the ingredient list. Give each listed ingredient at most one short entry.
- Do NOT repeat any sentence or phrase. Write each point exactly once.
- For an item in UNMATCHED INGREDIENTS, say only: "The local knowledge base does not yet
  contain a record for this ingredient." Do not guess its function, allergen status, or health effect.
- Mention any allergens noted in the context.
- Only give health considerations supported by the context, phrased cautiously.
- Use this exact compact structure: "Ingredient - what it is; role; allergen note if present."

RETRIEVED CONTEXT:
{context}

UNMATCHED INGREDIENTS:
{", ".join(unmatched_ingredients) if unmatched_ingredients else "None"}

USER QUESTION:
Explain the following ingredient list to a consumer, using the retrieved context above:
{ingredient_list}
"""


def call_rag_service(ingredient_list: str) -> dict:
    try:
        resp = requests.post(
            f"{RAG_SERVICE_URL}/retrieve",
            json={"ingredient_list": ingredient_list},
            timeout=SERVICE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        logger.error("RAG Service call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"RAG Service unavailable: {exc}") from exc


def call_llm_service(prompt: str, model: str | None = None, max_tokens: int | None = None, temperature: float | None = None) -> dict:
    try:
        payload: dict = {"prompt": prompt}
        if model:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        resp = requests.post(
            f"{LLM_SERVICE_URL}/generate",
            json=payload,
            timeout=SERVICE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        logger.error("LLM Service call failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"LLM Service unavailable: {exc}") from exc


@app.get("/api/health")
def health():
    """Reports this service's status AND whether downstream services are reachable."""
    downstream = {}
    for name, url in [("rag_service", RAG_SERVICE_URL), ("llm_service", LLM_SERVICE_URL)]:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            downstream[name] = r.json()
        except requests.exceptions.RequestException as exc:
            downstream[name] = {"status": "unreachable", "error": str(exc)}
            
    models = get_models().get("models", [])
    return {
        "status": "ok",
        "service": "app_service",
        "downstream": downstream,
        "available_models": models,
    }


@app.get("/api/models")
def get_models():
    """Proxy to LLM service: list all available Ollama models."""
    try:
        resp = requests.get(f"{LLM_SERVICE_URL}/models", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        logger.warning("Could not fetch models from LLM service: %s", exc)
        try:
            from app.ollama_client import list_models
            return {"models": list_models()}
        except Exception:
            return {"models": ["codellama:latest"]}


@app.post("/api/compare_models")
def compare_models_proxy(request: dict):
    """
    Evaluates and benchmarks all available models with shared RAG context.
    """
    try:
        from rag.model_comparator import compare_all_models
        return compare_all_models(
            ingredient_list=request.get("ingredient_list", ""),
            models=request.get("models"),
            temperature=request.get("temperature", 0.20),
            max_tokens=request.get("max_tokens", 350),
            min_similarity=request.get("min_similarity", 0.30),
        )
    except Exception as exc:
        logger.error("compare_models failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))



@app.get("/api/chunks")
def get_chunks():
    """Proxy to RAG service debug: full KB chunk list for Chunk Explorer."""
    try:
        resp = requests.get(f"{RAG_SERVICE_URL}/debug/chunks", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"RAG Service unavailable: {exc}") from exc


@app.get("/api/vectors")
def get_vectors():
    """Proxy to RAG service debug: embedding matrix snapshot for heatmap."""
    try:
        resp = requests.get(f"{RAG_SERVICE_URL}/debug/vectors", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"RAG Service unavailable: {exc}") from exc


@app.post("/api/query-embed")
def query_embed(body: dict):
    """Proxy to RAG service debug: embed a query and return top-k matches."""
    try:
        resp = requests.post(f"{RAG_SERVICE_URL}/debug/query-embed", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"RAG Service unavailable: {exc}") from exc


@app.get("/api/architecture")
def architecture():
    """Small, presentation-friendly description of the final exercise flow."""
    return {
        "application": "Food Label Decoder",
        "request_flow": [
            "Browser",
            "Application Service",
            "RAG Service",
            "FAISS vector similarity",
            "Application Service prompt orchestration",
            "LLM Service",
            "Ollama + Code Llama",
            "Browser response",
        ],
        "exercise_mapping": {
            "exercise_1": "Baseline API uses Ollama + Code Llama without retrieval.",
            "exercise_2": "Data Service creates chunk embeddings from the knowledge base.",
            "exercise_3": "RAG Service retrieves relevant FAISS vector matches.",
            "exercise_4": "Application, RAG, LLM, and Data responsibilities are separate services.",
            "exercise_5": "Docker Compose runs the complete service architecture.",
        },
    }


@app.post("/api/decode_no_rag", response_model=DecodeResponse)
def decode_no_rag(request: DecodeRequest) -> DecodeResponse:
    """Orchestration: Application Service -> LLM Service only (no retrieval)."""
    ingredient_list = request.ingredient_list.strip()
    if not ingredient_list:
        raise HTTPException(status_code=422, detail="Ingredient list cannot be blank.")

    logger.info("NO-RAG request: %s", ingredient_list)
    started_at = perf_counter()
    prompt = build_no_rag_prompt(ingredient_list)
    llm_result = call_llm_service(prompt, model=request.model, max_tokens=request.max_tokens, temperature=request.temperature)
    return DecodeResponse(
        ingredient_list=ingredient_list,
        explanation=llm_result["response"],
        mode="baseline_no_rag",
        processing_time_ms=round((perf_counter() - started_at) * 1000),
        model_used=llm_result.get("model_used", request.model or ""),
    )


@app.post("/api/decode", response_model=DecodeResponse)
def decode(request: DecodeRequest) -> DecodeResponse:
    """
    Full orchestration lifecycle:

        Application Service -> RAG Service (retrieval) -> context
        Application Service -> LLM Service (prompt = context + question) -> response
    """
    ingredient_list = request.ingredient_list.strip()
    if not ingredient_list:
        raise HTTPException(status_code=422, detail="Ingredient list cannot be blank.")

    logger.info("RAG request: %s", ingredient_list)
    started_at = perf_counter()

    rag_result = call_rag_service(ingredient_list)
    context = rag_result["context"]
    unmatched_ingredients = rag_result.get("unmatched_ingredients", [])

    # Do not let the LLM invent an answer if no local evidence was retrieved.
    if not rag_result["retrieved_chunks"]:
        return DecodeResponse(
            ingredient_list=ingredient_list,
            explanation=(
                "⚠️ No relevant records were found in the local food-ingredient knowledge base "
                "for any of these ingredients. "
                "Please paste the actual comma-separated ingredients from the package label, "
                "for example: 'Sugar, milk solids, E322'. "
                "The model will not guess — this prevents hallucination."
            ),
            mode="rag_no_match",
            processing_time_ms=round((perf_counter() - started_at) * 1000),
            unmatched_ingredients=unmatched_ingredients,
            context_sent_to_llm=context,
        )

    prompt = build_rag_prompt(ingredient_list, context, unmatched_ingredients)
    llm_result = call_llm_service(prompt, model=request.model, max_tokens=request.max_tokens, temperature=request.temperature)

    return DecodeResponse(
        ingredient_list=ingredient_list,
        explanation=llm_result["response"],
        mode="rag_grounded",
        processing_time_ms=round((perf_counter() - started_at) * 1000),
        retrieved_chunks=rag_result["retrieved_chunks"],
        unmatched_ingredients=unmatched_ingredients,
        context_sent_to_llm=context,
        model_used=llm_result.get("model_used", request.model or ""),
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(
        "static/index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )

