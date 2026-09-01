"""
Food Label Decoder - Application & API
======================================

Main FastAPI application serving the AI Food Label Decoder.
Supports dynamic model selection, RAG parameter tuning,
detailed vector metrics, and side-by-side RAG vs Baseline evaluation.
"""

import time
import logging
from typing import Optional, List, Dict, Any
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.ollama_client import ask_ollama, list_models, OllamaError, OLLAMA_MODEL
from rag.rag_pipeline import run_rag, build_no_rag_prompt
from rag.model_comparator import compare_all_models
from rag.vector_store import vector_store

from ingestion.embedder import embed_texts

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("food-label-decoder")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Food Label Decoder - Professional AI Studio",
    description="RAG-powered food ingredient, additive, and allergen decoder using Ollama + FAISS",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class DecodeRequest(BaseModel):
    ingredient_list: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Raw ingredient list text as printed on the food label",
        examples=["Wheat flour, sugar, palm oil, E322, E621, milk solids, artificial flavor"],
    )
    model: Optional[str] = Field(default=None, description="Ollama model to use (e.g. codellama, llama3)")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=None, description="Generation temperature (0.0 to 1.0)")
    top_k: Optional[int] = Field(default=2, description="Top-k chunks to retrieve per ingredient")
    min_similarity: Optional[float] = Field(default=0.30, description="Minimum cosine similarity threshold")


class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    name: str
    source: Optional[str] = ""
    score: float
    text: Optional[str] = ""


class DecodeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ingredient_list: str
    explanation: str
    mode: str = "no_rag"
    processing_time_ms: int = 0
    model_used: str = ""
    retrieved_chunks: List[RetrievedChunkSchema] = Field(default_factory=list)
    unmatched_ingredients: List[str] = Field(default_factory=list)
    context_sent_to_llm: str = ""
    detected_allergens: List[str] = Field(default_factory=list)
    dietary_flags: List[str] = Field(default_factory=list)
    allergen_breakdown: List[Dict[str, Any]] = Field(default_factory=list)


class CompareModelsRequest(BaseModel):
    ingredient_list: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Raw ingredient list text as printed on the food label",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Specific models to benchmark (defaults to all installed Ollama models)",
    )
    temperature: Optional[float] = Field(default=0.20)
    max_tokens: Optional[int] = Field(default=450)
    top_k: Optional[int] = Field(default=2)
    min_similarity: Optional[float] = Field(default=0.30)


class CompareModelsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    ingredient_list: str
    models_evaluated: List[str]
    total_evaluation_time_ms: int
    retrieved_chunks: List[RetrievedChunkSchema] = Field(default_factory=list)
    unmatched_ingredients: List[str] = Field(default_factory=list)
    detected_allergens: List[str] = Field(default_factory=list)
    dietary_flags: List[str] = Field(default_factory=list)
    allergen_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    results: Dict[str, Any]
    leaderboard: List[Dict[str, Any]]
    champion_model: str
    verdict_title: str
    referee_rationale: str
    pros_and_cons: Dict[str, Any]


class QueryEmbedRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    min_similarity: Optional[float] = 0.20


@app.on_event("startup")
def load_vector_index_on_startup():
    """
    Loads the FAISS index + chunk metadata into memory at startup.
    """
    try:
        vector_store.load()
        logger.info("FAISS vector store successfully initialized.")
    except Exception as exc:
        logger.warning(
            "Vector index not loaded yet: %s. Run `python -m ingestion.embedder` to build the vector store.",
            exc,
        )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    """Returns system status, vector index readiness, and installed Ollama models."""
    if not vector_store._loaded:
        try:
            vector_store.load()
        except Exception:
            pass

    models_available = list_models()
    index_loaded = bool(vector_store._loaded and vector_store.index is not None)
    chunks_count = len(vector_store.chunks) if index_loaded else 0

    return {
        "status": "ok",
        "index_loaded": index_loaded,
        "chunks_count": chunks_count,
        "vector_dimension": 384,
        "default_model": OLLAMA_MODEL,
        "available_models": models_available,
    }


@app.get("/api/models")
def get_available_models():
    """Returns the list of available Ollama models."""
    models = list_models()
    return {"models": models, "default": OLLAMA_MODEL}


@app.get("/api/chunks")
def get_all_chunks():
    """Returns all stored KB chunks for the Chunk Explorer tab."""
    if not vector_store._loaded:
        try:
            vector_store.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    chunks_summary = []
    for chunk in vector_store.chunks:
        chunks_summary.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "name": chunk.get("metadata", {}).get("name", "Unknown"),
            "category": chunk.get("metadata", {}).get("category", "General"),
            "e_number": chunk.get("metadata", {}).get("e_number"),
            "source": chunk.get("metadata", {}).get("source", ""),
            "text": chunk.get("text", ""),
            "text_preview": chunk.get("text", "")[:240] + ("..." if len(chunk.get("text", "")) > 240 else ""),
        })
    return {"total": len(chunks_summary), "chunks": chunks_summary}


@app.get("/api/vectors")
def get_vector_matrix():
    """Returns a truncated embedding matrix for the interactive heatmap."""
    if not vector_store._loaded:
        try:
            vector_store.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if vector_store.index is None or len(vector_store.chunks) == 0:
        return {"chunks": [], "matrix": [], "dim_count": 0}

    n_total = vector_store.index.ntotal
    full_dim = vector_store.index.d

    max_chunks = min(60, n_total)
    max_dims = min(32, full_dim)

    all_vectors = np.zeros((n_total, full_dim), dtype="float32")
    vector_store.index.reconstruct_n(0, n_total, all_vectors)

    matrix = all_vectors[:max_chunks, :max_dims].tolist()
    chunk_names = [
        c.get("metadata", {}).get("name", f"chunk_{i}")
        for i, c in enumerate(vector_store.chunks[:max_chunks])
    ]

    return {
        "chunks": chunk_names,
        "matrix": matrix,
        "full_dim": full_dim,
        "shown_dims": max_dims,
        "total_chunks": n_total,
        "shown_chunks": max_chunks,
    }


@app.post("/api/query-embed")
def debug_query_embed(request: QueryEmbedRequest):
    """Embeds a single query and returns its top-k matches with similarity scores."""
    if not vector_store._loaded:
        try:
            vector_store.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if not request.query.strip():
        return {"query": "", "embedding_preview": [], "matches": []}

    query_vector = embed_texts([request.query.strip()])[0]
    embedding_preview = query_vector[:32].tolist()

    results = vector_store.search(query_vector, top_k=request.top_k or 5)
    matches = [
        {
            "chunk_id": r["chunk"].get("chunk_id", ""),
            "name": r["chunk"].get("metadata", {}).get("name", ""),
            "score": round(r["score"], 4),
            "text_preview": r["chunk"].get("text", "")[:300],
        }
        for r in results
    ]

    return {
        "query": request.query,
        "embedding_preview": embedding_preview,
        "matches": matches,
    }


@app.post("/api/decode_no_rag", response_model=DecodeResponse)
def decode_ingredients_no_rag(request: DecodeRequest) -> DecodeResponse:
    """
    BASELINE (Without RAG):
    Generates response using LLM internal knowledge alone.
    """
    logger.info("Received NO-RAG decode request with model=%s", request.model or "default")
    start_time = time.perf_counter()

    prompt = build_no_rag_prompt(request.ingredient_list)

    try:
        explanation = ask_ollama(
            prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except OllamaError as exc:
        logger.error("Ollama call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Ollama: {exc}",
        ) from exc

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    return DecodeResponse(
        ingredient_list=request.ingredient_list,
        explanation=explanation,
        mode="no_rag",
        processing_time_ms=elapsed_ms,
        model_used=request.model or "codellama",
        retrieved_chunks=[],
        unmatched_ingredients=[],
        context_sent_to_llm="",
    )


@app.post("/api/decode", response_model=DecodeResponse)
def decode_ingredients(request: DecodeRequest) -> DecodeResponse:
    """
    RAG-POWERED DECODE:
    Performs multi-ingredient query embedding -> FAISS vector retrieval
    -> context injection -> grounded LLM generation.
    """
    logger.info("Received RAG decode request with model=%s", request.model or "default")

    try:
        result = run_rag(
            ingredient_list=request.ingredient_list,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k or 2,
            min_similarity=request.min_similarity if request.min_similarity is not None else 0.30,
        )
    except OllamaError as exc:
        logger.error("Ollama call failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach Ollama: {exc}",
        ) from exc
    except FileNotFoundError as exc:
        logger.error("Vector index missing: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return DecodeResponse(
        ingredient_list=request.ingredient_list,
        explanation=result["explanation"],
        mode=result.get("mode", "rag_grounded"),
        processing_time_ms=result.get("processing_time_ms", 0),
        model_used=result.get("model_used", request.model or "codellama"),
        retrieved_chunks=[
            RetrievedChunkSchema(
                chunk_id=c["chunk_id"],
                name=c["name"],
                source=c.get("source", ""),
                score=c["score"],
                text=c.get("text", ""),
            )
            for c in result.get("retrieved_chunks", [])
        ],
        unmatched_ingredients=result.get("unmatched_ingredients", []),
        context_sent_to_llm=result.get("context_sent_to_llm", ""),
        detected_allergens=result.get("detected_allergens", []),
        dietary_flags=result.get("dietary_flags", []),
        allergen_breakdown=result.get("allergen_breakdown", []),
    )


@app.post(
    "/api/compare_models",
    response_model=CompareModelsResponse,
    summary="Multi-Model Comparison Arena & AI Referee",
    description="Benchmarks all available Ollama models concurrently on a food label and synthesizes an AI referee evaluation.",
)
def compare_models_endpoint(request: CompareModelsRequest):
    try:
        data = compare_all_models(
            ingredient_list=request.ingredient_list,
            models=request.models,
            temperature=request.temperature if request.temperature is not None else 0.20,
            max_tokens=request.max_tokens if request.max_tokens is not None else 450,
            top_k=request.top_k if request.top_k is not None else 2,
            min_similarity=request.min_similarity if request.min_similarity is not None else 0.30,
        )
        return CompareModelsResponse(
            ingredient_list=data["ingredient_list"],
            models_evaluated=data["models_evaluated"],
            total_evaluation_time_ms=data["total_evaluation_time_ms"],
            retrieved_chunks=[
                RetrievedChunkSchema(
                    chunk_id=c["chunk_id"],
                    name=c["name"],
                    source=c.get("source", ""),
                    score=c["score"],
                    text=c.get("text", ""),
                )
                for c in data.get("retrieved_chunks", [])
            ],
            unmatched_ingredients=data.get("unmatched_ingredients", []),
            detected_allergens=data.get("detected_allergens", []),
            dietary_flags=data.get("dietary_flags", []),
            allergen_breakdown=data.get("allergen_breakdown", []),
            results=data["results"],
            leaderboard=data["leaderboard"],
            champion_model=data["champion_model"],
            verdict_title=data["verdict_title"],
            referee_rationale=data["referee_rationale"],
            pros_and_cons=data["pros_and_cons"],
        )
    except Exception as exc:
        logger.error("Compare models endpoint failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc





# ---------------------------------------------------------------------------
# Static frontend mounting
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(
        "app/static/index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"},
    )

