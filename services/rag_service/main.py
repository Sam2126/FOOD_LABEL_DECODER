"""
RAG Service
===========

Responsibility: RETRIEVAL only. Given an ingredient list, return the most
relevant knowledge-base chunks and a ready-to-use context block.

    User Question -> Query Embedding -> Vector Similarity -> Relevant Info -> Context

This service does NOT talk to Ollama/Code Llama - that's the LLM Service's
job. The Orchestrator (Application Service) is what wires RAG Service +
LLM Service together for a full request.

Endpoints:
    GET  /health
    POST /retrieve              {\"ingredient_list\": \"...\"} -> chunks + formatted context
    POST /reload-index          force-reload the FAISS index
    GET  /debug/chunks          full list of KB chunks (for the UI explorer)
    GET  /debug/vectors         embedding matrix snapshot (for the UI heatmap)
    POST /debug/query-embed     embed a single query and return scores (for the inspector)
"""

import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from retriever import retrieve_for_ingredient_list, format_context, retrieve
from vector_store import vector_store
from embedding_utils import embed_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("rag_service")

app = FastAPI(title="Food Label Decoder - RAG Service", version="0.5.0")


class RetrieveRequest(BaseModel):
    ingredient_list: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    name: str
    score: float


class RetrieveResponse(BaseModel):
    retrieved_chunks: list[RetrievedChunk]
    context: str
    unmatched_ingredients: list[str] = []


class QueryEmbedRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


@app.on_event("startup")
def load_index_on_startup():
    try:
        vector_store.load()
    except FileNotFoundError as exc:
        logger.warning("%s (call POST /reload-index once the Data Service has built it)", exc)


@app.get("/health")
def health():
    return {"status": "ok", "service": "rag_service", "index_loaded": vector_store._loaded}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve_endpoint(request: RetrieveRequest) -> RetrieveResponse:
    try:
        results, unmatched_ingredients = retrieve_for_ingredient_list(
            request.ingredient_list,
            include_unmatched=True,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    context = format_context(results)
    chunks = [
        RetrievedChunk(
            chunk_id=r["chunk"]["chunk_id"],
            name=r["chunk"]["metadata"]["name"],
            score=round(r["score"], 4),
        )
        for r in results
    ]
    logger.info("Retrieved %d chunks for '%s'", len(chunks), request.ingredient_list)
    return RetrieveResponse(
        retrieved_chunks=chunks,
        context=context,
        unmatched_ingredients=unmatched_ingredients,
    )


@app.post("/reload-index")
def reload_index():
    try:
        vector_store.reload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "index_loaded": True}


# ---------------------------------------------------------------------------
# Debug / visualisation endpoints (used by the enhanced frontend)
# ---------------------------------------------------------------------------

@app.get("/debug/chunks")
def debug_chunks():
    """Return all stored KB chunks for the Chunk Explorer panel."""
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
            "source": chunk.get("metadata", {}).get("source", ""),
            "text": chunk.get("text", ""),
            "text_preview": chunk.get("text", "")[:200] + ("..." if len(chunk.get("text", "")) > 200 else ""),
        })
    return {"total": len(chunks_summary), "chunks": chunks_summary}


@app.get("/debug/vectors")
def debug_vectors():
    """Return a truncated embedding matrix for the heatmap visualiser.

    Returns at most 60 chunks x 24 dimensions to keep payload small.
    """
    if not vector_store._loaded:
        try:
            vector_store.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    if vector_store.index is None or len(vector_store.chunks) == 0:
        return {"chunks": [], "matrix": [], "dim_count": 0}

    # Reconstruct vectors from FAISS IndexFlatIP
    n_total = vector_store.index.ntotal
    full_dim = vector_store.index.d

    max_chunks = min(60, n_total)
    max_dims = min(24, full_dim)

    # xb contains all stored vectors
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


@app.post("/debug/query-embed")
def debug_query_embed(request: QueryEmbedRequest):
    """Embed a single query string and return its top-k matches with scores.

    Used by the Query Inspector panel to show per-ingredient retrieval.
    """
    if not vector_store._loaded:
        try:
            vector_store.load()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    query_vector = embed_texts([request.query])[0]
    embedding_preview = query_vector[:32].tolist()  # first 32 dims for the bar chart

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
