import os
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── ChromaDB path ─────────────────────────────────────────────────────────────
# When running from services/retrieval_service/, go up two levels to repo root
CHROMA_PATH = Path(__file__).resolve().parents[2] / "knowledge_base" / "chroma_db"

app = FastAPI(
    title="Food Label Decoder – Retrieval Service",
    description="Semantic RAG retrieval over FSSAI regulations and Open Food Facts products.",
    version="1.0.0",
)

# ── Lazy-load heavy objects once at startup ───────────────────────────────────
_model: Optional[SentenceTransformer] = None
_client: Optional[chromadb.PersistentClient] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return _client


# ── Schemas ───────────────────────────────────────────────────────────────────
class RetrievalRequest(BaseModel):
    query: str
    collection: str = "both"   # "regulations" | "products" | "both"
    top_k: int = 3


# ── Helpers ───────────────────────────────────────────────────────────────────
def _query_collection(collection_name: str, query_embedding: List[float], top_k: int) -> List[dict]:
    client = get_client()
    try:
        col = client.get_collection(collection_name)
    except Exception:
        return []

    results = col.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        # ChromaDB returns L2 distance; convert to cosine-like similarity (0-1)
        similarity = max(0.0, 1.0 - dist / 2.0)
        output.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "similarity_score": round(similarity, 4),
            "collection": collection_name,
        })
    return output


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "retrieval"}


@app.post("/retrieve")
async def retrieve(payload: RetrievalRequest):
    """Embed the query and retrieve top-k results from ChromaDB."""
    model = get_model()
    query_embedding = model.encode(payload.query, normalize_embeddings=True).tolist()

    results: List[dict] = []

    if payload.collection in ("regulations", "both"):
        results.extend(_query_collection("regulations", query_embedding, payload.top_k))

    if payload.collection in ("products", "both"):
        results.extend(_query_collection("products", query_embedding, payload.top_k))

    # Sort all results by similarity descending
    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    top_context = "\n\n".join(r["text"] for r in results[: payload.top_k])

    return {
        "results": results,
        "top_context": top_context,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
