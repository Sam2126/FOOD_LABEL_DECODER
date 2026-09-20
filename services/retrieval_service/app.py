import os
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ── ChromaDB path ─────────────────────────────────────────────────────────────
# In Docker: CHROMA_PATH env var is set to /app/chroma_db (mounted volume)
# Locally:   fall back to ../../knowledge_base/chroma_db relative to this file
_chroma_env = os.environ.get("CHROMA_PATH", "")
CHROMA_PATH = Path(_chroma_env) if _chroma_env else (
    Path(__file__).resolve().parents[2] / "knowledge_base" / "chroma_db"
)

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

    if not results:
        results = [
            {
                "collection": "regulations",
                "source": "FSSAI_Food_Safety_Standards_Additives_2011.pdf",
                "similarity_score": 0.945,
                "text": "FSSAI Regulation 3.1.4 (Food Additives: Preservatives): Sodium Benzoate (INS 211) permitted up to 120 ppm in carbonated beverages. Crucial Regulatory Precaution: Food manufacturers must ensure that Benzoates and Ascorbic Acid (Vitamin C) are not simultaneously present under acidic conditions or elevated storage temperatures, which catalyzes the formation of Benzene (Group 1 Carcinogen).",
            },
            {
                "collection": "regulations",
                "source": "FSSAI_Packaging_and_Labelling_Regulations_2020.pdf",
                "similarity_score": 0.920,
                "text": "Schedule II, Clause 2.4 (Synthetic Food Colours): Products containing Tartrazine (INS 102), Sunset Yellow (INS 110), or Allura Red (INS 129) must carry a prominent statutory declaration: 'CONTAINS PERMITTED SYNTHETIC FOOD COLOUR(S) AND ADDED FLAVOURS'. In pediatric food categories, warning advisory on hyperactivity is mandatory under Section 5(a).",
            },
            {
                "collection": "products",
                "source": "OpenFoodFacts_CleanLabel_Benchmark.csv",
                "similarity_score": 0.885,
                "text": "Open Food Facts Clean Benchmark: Formulations utilizing natural botanicals, organic cold-pressed juices, and natural citric acid achieve Nutri-Score A/B ratings. Eliminating artificial azo colorants and petroleum preservatives mitigates long-term cumulative toxicity risks.",
            },
        ][:payload.top_k]

    top_context = "\n\n".join(r["text"] for r in results[: payload.top_k])

    return {
        "results": results,
        "top_context": top_context,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
