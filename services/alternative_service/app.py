import os
from pathlib import Path
from typing import List, Optional

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path(__file__).resolve().parents[2] / "knowledge_base" / "chroma_db"

app = FastAPI(
    title="Food Label Decoder – Alternative Service",
    description="Recommends healthier food alternatives using ChromaDB product embeddings.",
    version="1.0.0",
)

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
class AlternativeRequest(BaseModel):
    flagged_ingredients: List[str]
    product_category: Optional[str] = "food"


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "alternative"}


@app.post("/alternatives")
async def get_alternatives(payload: AlternativeRequest):
    """Query the products ChromaDB collection for healthier alternatives.

    Filters to only return products with nutrition_grade_fr of 'a' or 'b'.
    Returns top 3 matches.
    """
    flagged_str = ", ".join(payload.flagged_ingredients)
    query_text = (
        f"healthy alternative without {flagged_str} in {payload.product_category}"
    )

    model = get_model()
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    client = get_client()
    try:
        col = client.get_collection("products")
    except Exception as e:
        return {"alternatives": [], "error": f"Could not access products collection: {e}"}

    # Fetch more results so we can filter by grade
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=50,
        include=["documents", "metadatas", "distances"],
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    alternatives = []
    for doc, meta in zip(docs, metas):
        grade = str(meta.get("nutrition_grade_fr", "")).strip().lower()
        if grade not in ("a", "b"):
            continue

        # Parse product name and ingredients from the chunk text
        # Format: "{product_name}: {ingredients_text} | grade:{grade}"
        parts = doc.split(":", 1)
        product_name = parts[0].strip() if parts else "Unknown"
        ingredients = parts[1].split("|")[0].strip() if len(parts) > 1 else doc

        alternatives.append({
            "product_name": product_name,
            "grade": grade.upper(),
            "ingredients": ingredients,
        })

        if len(alternatives) >= 3:
            break

    return {"alternatives": alternatives}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
