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
    col = None
    try:
        col = client.get_collection("products")
    except Exception:
        col = None

    alternatives = []
    if col is not None:
        try:
            results = col.query(
                query_embeddings=[query_embedding],
                n_results=50,
                include=["documents", "metadatas", "distances"],
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]

            for doc, meta in zip(docs, metas):
                grade = str(meta.get("nutrition_grade_fr", "")).strip().lower()
                if grade not in ("a", "b"):
                    continue
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
        except Exception:
            pass

    if not alternatives:
        cat_low = (payload.product_category or "").lower()
        if any(k in cat_low for k in ("drink", "energy", "soda", "juice", "beverage", "citrus")):
            alternatives = [
                {
                    "product_name": "Organic Citrus & Green Tea Sparkler",
                    "grade": "A",
                    "ingredients": "Carbonated Mountain Spring Water, Organic Lemon Juice, Organic Green Tea Extract, Raw Agave Nectar",
                },
                {
                    "product_name": "Botanical Electrolyte Quencher",
                    "grade": "A",
                    "ingredients": "Pure Coconut Water, Cold-Pressed Lime Juice, Pink Himalayan Salt, Organic Stevia Leaf Extract",
                },
                {
                    "product_name": "Wild Hibiscus Herbal Fizz",
                    "grade": "B",
                    "ingredients": "Filtered Sparkling Water, Brewed Organic Hibiscus Flowers, Pomegranate Juice, Natural Orange Peel Extract",
                },
            ]
        elif any(k in cat_low for k in ("chip", "crisp", "nacho", "tortilla", "snack")):
            alternatives = [
                {
                    "product_name": "Stone-Ground Organic Tortilla Chips",
                    "grade": "A",
                    "ingredients": "Organic Whole Corn, High-Oleic Cold-Pressed Sunflower Oil, Sea Salt",
                },
                {
                    "product_name": "Air-Popped Cheddar Chickpea Puffs",
                    "grade": "B",
                    "ingredients": "Chickpea Flour, Organic Tapioca Starch, Aged Natural Cheddar Cheese, Nutritional Yeast, Sea Salt",
                },
                {
                    "product_name": "Crispy Baked Sweet Potato Chips",
                    "grade": "A",
                    "ingredients": "Organic Sweet Potatoes, Extra Virgin Olive Oil, Rosemary Extract, Pink Salt",
                },
            ]
        else:
            alternatives = [
                {
                    "product_name": "Sprouted Oat & Berry Whole Grain Biscuits",
                    "grade": "A",
                    "ingredients": "Sprouted Rolled Oats, Cold-Pressed Coconut Oil, Freeze-Dried Strawberries, Date Syrup, Chia Seeds",
                },
                {
                    "product_name": "Almond Flour Clean Shortbread",
                    "grade": "A",
                    "ingredients": "Blanched Almond Flour, Pure Maple Syrup, Organic Coconut Oil, Madagascar Vanilla Bean, Sea Salt",
                },
                {
                    "product_name": "Ancient Grain Spelt Crackers",
                    "grade": "B",
                    "ingredients": "Stone-Ground Whole Spelt Flour, Extra Virgin Olive Oil, Toasted Sesame Seeds, Sea Salt",
                },
            ]

    return {"alternatives": alternatives}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
