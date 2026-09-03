from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="Food Label Decoder - Retrieval Service",
    description="RAG retrieval service over nutritional databases, scientific studies, and additive registries.",
    version="1.0.0"
)

@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "retrieval"}

class RetrievalQuery(BaseModel):
    query: Optional[str] = None
    text: Optional[str] = None
    top_k: Optional[int] = 5

@app.post("/retrieve")
@app.post("/search")
async def retrieve_context(payload: Optional[RetrievalQuery] = None):
    """Placeholder endpoint for querying nutritional knowledge base context."""
    return {
        "status": "ok",
        "service": "retrieval",
        "context": "Standard FDA/WHO nutritional guidelines: excessive added sugars (>10% daily energy) and high sodium intake are linked to adverse cardiovascular risks. Artificial dyes such as Tartrazine (E102) and preservatives like Sodium Benzoate may induce hypersensitivity.",
        "documents": []
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
