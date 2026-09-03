from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="Food Label Decoder - Alternative Service",
    description="Recommender service suggesting healthier, cleaner, or allergen-free food alternatives.",
    version="1.0.0"
)

@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "alternative"}

class AlternativeRequest(BaseModel):
    product_name: Optional[str] = None
    ingredients: Optional[Any] = None
    dietary_preferences: Optional[List[str]] = None

@app.post("/alternatives")
@app.post("/suggest")
async def get_alternatives(payload: Optional[AlternativeRequest] = None):
    """Placeholder endpoint for recommending healthier alternatives."""
    return {
        "status": "ok",
        "service": "alternative",
        "alternatives": [
            {
                "product_name": "Organic Whole Grain Oats",
                "brand": "Nature's Path",
                "reason": "No added refined sugars or artificial preservatives",
                "score_improvement": 2.5
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
