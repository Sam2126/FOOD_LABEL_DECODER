from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

app = FastAPI(
    title="Food Label Decoder - Recipe Service",
    description="Custom healthy recipe generation service for homemade alternatives.",
    version="1.0.0"
)

@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "recipe"}

class RecipeRequest(BaseModel):
    target_dish: Optional[str] = None
    ingredients: Optional[Any] = None
    exclude_ingredients: Optional[List[str]] = None
    nutrition_goals: Optional[Dict[str, Any]] = None

@app.post("/recipe")
@app.post("/generate")
async def get_recipe(payload: Optional[RecipeRequest] = None):
    """Placeholder endpoint for generating recipes."""
    return {
        "status": "ok",
        "service": "recipe",
        "recipe": {
            "title": "Homemade Honey-Toasted Granola",
            "prep_time": "10 mins",
            "cook_time": "25 mins",
            "ingredients": ["Rolled oats", "Honey", "Almonds", "Cinnamon", "Flaxseed"],
            "instructions": "Mix oats, nuts, and honey. Bake at 325F (165C) for 25 minutes, stirring halfway."
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
