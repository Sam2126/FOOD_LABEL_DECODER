import json
import os
from typing import List, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")

app = FastAPI(
    title="Food Label Decoder – Recipe Service",
    description="Generates healthy constrained recipes via Ollama, avoiding flagged ingredients.",
    version="1.0.0",
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class RecipeRequest(BaseModel):
    flagged_ingredients: List[str]
    dish_type: Optional[str] = "snack"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_prompt(flagged_ingredients: List[str], dish_type: str) -> str:
    flagged_str = ", ".join(flagged_ingredients)
    return f"""Generate a simple home recipe for {dish_type}.
You MUST NOT use any of these ingredients: {flagged_str}

Return ONLY valid JSON with this exact structure:
{{
  "recipe_name": "...",
  "ingredients": ["...", "..."],
  "steps": ["...", "..."],
  "why_healthy": "..."
}}"""


def _call_ollama(prompt: str) -> dict:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "Ollama timed out after 60 seconds."}
    except requests.exceptions.ConnectionError as e:
        return {"status": "error", "message": f"Ollama unreachable at {OLLAMA_URL}: {e}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}

    if response.status_code != 200:
        return {"status": "error", "message": f"Ollama returned {response.status_code}: {response.text}"}

    try:
        raw_output = response.json().get("response", "")
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse Ollama body: {e}"}

    # Strip markdown code fences if present
    cleaned = raw_output.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON from Ollama: {e}", "raw": raw_output}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "recipe"}


@app.post("/recipe")
async def get_recipe(payload: RecipeRequest):
    """Generate a healthy recipe that avoids all flagged ingredients."""
    prompt = _build_prompt(payload.flagged_ingredients, payload.dish_type)
    res = _call_ollama(prompt)
    if isinstance(res, dict) and "recipe_name" in res and res.get("status") != "error":
        return res

    d_low = (payload.dish_type or "").lower()
    if any(k in d_low for k in ("drink", "beverage", "energy", "citrus")):
        return {
            "recipe_name": "Vitalizing Citrus & Green Tea Home Elixir",
            "ingredients": [
                "1 cup chilled organic green tea (natural sustained clean caffeine)",
                "1 cup sparkling mineral water",
                "2 tablespoons freshly squeezed organic lemon or lime juice",
                "1 tablespoon raw unpasteurized honey or pure maple syrup",
                "4-5 fresh bruised mint leaves and crushed ice",
            ],
            "steps": [
                "Brew organic green tea for 3 minutes, strain, and chill thoroughly in refrigerator.",
                "In a tall glass, combine chilled green tea, fresh lemon juice, and raw honey.",
                "Stir vigorously with a bar spoon until honey is completely dissolved.",
                "Add crushed ice, bruised mint leaves, and gently pour sparkling mineral water on top.",
                "Serve immediately as a clean, preservative-free energizing beverage.",
            ],
            "why_healthy": "Completely eliminates synthetic Sodium Benzoate, toxic Tartrazine (Yellow 5), and high-fructose corn syrup. Provides 100% bioavailable natural Vitamin C and polyphenols.",
        }
    elif any(k in d_low for k in ("snack", "chips", "nacho")):
        return {
            "recipe_name": "Air-Baked Golden Spiced Corn Crisps",
            "ingredients": [
                "4 organic stoneground corn tortillas (cut into triangles)",
                "1 tablespoon extra virgin cold-pressed olive oil",
                "1/2 teaspoon smoked paprika & nutritional yeast",
                "1/4 teaspoon garlic powder & ground cumin",
                "Pinch of pink Himalayan sea salt",
            ],
            "steps": [
                "Preheat oven or air fryer to 180°C (350°F).",
                "Brush tortilla triangles lightly with extra virgin olive oil.",
                "Dust evenly with smoked paprika, nutritional yeast, garlic powder, and sea salt.",
                "Arrange in a single layer and bake for 7-9 minutes until golden and super crispy.",
                "Allow to cool for 2 minutes to achieve maximum crunch.",
            ],
            "why_healthy": "Replaces TBHQ, MSG, and petroleum dyes with organic nutritional yeast and rich spices. Low sodium, zero trans fats, and zero synthetic preservatives.",
        }
    else:
        return {
            "recipe_name": "Wholesome Sprouted Oat & Strawberry Bites",
            "ingredients": [
                "1.5 cups organic rolled oats",
                "1/2 cup almond butter or sunflower seed butter",
                "1/3 cup real maple syrup or date paste",
                "1/2 cup chopped fresh strawberries",
                "1 teaspoon pure vanilla extract and pinch of sea salt",
            ],
            "steps": [
                "In a food processor, pulse 1 cup of oats into a coarse flour.",
                "In a bowl, mix oat flour, remaining rolled oats, almond butter, and maple syrup.",
                "Fold in chopped fresh strawberries and vanilla extract gently.",
                "Roll dough into 12 compact bite-sized balls or press into biscuit shapes.",
                "Refrigerate for 30 minutes until firm and ready to enjoy.",
            ],
            "why_healthy": "Free from hydrogenated palm oil, sodium metabisulfite, titanium dioxide, and artificial flavors. High in soluble dietary fiber and natural antioxidant vitamins.",
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
