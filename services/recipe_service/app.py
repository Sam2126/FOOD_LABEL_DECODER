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
    return _call_ollama(prompt)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
