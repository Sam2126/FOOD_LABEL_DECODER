import json
import os
from typing import Optional

import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(
    title="Food Label Decoder – Analysis Service",
    description="LLM-based food ingredient safety analysis with RAG context from retrieval service.",
    version="1.0.0",
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")
RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8001/retrieve")


# ── Schemas ───────────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    text: Optional[str] = ""
    context: Optional[str] = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _build_prompt(text: str, context: str) -> str:
    return f"""You are a food safety expert. Analyze this ingredient list.

Regulatory context (use this as your source of truth):
{context}

Ingredient list:
{text}

Return ONLY valid JSON:
{{
  "flagged_ingredients": [
    {{
      "name": "...",
      "reason": "...",
      "confidence": 0.0,
      "supported_by_context": true
    }}
  ],
  "allergens": ["..."],
  "combinations": [{{"ingredients": ["...", "..."], "risk": "..."}}],
  "hallucination_risk": "low",
  "summary": "..."
}}"""


def _fetch_context(text: str) -> str:
    """Call the retrieval service to get RAG context for the ingredient text."""
    try:
        resp = requests.post(
            RETRIEVAL_URL,
            json={"query": text, "collection": "both", "top_k": 3},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("top_context", "")
    except Exception:
        pass
    return ""


def _call_ollama(prompt: str) -> dict:
    """Send prompt to Ollama and parse JSON response."""
    ollama_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    try:
        response = requests.post(OLLAMA_URL, json=ollama_payload, timeout=60)
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
    return {"status": "ok", "service": "analysis"}


@app.post("/analyse")
@app.post("/analyze")
async def analyse(request: Request, payload: Optional[AnalysisRequest] = None):
    """Analyse ingredients WITH RAG context fetched from the retrieval service."""
    text = (payload.text if payload else "") or ""
    if not text:
        try:
            body = await request.json()
            text = body.get("text", "") or body.get("ingredients", "")
        except Exception:
            pass

    context = _fetch_context(text)
    prompt = _build_prompt(text, context)
    return _call_ollama(prompt)


@app.post("/analyse-without-rag")
@app.post("/analyze-without-rag")
async def analyse_without_rag(request: Request, payload: Optional[AnalysisRequest] = None):
    """Analyse ingredients WITHOUT RAG context (empty context string).
    Used for RAG comparison experiments.
    """
    text = (payload.text if payload else "") or ""
    if not text:
        try:
            body = await request.json()
            text = body.get("text", "") or body.get("ingredients", "")
        except Exception:
            pass

    prompt = _build_prompt(text, context="")
    return _call_ollama(prompt)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
