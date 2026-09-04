import json
import os
import sys
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Make graph module importable when running as standalone
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import detect_combinations, build_graph

app = FastAPI(
    title="Food Label Decoder – Analysis Service",
    description="LLM-based food ingredient safety analysis with RAG context and combination graph.",
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
    """Call the retrieval service to get RAG context."""
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
    """Send prompt to Ollama and return parsed JSON."""
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

    cleaned = raw_output.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON from Ollama: {e}", "raw": raw_output}


def _enrich_with_graph(result: dict) -> dict:
    """Append combination_graph to an analysis result dict."""
    if "status" in result and result.get("status") == "error":
        return result
    flagged = [f.get("name", "") for f in result.get("flagged_ingredients", [])]
    combos = detect_combinations(flagged)
    # Merge LLM-detected combos with rule-based combos (deduplicate)
    llm_combos = result.get("combinations", [])
    all_combos = llm_combos + [c for c in combos if c not in llm_combos]
    result["combinations"] = all_combos
    result["combination_graph"] = build_graph(flagged, all_combos)
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "analysis"}


@app.post("/analyse")
@app.post("/analyze")
async def analyse(request: Request, payload: Optional[AnalysisRequest] = None):
    """Analyse ingredients WITH RAG context from retrieval service."""
    text = (payload.text if payload else "") or ""
    if not text:
        try:
            body = await request.json()
            text = body.get("text", "") or body.get("ingredients", "")
        except Exception:
            pass

    context = _fetch_context(text)
    prompt = _build_prompt(text, context)
    result = _call_ollama(prompt)
    return _enrich_with_graph(result)


@app.post("/analyse-without-rag")
@app.post("/analyze-without-rag")
async def analyse_without_rag(request: Request, payload: Optional[AnalysisRequest] = None):
    """Analyse ingredients WITHOUT RAG context (empty string).
    Used for RAG A/B comparison demo.
    """
    text = (payload.text if payload else "") or ""
    if not text:
        try:
            body = await request.json()
            text = body.get("text", "") or body.get("ingredients", "")
        except Exception:
            pass

    prompt = _build_prompt(text, context="")
    result = _call_ollama(prompt)
    return _enrich_with_graph(result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
