import json
import os
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI(
    title="Food Label Decoder - Analysis Service",
    description="LLM-based food ingredient safety and regulatory analysis service via Ollama.",
    version="1.0.0"
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "codellama")


class AnalysisRequest(BaseModel):
    text: Optional[str] = ""
    context: Optional[str] = ""


@app.api_route("/health", methods=["GET", "POST"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "analysis"}


@app.post("/analyse")
@app.post("/analyze")
async def analyse_endpoint(request: Request, payload: Optional[AnalysisRequest] = None):
    """
    POST /analyse
    - Accepts: {"text": "...", "context": "..."}
    - Sends prompt to Ollama at http://localhost:11434/api/generate
    - Returns parsed JSON response
    - Handles timeout (30s), Ollama down -> {"status": "error", "message": "..."}
    """
    text = payload.text if (payload and payload.text) else ""
    context = payload.context if (payload and payload.context) else ""

    # Fallback to raw JSON body if not populated via Pydantic model
    if not text or not context:
        try:
            body = await request.json()
            if isinstance(body, dict):
                text = text or body.get("text", "") or body.get("ingredients", "")
                context = context or body.get("context", "")
        except Exception:
            pass

    prompt = f"""You are a food safety expert. Analyze this ingredient list.

Regulatory context:
{context}

Ingredient list:
{text}

Return ONLY valid JSON:
{{
  "flagged_ingredients": [{{"name": "...", "reason": "...", "confidence": 0.0-1.0}}],
  "allergens": ["..."],
  "combinations": [{{"ingredients": ["...", "..."], "risk": "..."}}],
  "summary": "..."
}}"""

    ollama_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }

    try:
        response = requests.post(OLLAMA_URL, json=ollama_payload, timeout=30)
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "Ollama service timed out after 30 seconds."
        }
    except requests.exceptions.ConnectionError as conn_err:
        return {
            "status": "error",
            "message": f"Ollama service is unreachable at {OLLAMA_URL}: {str(conn_err)}"
        }
    except requests.exceptions.RequestException as req_err:
        return {
            "status": "error",
            "message": f"Ollama request error: {str(req_err)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error while contacting Ollama: {str(e)}"
        }

    if response.status_code != 200:
        return {
            "status": "error",
            "message": f"Ollama returned status code {response.status_code}: {response.text}"
        }

    try:
        res_data = response.json()
        raw_output = res_data.get("response", "")
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to parse response body from Ollama: {str(e)}"
        }

    # Clean markdown code blocks if the model formatted with ```json ... ```
    cleaned_output = raw_output.strip()
    if "```json" in cleaned_output:
        cleaned_output = cleaned_output.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned_output:
        cleaned_output = cleaned_output.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed_json = json.loads(cleaned_output)
        return parsed_json
    except json.JSONDecodeError as json_err:
        return {
            "status": "error",
            "message": f"Failed to parse valid JSON from Ollama output: {str(json_err)}"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
