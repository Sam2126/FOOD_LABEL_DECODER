from typing import Optional
from fastapi import FastAPI, Request
from pydantic import BaseModel

app = FastAPI(
    title="Food Label Decoder - Guardrail Service",
    description="Safety and validation guardrail service to determine if text resembles a food ingredient list.",
    version="1.0.0"
)

KEYWORDS = [
    "ingredients",
    "contains",
    "per 100g",
    "mg",
    "sodium",
    "sugar",
    "fat",
    "protein",
    "E numbers",
    "permitted"
]


class GuardrailRequest(BaseModel):
    text: Optional[str] = ""


@app.api_route("/health", methods=["GET", "POST"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "guardrail"}


@app.post("/guardrail")
async def check_guardrail(request: Request, payload: Optional[GuardrailRequest] = None):
    """
    Checks if input text looks like a food ingredient list using keyword matching.
    - If 2+ keywords found -> {"verdict": "pass", "reason": "looks like food label"}
    - If fewer -> {"verdict": "reject", "reason": "input does not appear to be a food label"}
    """
    text = payload.text if (payload and payload.text) else ""

    # Fallback to JSON body if not populated via Pydantic model
    if not text:
        try:
            data = await request.json()
            if isinstance(data, dict):
                text = data.get("text", "") or data.get("raw_text", "")
            elif isinstance(data, str):
                text = data
        except Exception:
            pass

    # Fallback to form data
    if not text:
        try:
            form = await request.form()
            text = form.get("text", "")
        except Exception:
            pass

    # Fallback to raw body
    if not text:
        try:
            body = await request.body()
            if body:
                text = body.decode("utf-8", errors="ignore")
        except Exception:
            pass

    text_lower = (text or "").lower()
    matched_count = sum(1 for kw in KEYWORDS if kw.lower() in text_lower)

    if matched_count >= 2:
        return {
            "verdict": "pass",
            "reason": "looks like food label"
        }
    else:
        return {
            "verdict": "reject",
            "reason": "input does not appear to be a food label"
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
