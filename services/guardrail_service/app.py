from typing import Optional, Dict, Any
import sys
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel

# Import the guardrail engine from same directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from guardrails import (
    run_input_guardrails,
    run_output_guardrails,
    evaluate_guardrail,
    MAX_INPUT_LENGTH,
)

app = FastAPI(
    title="Food Label Decoder - Guardrail Service",
    description=(
        "Multi-layer AI safety guardrail service. "
        "Validates inputs for scope, injection, sufficiency; "
        "validates outputs for schema, grounding, hallucination, and severe claims."
    ),
    version="2.0.0",
)

KEYWORDS = [
    "ingredients", "contains", "per 100g", "mg", "sodium", "sugar",
    "fat", "protein", "E numbers", "permitted"
]


class GuardrailRequest(BaseModel):
    text: Optional[str] = ""


class AdvancedGuardrailRequest(BaseModel):
    text: Optional[str] = ""
    output: Optional[Dict[str, Any]] = None
    mode: Optional[str] = "with_guardrail"  # "with_guardrail" | "without_guardrail"


@app.api_route("/health", methods=["GET", "POST"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "guardrail", "version": "2.0.0"}


@app.post("/guardrail")
async def check_guardrail(request: Request, payload: Optional[GuardrailRequest] = None):
    """
    Basic guardrail: keyword-matching to determine if text looks like a food label.
    Preserved for backwards compatibility.
    """
    text = payload.text if (payload and payload.text) else ""

    if not text:
        try:
            data = await request.json()
            if isinstance(data, dict):
                text = data.get("text", "") or data.get("raw_text", "")
        except Exception:
            pass

    if not text:
        try:
            form = await request.form()
            text = form.get("text", "")
        except Exception:
            pass

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
        return {"verdict": "pass", "reason": "looks like food label"}
    else:
        return {"verdict": "reject", "reason": "input does not appear to be a food label"}


@app.post("/guardrail/advanced")
async def advanced_guardrail(request: Request, payload: Optional[AdvancedGuardrailRequest] = None):
    """
    Advanced multi-layer guardrail:
    - Layer 1 (Input): scope enforcement, prompt injection detection,
      length restriction, gibberish detection, information sufficiency.
    - Layer 2 (Output): schema conformance, hallucination grounding,
      severe claims verification, disclaimer injection.

    Accepts:
        text: str — Input text to validate
        output: dict (optional) — LLM output JSON to validate
        mode: "with_guardrail" | "without_guardrail" — demo comparison mode

    Returns:
        Full evaluation result with per-check pass/fail details.
    """
    text = ""
    llm_output = None
    mode = "with_guardrail"

    if payload:
        text = payload.text or ""
        llm_output = payload.output
        mode = payload.mode or "with_guardrail"

    if not text:
        try:
            data = await request.json()
            if isinstance(data, dict):
                text = data.get("text", "") or data.get("raw_text", "")
                llm_output = data.get("output") or llm_output
                mode = data.get("mode", mode)
        except Exception:
            pass

    result = evaluate_guardrail(
        input_text=text,
        output=llm_output,
        mode=mode,
    )
    return result


@app.post("/guardrail/input-only")
async def input_guardrail_only(request: Request, payload: Optional[GuardrailRequest] = None):
    """Run only input guardrails (Layer 1) — for use when checking before LLM call."""
    text = payload.text if (payload and payload.text) else ""
    if not text:
        try:
            data = await request.json()
            if isinstance(data, dict):
                text = data.get("text", "")
        except Exception:
            pass

    return run_input_guardrails(text)


@app.post("/guardrail/output-only")
async def output_guardrail_only(request: Request):
    """Run only output guardrails (Layer 2) — for use after LLM response received."""
    try:
        data = await request.json()
        output = data.get("output", {})
        source_text = data.get("source_text", "")
    except Exception:
        return {"error": "Invalid JSON body. Expected {'output': {...}, 'source_text': '...'}"}

    return run_output_guardrails(output, source_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
