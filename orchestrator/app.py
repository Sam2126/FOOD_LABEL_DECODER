import os
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, File, Form, Request, UploadFile
from pydantic import BaseModel

app = FastAPI(
    title="Food Label Decoder – Orchestrator",
    description="Central pipeline orchestrator: OCR → Guardrail → Drift → Retrieval → Analysis → Alternatives → Recipe.",
    version="1.0.0",
)

# ── Service URLs (all overridable via env vars) ───────────────────────────────
OCR_URL        = os.environ.get("OCR_URL",        "http://localhost:8002/ocr")
GUARDRAIL_URL  = os.environ.get("GUARDRAIL_URL",  "http://localhost:8007/guardrail")
DRIFT_URL      = os.environ.get("DRIFT_URL",      "http://localhost:8008/scan")
RETRIEVAL_URL  = os.environ.get("RETRIEVAL_URL",  "http://localhost:8001/retrieve")
ANALYSIS_URL   = os.environ.get("ANALYSIS_URL",   "http://localhost:8003/analyse")
ALTERNATIVE_URL = os.environ.get("ALTERNATIVE_URL", "http://localhost:8005/alternatives")
RECIPE_URL     = os.environ.get("RECIPE_URL",     "http://localhost:8006/recipe")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _trace(name: str, status: str, duration_ms: float, summary: str) -> Dict:
    return {
        "service": name,
        "status": status,
        "duration_ms": duration_ms,
        "output_summary": summary,
    }


def _infer_dish_type(product_name: str, flagged: List[str]) -> str:
    name = (product_name or "").lower()
    if any(k in name for k in ("cookie", "biscuit", "cake", "muffin")):
        return "baked snack"
    if any(k in name for k in ("chips", "crisp", "popcorn", "nachos")):
        return "snack"
    if any(k in name for k in ("drink", "juice", "soda", "cola", "beverage")):
        return "drink"
    if any(k in name for k in ("sauce", "ketchup", "dip", "dressing")):
        return "sauce"
    return "healthy snack"


# ── Schemas ───────────────────────────────────────────────────────────────────
class ProcessJSONRequest(BaseModel):
    text: Optional[str] = ""
    product_name: Optional[str] = "Unknown Product"


# ── Health ────────────────────────────────────────────────────────────────────
@app.api_route("/health", methods=["GET", "POST"])
def health():
    return {"status": "ok", "service": "orchestrator"}


# ── Shared pipeline logic ─────────────────────────────────────────────────────
async def _run_pipeline(
    request: Request,
    upload_file: Optional[UploadFile],
    req_text: str,
    req_product_name: str,
    use_rag: bool,
) -> Dict[str, Any]:

    pipeline_trace: List[Dict] = []

    # Fallback to JSON body if form fields are empty
    if upload_file is None and not req_text:
        try:
            body = await request.json()
            if isinstance(body, dict):
                req_text = body.get("text", "") or body.get("raw_text", "")
                req_product_name = body.get("product_name", req_product_name)
        except Exception:
            pass

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    extracted_text = req_text
    ocr_status = "skipped (text provided)"

    if upload_file is not None:
        try:
            file_bytes = await upload_file.read()
            ocr_res = requests.post(
                OCR_URL,
                files={"file": (upload_file.filename or "image.png", file_bytes, upload_file.content_type or "image/png")},
                timeout=30,
            )
            ocr_data = ocr_res.json() if ocr_res.ok else {}
            extracted_text = ocr_data.get("extracted_text", req_text)
            ocr_status = "ok" if ocr_res.ok else f"error_{ocr_res.status_code}"
        except Exception as e:
            ocr_status = f"error: {e}"

    dur_ocr = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("ocr-service", ocr_status, dur_ocr,
                                 f"extracted {len(extracted_text)} chars"))

    # ── Step 2: Guardrail ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    guardrail_verdict = "pass"
    guardrail_data: Dict = {}
    gr_status = "ok"

    try:
        gr_res = requests.post(GUARDRAIL_URL, json={"text": extracted_text}, timeout=10)
        if gr_res.ok:
            guardrail_data = gr_res.json()
            guardrail_verdict = guardrail_data.get("verdict", "pass")
            gr_status = guardrail_verdict
        else:
            gr_status = f"error_{gr_res.status_code}"
    except Exception as e:
        gr_status = f"error: {e}"

    dur_gr = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("guardrail-service", gr_status, dur_gr,
                                 guardrail_data.get("reason", "passed")))

    if guardrail_verdict == "reject":
        return {
            "verdict": "reject",
            "reason": guardrail_data.get("reason", "input does not appear to be a food label"),
            "extracted_text": extracted_text,
            "pipeline_trace": pipeline_trace,
        }

    # ── Step 3: Drift ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    drift_info: Dict = {}
    drift_status = "ok"

    try:
        drift_res = requests.post(
            DRIFT_URL,
            json={"product_name": req_product_name, "ingredients": extracted_text},
            timeout=10,
        )
        if drift_res.ok:
            drift_info = drift_res.json()
            drift_status = drift_info.get("status", "ok")
        else:
            drift_status = f"error_{drift_res.status_code}"
    except Exception as e:
        drift_status = f"error: {e}"

    dur_drift = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("drift-service", drift_status, dur_drift,
                                 str(drift_info.get("drift_detected", "unknown"))))

    # ── Step 4: Retrieval (only when use_rag=True) ────────────────────────────
    t0 = time.perf_counter()
    retrieval_data: Dict = {}
    retrieval_status = "skipped"
    chunks_count = 0

    if use_rag:
        try:
            ret_res = requests.post(
                RETRIEVAL_URL,
                json={"query": extracted_text, "collection": "both", "top_k": 3},
                timeout=15,
            )
            if ret_res.ok:
                retrieval_data = ret_res.json()
                retrieval_status = "ok"
                chunks_count = len(retrieval_data.get("results", []))
            else:
                retrieval_status = f"error_{ret_res.status_code}"
        except Exception as e:
            retrieval_status = f"error: {e}"

    dur_ret = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("retrieval-service", retrieval_status, dur_ret,
                                 f"{chunks_count} chunks retrieved"))

    # ── Step 5: Analysis ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    flags_info: Dict = {}
    analysis_status = "ok"

    analysis_endpoint = ANALYSIS_URL if use_rag else ANALYSIS_URL.replace("/analyse", "/analyse-without-rag")

    try:
        ana_res = requests.post(
            analysis_endpoint,
            json={"text": extracted_text},
            timeout=60,
        )
        if ana_res.ok:
            flags_info = ana_res.json()
            analysis_status = flags_info.get("status", "ok") if isinstance(flags_info, dict) else "ok"
        else:
            analysis_status = f"error_{ana_res.status_code}"
    except Exception as e:
        analysis_status = f"error: {e}"

    flagged_ingredients = [
        f.get("name", "") for f in flags_info.get("flagged_ingredients", [])
    ] if isinstance(flags_info, dict) else []

    dur_ana = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("analysis-service", analysis_status, dur_ana,
                                 f"{len(flagged_ingredients)} ingredients flagged"))

    # ── Step 6: Alternatives ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    alternatives_list: List = []
    alt_status = "ok"

    try:
        alt_res = requests.post(
            ALTERNATIVE_URL,
            json={"flagged_ingredients": flagged_ingredients, "product_category": req_product_name},
            timeout=15,
        )
        if alt_res.ok:
            alt_data = alt_res.json()
            alternatives_list = alt_data.get("alternatives", []) if isinstance(alt_data, dict) else alt_data
        else:
            alt_status = f"error_{alt_res.status_code}"
    except Exception as e:
        alt_status = f"error: {e}"

    dur_alt = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("alternative-service", alt_status, dur_alt,
                                 f"{len(alternatives_list)} alternatives found"))

    # ── Step 7: Recipe ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    recipe_info: Dict = {}
    recipe_status = "ok"
    dish_type = _infer_dish_type(req_product_name, flagged_ingredients)

    try:
        rec_res = requests.post(
            RECIPE_URL,
            json={"flagged_ingredients": flagged_ingredients, "dish_type": dish_type},
            timeout=60,
        )
        if rec_res.ok:
            recipe_info = rec_res.json()
        else:
            recipe_status = f"error_{rec_res.status_code}"
    except Exception as e:
        recipe_status = f"error: {e}"

    dur_rec = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append(_trace("recipe-service", recipe_status, dur_rec,
                                 recipe_info.get("recipe_name", "generated") if isinstance(recipe_info, dict) else "generated"))

    return {
        "extracted_text": extracted_text,
        "drift": drift_info,
        "flags": flags_info,
        "combination_graph": flags_info.get("combination_graph", {}),
        "retrieval_results": retrieval_data.get("results", []),
        "alternatives": alternatives_list,
        "recipe": recipe_info,
        "pipeline_trace": pipeline_trace,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/process")
async def process_pipeline(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    product_name: Optional[str] = Form(None),
):
    """Full pipeline WITH RAG retrieval (Step 4 enabled)."""
    return await _run_pipeline(
        request,
        upload_file=file or image,
        req_text=text or "",
        req_product_name=product_name or "Unknown Product",
        use_rag=True,
    )


@app.post("/process-no-rag")
async def process_pipeline_no_rag(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    product_name: Optional[str] = Form(None),
):
    """Same pipeline but WITHOUT RAG retrieval — calls /analyse-without-rag.
    Used for RAG A/B comparison demo.
    """
    return await _run_pipeline(
        request,
        upload_file=file or image,
        req_text=text or "",
        req_product_name=product_name or "Unknown Product",
        use_rag=False,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
