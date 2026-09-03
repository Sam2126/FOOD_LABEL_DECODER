import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from pydantic import BaseModel
import requests

app = FastAPI(
    title="Food Label Decoder - Orchestrator",
    description="Central pipeline orchestrator executing OCR, Guardrails, Drift, Retrieval, Analysis, Alternatives, and Recipe generation.",
    version="1.0.0"
)

# Configurable service URLs with environment variable overrides
OCR_URL = os.environ.get("OCR_SERVICE_URL", "http://ocr-service/ocr")
GUARDRAIL_URL = os.environ.get("GUARDRAIL_SERVICE_URL", "http://guardrail-service/guardrail")
DRIFT_URL = os.environ.get("DRIFT_SERVICE_URL", "http://drift-service/scan")
RETRIEVAL_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://retrieval-service/retrieve")
ANALYSIS_URL = os.environ.get("ANALYSIS_SERVICE_URL", "http://analysis-service/analyse")
ALTERNATIVE_URL = os.environ.get("ALTERNATIVE_SERVICE_URL", "http://alternative-service/alternatives")
RECIPE_URL = os.environ.get("RECIPE_SERVICE_URL", "http://recipe-service/recipe")


class ProcessJSONRequest(BaseModel):
    text: Optional[str] = ""
    product_name: Optional[str] = "Unknown Product"


@app.api_route("/health", methods=["GET", "POST"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "orchestrator"}


@app.post("/process")
async def process_pipeline(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    product_name: Optional[str] = Form(None)
):
    """
    Orchestrate pipeline in exact order:
      Step 1: POST http://ocr-service/ocr -> get extracted_text
      Step 2: POST http://guardrail-service/guardrail -> if reject, return rejection
      Step 3: POST http://drift-service/scan -> get drift info
      Step 4: POST http://retrieval-service/retrieve -> get context
      Step 5: POST http://analysis-service/analyse -> get flags
      Step 6: POST http://alternative-service/alternatives -> get alternatives
      Step 7: POST http://recipe-service/recipe -> get recipe
    """
    pipeline_trace: List[Dict[str, Any]] = []

    upload_file = file or image
    req_text = text or ""
    req_product_name = product_name or "Unknown Product"

    # Fallback to JSON payload if text / product_name were not provided via form
    if upload_file is None and not req_text:
        try:
            body = await request.json()
            if isinstance(body, dict):
                req_text = body.get("text", "") or body.get("raw_text", "")
                req_product_name = body.get("product_name", req_product_name)
        except Exception:
            pass

    # -------------------------------------------------------------
    # Step 1: OCR Service
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    extracted_text = req_text
    ocr_status = "ok"

    try:
        if upload_file is not None:
            file_bytes = await upload_file.read()
            files_payload = {
                "file": (
                    upload_file.filename or "image.png",
                    file_bytes,
                    upload_file.content_type or "image/png"
                )
            }
            ocr_res = requests.post(OCR_URL, files=files_payload, timeout=30)
        else:
            ocr_res = requests.post(OCR_URL, data={"text": req_text}, timeout=30)

        ocr_data = ocr_res.json() if ocr_res.status_code == 200 else {}
        extracted_text = ocr_data.get("extracted_text", req_text)
        ocr_status = ocr_data.get("status", "ok" if ocr_res.ok else "error")
    except Exception as e:
        ocr_status = f"error: {str(e)}"

    dur_ocr = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append({
        "service": "ocr-service",
        "status": ocr_status,
        "duration_ms": dur_ocr
    })

    # -------------------------------------------------------------
    # Step 2: Guardrail Service
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    guardrail_verdict = "pass"
    guardrail_data = {}

    try:
        gr_res = requests.post(GUARDRAIL_URL, json={"text": extracted_text}, timeout=10)
        if gr_res.status_code == 200:
            guardrail_data = gr_res.json()
            guardrail_verdict = guardrail_data.get("verdict", "pass")
            gr_status = guardrail_verdict
        else:
            gr_status = f"error_{gr_res.status_code}"
    except Exception as e:
        gr_status = f"error: {str(e)}"

    dur_gr = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append({
        "service": "guardrail-service",
        "status": gr_status,
        "duration_ms": dur_gr
    })

    # If rejected, stop and return rejection immediately
    if guardrail_verdict == "reject":
        return {
            "verdict": "reject",
            "reason": guardrail_data.get("reason", "input does not appear to be a food label"),
            "extracted_text": extracted_text,
            "pipeline_trace": pipeline_trace
        }

    # -------------------------------------------------------------
    # Step 3: Drift Service
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    drift_info: Dict[str, Any] = {}
    drift_status = "ok"

    try:
        drift_res = requests.post(
            DRIFT_URL,
            json={"product_name": req_product_name, "ingredients": extracted_text},
            timeout=10
        )
        if drift_res.status_code == 200:
            drift_info = drift_res.json()
            drift_status = drift_info.get("status", "ok")
        else:
            drift_status = f"error_{drift_res.status_code}"
    except Exception as e:
        drift_status = f"error: {str(e)}"

    dur_drift = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append({
        "service": "drift-service",
        "status": drift_status,
        "duration_ms": dur_drift
    })

    # -------------------------------------------------------------
    # Step 4: Retrieval Service
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    retrieval_context = ""
    retrieval_status = "ok"

    try:
        ret_res = requests.post(
            RETRIEVAL_URL,
            json={"text": extracted_text, "query": extracted_text},
            timeout=10
        )
        if ret_res.status_code == 200:
            ret_data = ret_res.json()
            retrieval_context = ret_data.get("context", "")
            retrieval_status = ret_data.get("status", "ok")
        else:
            retrieval_status = f"error_{ret_res.status_code}"
    except Exception as e:
        retrieval_status = f"error: {str(e)}"

    dur_ret = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append({
        "service": "retrieval-service",
        "status": retrieval_status,
        "duration_ms": dur_ret
    })

    # -------------------------------------------------------------
    # Step 5: Analysis Service
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    flags_info: Dict[str, Any] = {}
    analysis_status = "ok"

    try:
        ana_res = requests.post(
            ANALYSIS_URL,
            json={"text": extracted_text, "context": retrieval_context},
            timeout=35
        )
        if ana_res.status_code == 200:
            flags_info = ana_res.json()
            analysis_status = flags_info.get("status", "ok") if isinstance(flags_info, dict) else "ok"
        else:
            analysis_status = f"error_{ana_res.status_code}"
    except Exception as e:
        analysis_status = f"error: {str(e)}"

    dur_ana = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append({
        "service": "analysis-service",
        "status": analysis_status,
        "duration_ms": dur_ana
    })

    # -------------------------------------------------------------
    # Step 6: Alternative Service
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    alternatives_list: List[Any] = []
    alt_status = "ok"

    try:
        alt_res = requests.post(
            ALTERNATIVE_URL,
            json={"product_name": req_product_name, "ingredients": extracted_text},
            timeout=10
        )
        if alt_res.status_code == 200:
            alt_data = alt_res.json()
            if isinstance(alt_data, dict):
                alternatives_list = alt_data.get("alternatives", [])
            elif isinstance(alt_data, list):
                alternatives_list = alt_data
            alt_status = "ok"
        else:
            alt_status = f"error_{alt_res.status_code}"
    except Exception as e:
        alt_status = f"error: {str(e)}"

    dur_alt = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append({
        "service": "alternative-service",
        "status": alt_status,
        "duration_ms": dur_alt
    })

    # -------------------------------------------------------------
    # Step 7: Recipe Service
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    recipe_info: Dict[str, Any] = {}
    recipe_status = "ok"

    try:
        rec_res = requests.post(
            RECIPE_URL,
            json={"target_dish": req_product_name, "ingredients": extracted_text},
            timeout=10
        )
        if rec_res.status_code == 200:
            rec_data = rec_res.json()
            if isinstance(rec_data, dict):
                recipe_info = rec_data.get("recipe", rec_data)
            recipe_status = "ok"
        else:
            recipe_status = f"error_{rec_res.status_code}"
    except Exception as e:
        recipe_status = f"error: {str(e)}"

    dur_rec = round((time.perf_counter() - t0) * 1000, 2)
    pipeline_trace.append({
        "service": "recipe-service",
        "status": recipe_status,
        "duration_ms": dur_rec
    })

    # -------------------------------------------------------------
    # Return Assembled JSON Response
    # -------------------------------------------------------------
    return {
        "extracted_text": extracted_text,
        "drift": drift_info,
        "flags": flags_info,
        "alternatives": alternatives_list,
        "recipe": recipe_info,
        "pipeline_trace": pipeline_trace
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
