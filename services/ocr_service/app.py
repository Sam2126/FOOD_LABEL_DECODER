import io
import os
import re
import shutil
from typing import Optional

from fastapi import FastAPI, File, Form, Request, UploadFile
from PIL import Image, UnidentifiedImageError
import pytesseract

app = FastAPI(
    title="Food Label Decoder - OCR Service",
    description="FastAPI service extracting text from food nutrition label images using Tesseract, with raw text fallback.",
    version="1.0.0",
)

# Auto-detect Tesseract binary path on Windows / environments if not in default PATH
if not shutil.which("tesseract"):
    possible_paths = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in possible_paths:
        if candidate and os.path.isfile(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
            break


@app.api_route("/health", methods=["GET", "POST"])
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "ocr"}


@app.post("/ocr")
async def ocr_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    raw_text: Optional[str] = Form(None),
):
    """
    Accepts a multipart image file OR raw text string.
    - If image: run pytesseract.image_to_string(), clean whitespace
    - If text: pass through directly
    - Returns: {"extracted_text": "...", "input_type": "image|text", "status": "ok"}
    - Errors: unreadable image -> {"status": "error", "message": "..."}
    """
    upload_file = file or image
    input_text = text or raw_text

    # Fallback: check JSON body if text was not provided in form-data
    if upload_file is None and input_text is None:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_json = await request.json()
                if isinstance(body_json, dict):
                    input_text = (
                        body_json.get("text")
                        or body_json.get("raw_text")
                        or body_json.get("content")
                    )
            except Exception:
                pass
        elif "text/plain" in content_type:
            try:
                raw_body = await request.body()
                if raw_body:
                    input_text = raw_body.decode("utf-8", errors="replace")
            except Exception:
                pass

    # Process multipart image file
    if upload_file is not None:
        try:
            file_bytes = await upload_file.read()
            if not file_bytes:
                return {
                    "status": "error",
                    "message": "Unreadable image: uploaded file is empty.",
                }

            try:
                pil_img = Image.open(io.BytesIO(file_bytes))
                pil_img.load()  # Force decode to detect corrupted or unreadable images early
                if pil_img.mode in ("RGBA", "P"):
                    pil_img = pil_img.convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError) as img_err:
                return {
                    "status": "error",
                    "message": f"Unreadable image: {str(img_err)}",
                }

            # Run Tesseract OCR
            try:
                raw_extracted = pytesseract.image_to_string(pil_img)
            except pytesseract.TesseractNotFoundError as t_err:
                return {
                    "status": "error",
                    "message": f"Tesseract engine not found: {str(t_err)}",
                }
            except Exception as ocr_err:
                return {
                    "status": "error",
                    "message": f"Unreadable image: OCR processing failed - {str(ocr_err)}",
                }

            # Clean whitespace: normalize multiple spaces/tabs, strip lines, remove empty lines
            lines = [
                re.sub(r"[ \t]+", " ", line).strip()
                for line in raw_extracted.splitlines()
            ]
            cleaned_text = "\n".join(line for line in lines if line).strip()

            return {
                "extracted_text": cleaned_text,
                "input_type": "image",
                "status": "ok",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Unreadable image: {str(e)}",
            }

    # Process raw text string
    if input_text is not None:
        return {
            "extracted_text": input_text,
            "input_type": "text",
            "status": "ok",
        }

    return {
        "status": "error",
        "message": "Missing input: please provide a multipart image file or raw text string.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
