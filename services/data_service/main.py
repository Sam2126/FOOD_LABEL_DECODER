"""
Data Service
============

Responsibility: OWN the knowledge base.

    Knowledge Base -> Data Processing -> Chunking -> Embeddings -> Vector Storage

This service is the only one that reads knowledge_base/data.json directly.
Other services (RAG Service) only ever read the derived vector index that
this service produces - they never touch the raw knowledge base.

Endpoints:
    GET  /health              - liveness check
    GET  /records              - list all knowledge-base records (debugging/inspection)
    GET  /records/{record_id}  - get a single record
    POST /rebuild-index        - re-run chunking + embedding, refresh the vector index
"""

import logging
from fastapi import FastAPI, HTTPException

from ingestion.chunker import load_knowledge_base
from ingestion.embedder import build_and_save_index, VECTORS_PATH, CHUNKS_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("data_service")

app = FastAPI(title="Food Label Decoder - Data Service", version="0.4.0")


@app.on_event("startup")
def build_index_if_missing():
    """
    Convenience for first-run / Docker environments: if no vector index
    exists yet on the shared volume, build one automatically so the RAG
    Service has something to load without a manual curl step.
    """
    if VECTORS_PATH.exists() and CHUNKS_PATH.exists():
        logger.info("Vector index already exists at %s, skipping auto-build.", VECTORS_PATH)
        return
    logger.info("No vector index found - building it now (first run)...")
    build_and_save_index()


@app.get("/health")
def health():
    return {"status": "ok", "service": "data_service"}


@app.get("/health/ready")
def readiness():
    """
    Used by Docker Compose's healthcheck (Exercise 5) so that the RAG
    Service only starts once a vector index actually exists on the
    shared volume.
    """
    if VECTORS_PATH.exists() and CHUNKS_PATH.exists():
        return {"status": "ready"}
    raise HTTPException(status_code=503, detail="Vector index not built yet")


@app.get("/records")
def list_records():
    records = load_knowledge_base()
    return {"count": len(records), "records": records}


@app.get("/records/{record_id}")
def get_record(record_id: str):
    records = load_knowledge_base()
    for r in records:
        if r["id"] == record_id:
            return r
    raise HTTPException(status_code=404, detail=f"Record '{record_id}' not found")


@app.post("/rebuild-index")
def rebuild_index():
    """
    Re-runs: knowledge_base/data.json -> chunking -> embeddings -> vector storage.
    The RAG Service should be restarted (or reload its index) after this call
    so it picks up the refreshed vectors.
    """
    logger.info("Rebuilding vector index...")
    try:
        stats = build_and_save_index()
    except Exception as exc:
        logger.error("Failed to rebuild index: %s", exc)
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {exc}") from exc

    return {"status": "ok", **stats}
