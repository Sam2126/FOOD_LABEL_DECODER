"""
embedder.py (Data Service)

Builds the vector index and writes it to EMBEDDINGS_DIR, a location shared
with the RAG Service.

- Locally (no Docker): EMBEDDINGS_DIR defaults to services/shared_embeddings/
  on disk, which both data_service and rag_service can read since they run
  on the same machine.
- In Docker (Exercise 5): EMBEDDINGS_DIR is a shared named volume mounted
  into both the data_service and rag_service containers.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from ingestion.chunker import build_chunks
from ingestion.embedding_utils import embed_texts

logger = logging.getLogger("data_service.embedder")

DEFAULT_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent.parent / "shared_embeddings"
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", str(DEFAULT_EMBEDDINGS_DIR)))
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"


def build_and_save_index() -> dict:
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = build_chunks()
    texts = [c["text"] for c in chunks]

    logger.info("Embedding %d chunks...", len(texts))
    vectors = embed_texts(texts)

    np.save(VECTORS_PATH, vectors)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info(
        "Saved %d vectors (dim=%d) to %s", vectors.shape[0], vectors.shape[1], VECTORS_PATH
    )
    return {"num_chunks": len(chunks), "vector_dimension": int(vectors.shape[1])}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_and_save_index()
