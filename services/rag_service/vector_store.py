"""
vector_store.py (RAG Service)

Same FAISS wrapper explained in Exercise 3, now reading from EMBEDDINGS_DIR
- a location written by the Data Service (shared volume) rather than a
local project folder.
"""

import json
import logging
import os
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("rag_service.vector_store")

DEFAULT_EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "shared_embeddings"
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", str(DEFAULT_EMBEDDINGS_DIR)))
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"


class VectorStore:
    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self._loaded = False

    def load(self) -> None:
        if not VECTORS_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Vector index not found at {EMBEDDINGS_DIR}. "
                "Call POST /rebuild-index on the Data Service first."
            )

        vectors = np.load(VECTORS_PATH).astype("float32")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        self._loaded = True
        logger.info("Loaded FAISS index: %d vectors, dim %d", vectors.shape[0], dimension)

    def reload(self) -> None:
        self._loaded = False
        self.load()

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        if not self._loaded:
            self.load()

        query_vector = query_vector.reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({"chunk": self.chunks[idx], "score": float(score)})
        return results


vector_store = VectorStore()
