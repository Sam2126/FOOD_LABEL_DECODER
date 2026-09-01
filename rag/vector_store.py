"""
vector_store.py
================

Wraps FAISS (Facebook AI Similarity Search) - a local, in-process vector
database. It loads the vectors + chunk metadata that Exercise 2 generated
(embeddings/vectors.npy, embeddings/chunks.json) and provides fast
similarity search over them.

What is a vector database?
-----------------------------
A regular database finds rows by exact/keyword match ("WHERE name = 'MSG'").
A vector database finds rows by NUMERIC CLOSENESS between vectors - i.e. by
meaning, not exact text. Under the hood it still stores vectors + metadata,
but the query mechanism is "find the K vectors nearest to this one" instead
of "find rows matching this filter".

What is vector similarity / cosine similarity?
--------------------------------------------------
Cosine similarity measures the angle between two vectors, ignoring their
length - a value from -1 (opposite meaning) to 1 (identical meaning).
Because our embeddings are L2-normalized (see ingestion/embedder.py,
normalize_embeddings=True), the INNER PRODUCT of two vectors is
mathematically equivalent to their cosine similarity. That's why we use
faiss.IndexFlatIP (Inner Product) below instead of a raw L2-distance index.

Why FAISS specifically?
--------------------------
- Runs fully in-process, no separate server to install/manage
- Extremely fast for small-to-medium datasets (our 27 chunks are trivial for it)
- Simple, well-documented, ideal for a college project
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger("food-label-decoder.vector_store")

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"


class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP + the chunk metadata that
    corresponds to each vector row.
    """

    def __init__(self):
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self._loaded = False

    def load(self) -> None:
        if not VECTORS_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                "Vector index not found. Run `python -m ingestion.embedder` first "
                "(Exercise 2) to generate embeddings/vectors.npy and embeddings/chunks.json."
            )

        vectors = np.load(VECTORS_PATH).astype("float32")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)
        self._loaded = True

        logger.info(
            "Loaded FAISS index: %d vectors, dimension %d", vectors.shape[0], dimension
        )

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        """
        Query Embedding -> Vector Similarity -> Relevant Information

        Returns a list of dicts: {"chunk": <chunk dict>, "score": float}
        sorted by descending similarity.
        """
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


# Module-level singleton, shared across the app
vector_store = VectorStore()
