"""
embedder.py
===========

Converts text chunks into EMBEDDINGS (numeric vectors) and saves them
to disk, along with the chunk metadata they correspond to.

What is an embedding?
----------------------
An embedding is a fixed-length list of numbers (a vector) produced by a
neural network, such that pieces of text with SIMILAR MEANING end up with
vectors that are close together in that numeric space, and unrelated text
ends up far apart. It's a mathematical representation of meaning, not of
exact words.

Why do we need embeddings here?
---------------------------------
A user might type "MSG" while our knowledge base says "Monosodium
glutamate (E621)". Plain keyword matching would miss that connection.
Embeddings capture SEMANTIC similarity, so "MSG" and "Monosodium
glutamate" end up close together in vector space even though the exact
characters don't match - this is what makes semantic search possible in
Exercise 3.

Model used: sentence-transformers/all-MiniLM-L6-v2
-----------------------------------------------------
- Small (~80MB), fast, runs fully locally/offline (no API key needed)
- Produces 384-dimensional vectors
- Well-suited for short texts like our knowledge-base chunks
"""

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from ingestion.chunker import build_chunks

logger = logging.getLogger("food-label-decoder.embedder")

EMBEDDINGS_DIR = Path(__file__).resolve().parent.parent / "embeddings"
VECTORS_PATH = EMBEDDINGS_DIR / "vectors.npy"
CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None  # lazy-loaded singleton so we don't reload the model repeatedly


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Knowledge Chunk -> Embedding Model -> Vector

    Returns an (N, 384) numpy array, one row per input text.
    """
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors, dtype="float32")


def build_and_save_index() -> None:
    """
    Full Exercise 2 pipeline:
        Documents -> Chunking -> Embeddings -> Vector Representation (saved to disk)
    """
    EMBEDDINGS_DIR.mkdir(exist_ok=True)

    chunks = build_chunks()
    texts = [c["text"] for c in chunks]

    logger.info("Embedding %d chunks...", len(texts))
    vectors = embed_texts(texts)

    np.save(VECTORS_PATH, vectors)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    logger.info(
        "Saved %d vectors of dimension %d to %s", vectors.shape[0], vectors.shape[1], VECTORS_PATH
    )
    logger.info("Saved chunk metadata to %s", CHUNKS_PATH)


if __name__ == "__main__":
    # Run this to (re)build the vector store:
    #   python -m ingestion.embedder
    logging.basicConfig(level=logging.INFO)
    build_and_save_index()
