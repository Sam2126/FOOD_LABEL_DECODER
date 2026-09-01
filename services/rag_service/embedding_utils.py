"""
embedding_utils.py (RAG Service)

Same embedding wrapper as the Data Service uses (see data_service for full
explanation). The RAG Service needs this to embed the incoming USER QUERY
at request time - the Data Service only embeds the knowledge base once,
offline/on-demand.
"""

import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger("rag_service.embedding_utils")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors, dtype="float32")
