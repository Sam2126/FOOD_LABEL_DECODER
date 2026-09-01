"""
embedding_utils.py (Data Service)
==================================

Small, self-contained wrapper around the sentence-transformers embedding
model. Deliberately duplicated (not imported) between data_service and
rag_service - in a microservice architecture each service should be
independently deployable, without importing another service's internal
code across a network/container boundary.
"""

import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger("data_service.embedding_utils")

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
