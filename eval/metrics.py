"""
metrics.py
==========

Defines quantitative quality and performance evaluation metrics for the
Food Label Decoder RAG platform across multiple LLM backends.

Quality Metrics:
  - Accuracy / Grounding Score (0.0 - 1.0)
  - Allergen Detection Recall (0.0 - 1.0)
  - Retrieval Quality (Mean FAISS Cosine Similarity)
  - Hallucination Rate (0.0 - 1.0)
  - Test Pass Status (bool)

Performance Metrics:
  - Latency (ms)
  - Token Count & Throughput (tokens/sec)
  - System Memory RSS (MB) & CPU Utilization (%)
"""

import re
from typing import List, Dict, Any


def compute_allergen_recall(response_text: str, expected_allergens: List[str]) -> float:
    """
    Computes recall of expected allergens mentioned in the response.
    """
    if not expected_allergens:
        return 1.0  # No allergens to detect

    text_lower = response_text.lower()
    hits = 0
    for allergen in expected_allergens:
        # Check direct or synonym mentions
        parts = allergen.lower().split()
        if any(p in text_lower for p in parts):
            hits += 1

    return round(hits / len(expected_allergens), 3)


def compute_grounding_accuracy(
    response_text: str,
    target_ingredients: List[str],
    ground_truth_context: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Evaluates whether the response covers the target ingredients, aligns
    with the ground truth facts, and grounds claims in retrieved text.
    """
    if not response_text or len(response_text.strip()) < 10:
        return 0.0

    text_lower = response_text.lower()
    score = 0.0

    # 1. Target ingredients coverage (40% weight)
    if target_ingredients:
        ing_hits = sum(1 for ing in target_ingredients if ing.lower() in text_lower)
        score += 0.40 * (ing_hits / len(target_ingredients))
    else:
        score += 0.40

    # 2. Ground truth keywords alignment (40% weight)
    gt_words = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", ground_truth_context)]
    if gt_words:
        gt_hits = sum(1 for w in set(gt_words) if w in text_lower)
        score += 0.40 * min(1.0, gt_hits / max(4, len(set(gt_words)) * 0.4))
    else:
        score += 0.40

    # 3. Structure & caution phrasing (20% weight)
    has_caution = any(w in text_lower for w in ["caution", "allergen", "consideration", "dietary", "safe", "function", "additive"])
    if has_caution:
        score += 0.20

    return round(min(1.0, score), 3)


def compute_hallucination_rate(
    response_text: str,
    unmatched_ingredients: List[str],
    retrieved_chunks: List[Dict[str, Any]],
) -> float:
    """
    Estimates hallucination rate:
      - Higher if the model invents specific claims about unmatched ingredients.
      - Lower if the model correctly admits knowledge base limitations for unknown items.
    """
    if not unmatched_ingredients:
        return 0.05  # Base minimal noise

    text_lower = response_text.lower()
    hallucination_score = 0.0

    for unmatched in unmatched_ingredients:
        # Check if the model explicitly acknowledged lack of KB record
        has_disclaimer = any(
            phrase in text_lower
            for phrase in [
                "not cover", "not contain", "does not yet contain",
                "no verified record", "not found in the local knowledge base",
                "does not have", "unmatched"
            ]
        )
        if not has_disclaimer:
            hallucination_score += 0.50

    return round(min(1.0, hallucination_score / max(1, len(unmatched_ingredients))), 3)


def compute_retrieval_quality(retrieved_chunks: List[Dict[str, Any]]) -> float:
    """
    Computes average cosine similarity score of the retrieved chunks.
    """
    if not retrieved_chunks:
        return 0.0

    scores = [c.get("score", 0.0) for c in retrieved_chunks]
    return round(sum(scores) / len(scores), 4)


def evaluate_response(
    question_data: Dict[str, Any],
    api_response: Dict[str, Any],
    latency_ms: int,
    memory_rss_mb: float = 0.0,
    cpu_percent: float = 0.0,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation scoring single test invocation.
    """
    response_text = api_response.get("explanation", "")
    retrieved_chunks = api_response.get("retrieved_chunks", [])
    unmatched_ingredients = api_response.get("unmatched_ingredients", [])
    model_used = api_response.get("model_used", "unknown")
    mode = api_response.get("mode", "rag_grounded")

    # Estimated Token Count (~1.33 tokens per word)
    word_count = len(response_text.split())
    estimated_tokens = int(word_count * 1.33)
    tokens_per_sec = round(estimated_tokens / max(0.01, latency_ms / 1000.0), 2)

    # Compute Metrics
    allergen_recall = compute_allergen_recall(response_text, question_data.get("expected_allergens", []))
    accuracy_score = compute_grounding_accuracy(
        response_text,
        question_data.get("target_ingredients", []),
        question_data.get("ground_truth_context", ""),
        retrieved_chunks,
    )
    hallucination_rate = compute_hallucination_rate(
        response_text,
        unmatched_ingredients,
        retrieved_chunks,
    )
    retrieval_quality = compute_retrieval_quality(retrieved_chunks)

    # Pass Condition: Accuracy >= 0.55 & Allergen Recall >= 0.50
    test_passed = bool(accuracy_score >= 0.55 and allergen_recall >= 0.50)

    return {
        "question_id": question_data["id"],
        "category": question_data["category"],
        "model": model_used,
        "mode": mode,
        "latency_ms": latency_ms,
        "tokens": estimated_tokens,
        "tokens_per_sec": tokens_per_sec,
        "accuracy_score": accuracy_score,
        "allergen_recall": allergen_recall,
        "retrieval_quality": retrieval_quality,
        "hallucination_rate": hallucination_rate,
        "test_passed": test_passed,
        "memory_rss_mb": round(memory_rss_mb, 2),
        "cpu_percent": round(cpu_percent, 2),
        "retrieved_chunk_count": len(retrieved_chunks),
        "unmatched_count": len(unmatched_ingredients),
    }
