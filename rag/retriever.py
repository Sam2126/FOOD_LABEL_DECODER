"""
retriever.py
============

Implements the RETRIEVAL half of RAG:
    User Question -> Query Embedding -> Vector Similarity -> Relevant Information -> Context

Transforms ingredient text into semantically relevant knowledge-base chunks.
"""

import re
import logging
from typing import Optional, Union

from ingestion.embedder import embed_texts
from rag.vector_store import vector_store

logger = logging.getLogger("food-label-decoder.retriever")

DEFAULT_TOP_K = 4
MIN_SIMILARITY_THRESHOLD = 0.30


def parse_balanced_tokens(text: str) -> list[str]:
    """Splits text by separators (, ;\n•·*) while preserving nested parentheses."""
    tokens = []
    current = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch in ",;\n•·*\t\r" and depth == 0:
            tok = "".join(current).strip()
            if tok:
                tokens.append(tok)
            current = []
        else:
            current.append(ch)
    if current:
        tok = "".join(current).strip()
        if tok:
            tokens.append(tok)
    return tokens


def parse_ingredients(raw_text: str) -> list[str]:
    """
    Intelligently splits ingredient lists:
      - Respects nested parentheses (e.g., 'Edible Vegetable Oil (Palmolein, Rice Bran Oil)')
      - Expands multiple INS/E-numbers (e.g., 'Acidity Regulators (330, 296, 334)' -> 'Acidity Regulators 330', etc.)
      - Extracts sub-ingredients from compound seasoning blends
      - Extracts items from 'Contains Onion and Garlic'
    """
    if not raw_text or not raw_text.strip():
        return []

    results = []
    outer_tokens = parse_balanced_tokens(raw_text.strip())

    for token in outer_tokens:
        clean_tok = re.sub(r"^[~•·\*\-\d\.\s]+", "", token).strip()
        if not clean_tok or len(clean_tok) < 2:
            continue

        # Match category with parentheses: Name (inner)
        m = re.match(r"^(.*?)\s*[\(\[](.*)[\)\]]\s*$", clean_tok)
        if m:
            category_name = m.group(1).strip()
            inner_content = m.group(2).strip()
            inner_tokens = parse_balanced_tokens(inner_content)

            # Case A: Inside parentheses are numbers e.g. Acidity Regulators (330, 296, 334)
            if all(re.match(r"^[0-9a-zA-Z\s]+$", t) and re.search(r"\d+", t) for t in inner_tokens):
                for num in inner_tokens:
                    results.append(f"{category_name} {num.strip()}".strip())
            # Case B: Inside parentheses are multiple sub-ingredients e.g. (Palmolein, Rice Bran Oil)
            elif len(inner_tokens) > 1:
                if category_name and len(category_name) >= 3:
                    results.append(category_name)
                for sub in inner_tokens:
                    sub_clean = re.sub(r"^[~•·\*\-\s]+", "", sub).strip()
                    if sub_clean:
                        # Nested sub-check
                        sub_m = re.match(r"^(.*?)\s*[\(\[](.*)[\)\]]\s*$", sub_clean)
                        if sub_m:
                            sub_cat = sub_m.group(1).strip()
                            sub_in = sub_m.group(2).strip()
                            sub_in_toks = parse_balanced_tokens(sub_in)
                            if all(re.match(r"^[0-9a-zA-Z\s]+$", t) and re.search(r"\d+", t) for t in sub_in_toks):
                                for num in sub_in_toks:
                                    results.append(f"{sub_cat} {num.strip()}".strip())
                            else:
                                results.append(sub_clean)
                        else:
                            results.append(sub_clean)
            else:
                results.append(clean_tok)
        else:
            # Check for 'Contains X and Y'
            if clean_tok.lower().startswith("contains "):
                after = clean_tok[9:].strip()
                for sub in re.split(r"\s+(?:and|&)\s+|,", after):
                    if sub.strip():
                        results.append(sub.strip())
            else:
                results.append(clean_tok)

    # Deduplicate while preserving order
    deduped = list(dict.fromkeys(results))
    return deduped if deduped else [raw_text.strip()]



def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = MIN_SIMILARITY_THRESHOLD,
) -> list[dict]:
    """
    Runs the retrieval pipeline for a single query string and
    returns the relevant chunks scoring above min_similarity.
    """
    if not query or not query.strip():
        return []

    query_vector = embed_texts([query.strip()])[0]
    results = vector_store.search(query_vector, top_k=top_k)

    filtered = [r for r in results if r["score"] >= min_similarity]
    logger.info(
        "Query '%s' -> %d results (%d >= %.2f)",
        query, len(results), len(filtered), min_similarity,
    )
    return filtered


def retrieve_for_ingredient_list(
    ingredient_list: str,
    top_k_per_ingredient: int = 2,
    min_similarity: float = MIN_SIMILARITY_THRESHOLD,
    include_unmatched: bool = False,
) -> Union[list[dict], tuple[list[dict], list[str]]]:
    """
    Extracts individual ingredients, embeds each independently,
    performs FAISS vector similarity search, and aggregates matched chunks.
    """
    ingredients = parse_ingredients(ingredient_list)

    seen_chunk_ids = set()
    combined_results = []
    unmatched_ingredients = []

    for ingredient in ingredients:
        results = retrieve(ingredient, top_k=top_k_per_ingredient, min_similarity=min_similarity)
        if not results:
            unmatched_ingredients.append(ingredient)
        for r in results:
            chunk_id = r["chunk"]["chunk_id"]
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                combined_results.append(r)

    # Highest similarity first, across all ingredients
    combined_results.sort(key=lambda r: r["score"], reverse=True)
    
    if include_unmatched:
        return combined_results, unmatched_ingredients
    return combined_results

