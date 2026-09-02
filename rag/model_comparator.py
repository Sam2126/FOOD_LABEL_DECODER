"""
model_comparator.py
===================

Executes dynamic multi-model benchmarking and AI Referee evaluation
across all available local Ollama models for any food label.

Features:
  - Shared FAISS vector retrieval for fair ground-truth benchmarking
  - Real-time quantitative telemetry (Latency, Throughput, Grounding Accuracy, Allergen Recall, Hallucination Rate)
  - Composite multi-criteria scoring
  - AI Referee analysis & recommendation synthesis
"""

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from app.ollama_client import ask_ollama, list_models
from rag.retriever import retrieve_for_ingredient_list, retrieve_mapped_ingredients
from rag.rag_pipeline import format_context, format_mapped_context, build_rag_prompt, detect_allergens_and_dietary
from eval.metrics import compute_grounding_accuracy, compute_allergen_recall, compute_hallucination_rate


logger = logging.getLogger("food-label-decoder.model_comparator")

def build_fast_compare_prompt(ingredient_list: str, context: str, unmatched: list[str]) -> str:
    unmatched_str = ", ".join(unmatched) if unmatched else "None"
    return f"""[INST] <<SYS>>
You are an expert food scientist and consumer label analyst.
Explain ONLY the ingredients in the USER INGREDIENT LIST below.
Base your explanations strictly on the VERIFIED EVIDENCE provided. Do NOT invent ingredients not in the list.
<</SYS>>

USER INGREDIENT LIST:
{ingredient_list}

VERIFIED KNOWLEDGE BASE EVIDENCE:
{context}

UNMATCHED ITEMS (No entry in database):
{unmatched_str}

OUTPUT FORMAT:
For each ingredient in the user list, write a concise, informative bullet point:
• **[Ingredient Name]**: [Function in food]. [Allergen: Note if present or "None"]. [Health consideration].
[/INST]
"""



def evaluate_single_model(
    model_name: str,
    ingredient_list: str,
    context: str,
    unmatched: list[str],
    retrieved: list[dict],
    expected_allergens: list[str],
    temperature: float = 0.20,
    max_tokens: int = 400,
) -> dict:
    """
    Runs RAG decoding for one model and computes real-time evaluation metrics.
    """
    prompt = build_fast_compare_prompt(ingredient_list, context, unmatched)
    eff_tokens = max(max_tokens if max_tokens else 350, 380)
    start_time = time.perf_counter()

    try:
        explanation = ask_ollama(
            prompt=prompt,
            model=model_name,
            max_tokens=eff_tokens,
            temperature=temperature,
        )

        elapsed_sec = max(0.01, time.perf_counter() - start_time)
        latency_ms = int(elapsed_sec * 1000)


        # Approximate token count (words * 1.33)
        words = len(explanation.split())
        approx_tokens = int(words * 1.33)
        throughput = round(approx_tokens / elapsed_sec, 1)

        # Extract target ingredient tokens
        target_ings = [r["chunk"]["metadata"]["name"] for r in retrieved if "metadata" in r.get("chunk", {})]

        # Compute quantitative metrics
        grounding_score = compute_grounding_accuracy(
            response_text=explanation,
            target_ingredients=target_ings,
            ground_truth_context=context,
            retrieved_chunks=retrieved,
        )
        allergen_recall = compute_allergen_recall(
            response_text=explanation,
            expected_allergens=expected_allergens,
        )
        hallucination_rate = compute_hallucination_rate(
            response_text=explanation,
            unmatched_ingredients=unmatched,
            retrieved_chunks=retrieved,
        )

        # Speed score (normalized: <=1s is 1.0, >=15s is 0.1)
        speed_score = round(max(0.1, min(1.0, 3.0 / (elapsed_sec + 0.5))), 3)

        # Composite score (Weighted Quality + Performance)
        composite_score = round(
            0.35 * grounding_score
            + 0.30 * allergen_recall
            + 0.20 * (1.0 - hallucination_rate)
            + 0.15 * speed_score,
            3,
        )

        return {
            "model": model_name,
            "status": "success",
            "explanation": explanation,
            "latency_ms": latency_ms,
            "latency_sec": round(elapsed_sec, 2),
            "tokens_generated": approx_tokens,
            "throughput_tokens_sec": throughput,
            "grounding_score": round(grounding_score * 100, 1),
            "allergen_recall": round(allergen_recall * 100, 1),
            "hallucination_rate": round(hallucination_rate * 100, 1),
            "composite_score": round(composite_score * 100, 1),
            "speed_score": round(speed_score * 100, 1),
        }

    except Exception as exc:
        logger.error(f"Error evaluating model {model_name}: {exc}")
        return {
            "model": model_name,
            "status": "error",
            "explanation": f"⚠️ Error executing model '{model_name}': {str(exc)}",
            "latency_ms": 0,
            "latency_sec": 0,
            "tokens_generated": 0,
            "throughput_tokens_sec": 0,
            "grounding_score": 0.0,
            "allergen_recall": 0.0,
            "hallucination_rate": 100.0,
            "composite_score": 0.0,
            "speed_score": 0.0,
        }


def synthesize_referee_verdict(
    leaderboard: list[dict],
    ingredient_list: str,
    detected_allergens: list[str],
) -> dict:
    """
    Synthesizes an intelligent AI referee evaluation and crowns the best model.
    """
    if not leaderboard:
        return {
            "champion_model": "None",
            "verdict_title": "No Models Evaluated",
            "rationale": "No models were available for evaluation.",
            "pros_and_cons": {},
        }

    valid_models = [m for m in leaderboard if m["status"] == "success"]
    if not valid_models:
        return {
            "champion_model": leaderboard[0]["model"] if leaderboard else "None",
            "verdict_title": "Model Evaluation Incomplete",
            "referee_rationale": "Some models encountered timeouts or connection errors.",
            "pros_and_cons": {},
        }

    # Best model is top of leaderboard (sorted by composite score)
    winner = valid_models[0]
    winner_name = winner["model"]

    # Build smart pros/cons for each model
    pros_cons = {}
    for m in valid_models:
        p_list = []
        c_list = []

        if m["latency_sec"] <= 2.5:
            p_list.append(f"⚡ Ultra-low latency ({m['latency_sec']}s, {m['throughput_tokens_sec']} t/s)")
        elif m["latency_sec"] >= 8.0:
            c_list.append(f"⏱ Slower response time ({m['latency_sec']}s)")

        if m["grounding_score"] >= 80.0:
            p_list.append(f"🎯 High factual grounding ({m['grounding_score']}%)")
        else:
            c_list.append(f"📉 Lower knowledge base alignment ({m['grounding_score']}%)")

        if m["allergen_recall"] >= 90.0:
            p_list.append("🛡️ Complete allergen safety recall")
        else:
            c_list.append("⚠️ Missed or weakly highlighted allergen groups")

        if m["hallucination_rate"] <= 10.0:
            p_list.append("🔒 Minimal hallucination rate")
        else:
            c_list.append(f"⚠️ Higher potential hallucination ({m['hallucination_rate']}%)")

        pros_cons[m["model"]] = {
            "pros": p_list if p_list else ["Standard structured output"],
            "cons": c_list if c_list else ["None significant"],
        }

    # Generate rationale
    speed_diff = ""
    if len(valid_models) > 1:
        slowest = max(valid_models, key=lambda x: x["latency_sec"])
        if slowest["model"] != winner["model"] and slowest["latency_sec"] > winner["latency_sec"]:
            speedup = round(slowest["latency_sec"] / max(0.1, winner["latency_sec"]), 1)
            speed_diff = f" It delivered {speedup}x faster response times compared to {slowest['model']} while maintaining "

    rationale = (
        f"**{winner_name}** is crowned the **Overall Best Model** for this food label.{speed_diff}"
        f"achieving a **{winner['grounding_score']}% Grounding Accuracy** and **{winner['allergen_recall']}% Allergen Recall** "
        f"with an overall composite efficiency score of **{winner['composite_score']}/100**."
    )

    return {
        "champion_model": winner_name,
        "verdict_title": f"🏆 {winner_name} — Optimal Production Choice",
        "referee_rationale": rationale,
        "pros_and_cons": pros_cons,
    }



def compare_all_models(
    ingredient_list: str,
    models: Optional[List[str]] = None,
    temperature: float = 0.20,
    max_tokens: int = 450,
    top_k: int = 2,
    min_similarity: float = 0.30,
) -> dict:
    """
    Main entrypoint: executes multi-model benchmarking across all available models.
    """
    start_total = time.perf_counter()

    # 1. Determine models to evaluate
    if not models or len(models) == 0:
        available = list_models()
        # Prioritize lightweight and benchmark models (including StarCoder2)
        priority = ["llama3.2:1b", "starcoder2:3b", "codellama:latest", "llama3:latest"]
        models = [m for m in priority if m in available]
        for m in available:
            if m not in models:
                models.append(m)
        if not models:
            models = available if available else ["llama3.2:1b"]



    # 2. Shared FAISS Retrieval (Mapped 1:1 for ground truth context)
    mapped_items, retrieved, unmatched = retrieve_mapped_ingredients(
        ingredient_list,
        min_similarity=min_similarity,
    )
    context = format_mapped_context(mapped_items)


    # Format chunks
    formatted_chunks = [
        {
            "chunk_id": r["chunk"]["chunk_id"],
            "name": r["chunk"]["metadata"]["name"],
            "source": r["chunk"]["metadata"].get("source", ""),
            "score": round(r["score"], 4),
            "text": r["chunk"]["text"],
        }
        for r in retrieved
    ]

    # Detect allergens
    detected_allergens, dietary_flags, allergen_breakdown = detect_allergens_and_dietary(
        retrieved_chunks=formatted_chunks,
        ingredient_text=ingredient_list,
    )

    # 3. Execute Multi-Model Inference Sequentially (GPU VRAM stability)
    model_results = {}
    for m in models:
        res = evaluate_single_model(
            model_name=m,
            ingredient_list=ingredient_list,
            context=context,
            unmatched=unmatched,
            retrieved=retrieved,
            expected_allergens=detected_allergens,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        model_results[m] = res


    # 4. Compute Leaderboard Ranking
    leaderboard = sorted(
        model_results.values(),
        key=lambda x: x["composite_score"],
        reverse=True,
    )
    for rank, entry in enumerate(leaderboard, 1):
        entry["rank"] = rank

    # 5. AI Referee Synthesis
    referee = synthesize_referee_verdict(
        leaderboard=leaderboard,
        ingredient_list=ingredient_list,
        detected_allergens=detected_allergens,
    )

    total_duration_ms = int((time.perf_counter() - start_total) * 1000)

    return {
        "ingredient_list": ingredient_list,
        "models_evaluated": models,
        "total_evaluation_time_ms": total_duration_ms,
        "retrieved_chunks": formatted_chunks,
        "unmatched_ingredients": unmatched,
        "detected_allergens": detected_allergens,
        "dietary_flags": dietary_flags,
        "allergen_breakdown": allergen_breakdown,
        "results": model_results,
        "leaderboard": leaderboard,
        "champion_model": referee["champion_model"],
        "verdict_title": referee["verdict_title"],
        "referee_rationale": referee["referee_rationale"],
        "pros_and_cons": referee["pros_and_cons"],
    }
