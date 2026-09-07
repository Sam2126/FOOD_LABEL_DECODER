"""RAG Analysis Script — evaluation/rag_analysis.py

Runs a focused RAG vs No-RAG comparison on 6 selected questions (ids: 14, 15, 16, 20, 24, 25).
For each question:
  1. Retrieves top-3 context chunks from retrieval_service
  2. Queries codellama WITH context
  3. Queries codellama WITHOUT context
  4. Saves full results to evaluation/rag_analysis_results.json

After running, manually fill in the "analysis" fields in the output JSON.
"""

import json
import os
import time
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8001/retrieve")
OLLAMA_URL    = os.environ.get("OLLAMA_URL",    "http://localhost:11434/api/generate")
OLLAMA_MODEL  = os.environ.get("OLLAMA_MODEL",  "codellama")

RAG_QUESTION_IDS = [14, 15, 16, 20, 24, 25]

CASE_TYPE_OPTIONS = [
    "good_retrieval_correct_response",
    "bad_retrieval_hallucination",
    "good_retrieval_still_hallucinated",
    "no_retrieval_confident_wrong",
]

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
QUESTIONS_PATH = SCRIPT_DIR / "questions.json"
OUTPUT_PATH    = SCRIPT_DIR / "rag_analysis_results.json"


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_questions() -> list:
    """Load and filter questions by RAG_QUESTION_IDS."""
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
    selected = [q for q in all_questions if q["id"] in RAG_QUESTION_IDS]
    # Preserve order of RAG_QUESTION_IDS
    id_order = {qid: i for i, qid in enumerate(RAG_QUESTION_IDS)}
    selected.sort(key=lambda q: id_order.get(q["id"], 999))
    return selected


def retrieve_context(question: str) -> dict:
    """Call retrieval service and return chunks + similarity scores."""
    try:
        resp = requests.post(
            RETRIEVAL_URL,
            json={"query": question, "collection": "both", "top_k": 3},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            return {
                "chunks": [r.get("text", "") for r in results],
                "similarity_scores": [r.get("similarity_score", 0.0) for r in results],
                "top_context": data.get("top_context", ""),
            }
    except requests.exceptions.ConnectionError:
        print(f"  ⚠️  Retrieval service unreachable at {RETRIEVAL_URL}")
    except Exception as e:
        print(f"  ⚠️  Retrieval error: {e}")
    return {"chunks": [], "similarity_scores": [], "top_context": ""}


def query_ollama(question: str, context: str = "") -> dict:
    """Query Ollama codellama with or without RAG context. Returns response + latency."""
    if context:
        prompt = f"""You are a food safety expert. Use the regulatory context below as your source of truth.

Regulatory context:
{context}

Question: {question}

Answer clearly and concisely. If the context does not contain the answer, say so explicitly — do NOT invent facts."""
    else:
        prompt = f"""You are a food safety expert.

Question: {question}

Answer clearly and concisely."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    start = time.time()
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        latency = round(time.time() - start, 2)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "response": data.get("response", "").strip(),
                "latency_seconds": latency,
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
            }
        else:
            return {"response": f"ERROR {resp.status_code}: {resp.text}", "latency_seconds": latency}
    except requests.exceptions.Timeout:
        return {"response": "ERROR: Ollama timed out after 120s", "latency_seconds": 120.0}
    except requests.exceptions.ConnectionError:
        return {"response": f"ERROR: Ollama unreachable at {OLLAMA_URL}", "latency_seconds": 0.0}
    except Exception as e:
        return {"response": f"ERROR: {e}", "latency_seconds": 0.0}


def build_empty_analysis() -> dict:
    """Return the empty analysis block — to be filled in manually."""
    return {
        "retrieval_relevant": None,        # bool: were the retrieved chunks relevant?
        "rag_response_correct": None,      # bool: was the WITH-RAG response accurate?
        "no_rag_response_correct": None,   # bool: was the WITHOUT-RAG response accurate?
        "hallucination_without_rag": None, # bool: did the model hallucinate without context?
        "hallucination_with_rag": None,    # bool: did the model hallucinate even with context?
        "case_type": None,                 # str: one of CASE_TYPE_OPTIONS
    }


def print_summary_table(results: list):
    """Print a readable summary table to console."""
    col_widths = [4, 42, 18, 8, 8]
    headers = ["ID", "Question", "Case Type", "W/ RAG", "No RAG"]

    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    divider    = "-+-".join("-" * w for w in col_widths)

    print("\n" + "=" * len(header_row))
    print("  RAG ANALYSIS SUMMARY")
    print("=" * len(header_row))
    print(header_row)
    print(divider)

    for r in results:
        qid      = str(r["question_id"]).ljust(col_widths[0])
        question = r["question"][:40].ljust(col_widths[1])
        case     = (r["analysis"].get("case_type") or "— not filled —")[:18].ljust(col_widths[2])
        rag_ok   = _bool_icon(r["analysis"].get("rag_response_correct")).ljust(col_widths[3])
        norag_ok = _bool_icon(r["analysis"].get("no_rag_response_correct")).ljust(col_widths[4])
        print(f"{qid} | {question} | {case} | {rag_ok} | {norag_ok}")

    print("=" * len(header_row))

    print("\n  Case type options for manual annotation:")
    for i, opt in enumerate(CASE_TYPE_OPTIONS, 1):
        print(f"    {i}. {opt}")
    print()


def _bool_icon(val) -> str:
    if val is True:
        return "✅"
    if val is False:
        return "❌"
    return "—"


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print("  FOOD LABEL DECODER — RAG ANALYSIS")
    print(f"  Model  : {OLLAMA_MODEL}")
    print(f"  Questions : {RAG_QUESTION_IDS}")
    print(f"{'='*60}\n")

    questions = load_questions()
    if not questions:
        print("❌ No questions found. Check that evaluation/questions.json exists.")
        return

    results = []

    for q in questions:
        qid      = q["id"]
        question = q["question"]
        print(f"── Q{qid}: {question[:60]}…")

        # Step 1: Retrieve context
        print(f"   [1/3] Retrieving context…")
        retrieval = retrieve_context(question)
        chunks    = retrieval["chunks"]
        scores    = retrieval["similarity_scores"]
        context   = retrieval["top_context"]
        print(f"         → {len(chunks)} chunks retrieved "
              f"(scores: {[round(s,3) for s in scores]})")

        # Step 2: Query WITH context
        print(f"   [2/3] Querying {OLLAMA_MODEL} WITH context…")
        with_rag = query_ollama(question, context=context)
        print(f"         → {with_rag['latency_seconds']}s  "
              f"| {with_rag.get('completion_tokens', '?')} tokens")

        # Step 3: Query WITHOUT context
        print(f"   [3/3] Querying {OLLAMA_MODEL} WITHOUT context…")
        without_rag = query_ollama(question, context="")
        print(f"         → {without_rag['latency_seconds']}s  "
              f"| {without_rag.get('completion_tokens', '?')} tokens")

        results.append({
            "question_id":          qid,
            "question":             question,
            "category":             q.get("category", ""),
            "retrieved_chunks":     chunks,
            "similarity_scores":    scores,
            "response_with_rag":    with_rag["response"],
            "latency_with_rag_s":   with_rag["latency_seconds"],
            "response_without_rag": without_rag["response"],
            "latency_no_rag_s":     without_rag["latency_seconds"],
            "analysis":             build_empty_analysis(),
        })

        print()

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to: {OUTPUT_PATH}")
    print(f"   → Now open the file and fill in the 'analysis' fields manually.\n")

    # Print summary table
    print_summary_table(results)


if __name__ == "__main__":
    main()
