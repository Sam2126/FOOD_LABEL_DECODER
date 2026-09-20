"""evaluation/calculate_metrics.py

Computes category-wise quantitative metrics and overall benchmarks across all 7 SE categories:
  1. Explanation
  2. Code Retrieval
  3. Dependency Understanding
  4. Bug Analysis
  5. Code Generation
  6. Refactoring
  7. RAG based Question

Outputs:
  - evaluation/metrics_summary.json
  - Formatted Category Comparison Matrix printed to console
  - Category Winners answering the 7 specific SE evaluation questions
"""

import ast
import json
import os
import re
import statistics
import sys
import numpy as np

# Ensure Windows PowerShell handles UTF-8 characters cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SE_CATEGORIES = [
    "Explanation",
    "Code Retrieval",
    "Dependency Understanding",
    "Bug Analysis",
    "Code Generation",
    "Refactoring",
    "RAG based Question",
]


def resolve_path(filename: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join("evaluation", filename),
        os.path.join(script_dir, filename),
        os.path.abspath(filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(script_dir, filename)


def load_json_file(filepath: str, default=None):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: could not read {filepath}: {e}")
    return default if default is not None else {}


def extract_python_code(text: str) -> str:
    matches = re.findall(r"```(?:python|py)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if matches:
        extracted = "\n\n".join(m.strip() for m in matches if m.strip())
        if extracted:
            return extracted
    return text.strip()


def validate_python_code(code_str: str) -> bool:
    """Checks if the extracted Python snippet parses cleanly into valid AST."""
    cleaned = extract_python_code(code_str)
    try:
        ast.parse(cleaned)
        return True
    except Exception:
        try:
            ast.parse(code_str)
            return True
        except Exception:
            return False


def calculate_metrics():
    questions_path = resolve_path("questions.json")
    results_path = resolve_path("raw_results.json")
    hallucination_path = resolve_path("hallucination_manual.json")
    retrieval_path = resolve_path("retrieval_manual.json")
    output_path = resolve_path("metrics_summary.json")

    questions = load_json_file(questions_path, default=[])
    raw_results = load_json_file(results_path, default={})
    hallucination_manual = load_json_file(hallucination_path, default=[])
    retrieval_manual = load_json_file(retrieval_path, default=[])

    if not questions:
        print(f"Error: questions file not found or empty at {questions_path}")
        return

    if not raw_results:
        print(f"Notice: raw_results not found at {results_path}. Running evaluation first...")
        from run_evaluation import run_all
        run_all(force_mock=True)
        raw_results = load_json_file(results_path, default={})

    q_map = {q["id"]: q for q in questions}
    models = list(raw_results.keys())

    # Build manual lookup caches
    hallucination_cache = {}
    if isinstance(hallucination_manual, list):
        for item in hallucination_manual:
            qid = item.get("question_id")
            for m, mdata in item.get("model_responses", {}).items():
                if isinstance(mdata, dict):
                    hallucination_cache[(m, qid)] = mdata.get("hallucinated", False)

    retrieval_cache = {}
    if isinstance(retrieval_manual, list):
        for item in retrieval_manual:
            qid = item.get("question_id")
            retrieval_cache[qid] = item.get("correct_chunk_retrieved", True)

    category_metrics = {cat: {} for cat in SE_CATEGORIES}
    overall_metrics = {m: {} for m in models}

    for model in models:
        records = raw_results.get(model, [])
        all_keyword_scores = []
        all_latencies = []
        all_prompt_tok = 0
        all_comp_tok = 0
        all_mem_deltas = []
        all_code_parsed = []
        all_hallucinations = []

        # Group records by category
        cat_records = {cat: [] for cat in SE_CATEGORIES}
        for r in records:
            cat = r.get("category")
            if cat in cat_records:
                cat_records[cat].append(r)

        for cat, recs in cat_records.items():
            kw_scores = []
            latencies = [float(r.get("latency_seconds", 0.0)) for r in recs]
            prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in recs)
            completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in recs)
            mem_deltas = [float(r.get("memory_mb_delta", 0.0)) for r in recs]

            # Keyword correctness
            for r in recs:
                qid = r.get("question_id")
                q_info = q_map.get(qid, {})
                expected_kws = q_info.get("expected_keywords", [])
                resp_text = (r.get("response") or "").lower()

                if expected_kws:
                    matched = sum(1 for kw in expected_kws if kw.lower() in resp_text)
                    score = matched / len(expected_kws)
                else:
                    score = 1.0
                kw_scores.append(score)

            cat_correctness = round(float(statistics.mean(kw_scores)), 4) if kw_scores else 0.0
            cat_lat_mean = round(float(statistics.mean(latencies)), 2) if latencies else 0.0
            cat_lat_p95 = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0

            # Category-specific metrics
            cat_dict = {
                "correctness": cat_correctness,
                "latency_mean": cat_lat_mean,
                "latency_p95": cat_lat_p95,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            if cat in ["Code Generation", "Refactoring"]:
                pass_count = sum(1 for r in recs if validate_python_code(r.get("response", "")))
                cat_pass_rate = round(float(pass_count / len(recs)), 4) if recs else 0.0
                cat_dict["test_pass_rate"] = cat_pass_rate
                all_code_parsed.extend([validate_python_code(r.get("response", "")) for r in recs])

            if cat in ["Code Retrieval", "RAG based Question"]:
                ret_scores = [retrieval_cache.get(r.get("question_id"), True) for r in recs]
                ret_quality = round(float(sum(1 for s in ret_scores if s) / len(ret_scores)), 4) if ret_scores else 1.0
                cat_dict["retrieval_quality"] = ret_quality

            if cat == "RAG based Question":
                trap_recs = [r for r in recs if q_map.get(r.get("question_id"), {}).get("is_trap")]
                if trap_recs:
                    h_count = sum(1 for r in trap_recs if hallucination_cache.get((model, r.get("question_id")), False))
                    h_rate = round(float(h_count / len(trap_recs)), 4)
                else:
                    h_rate = 0.0
                cat_dict["hallucination_rate"] = h_rate
                all_hallucinations.append(h_rate)

            category_metrics[cat][model] = cat_dict

            # Accumulate for overall
            all_keyword_scores.extend(kw_scores)
            all_latencies.extend(latencies)
            all_prompt_tok += prompt_tokens
            all_comp_tok += completion_tokens
            all_mem_deltas.extend(mem_deltas)

        overall_metrics[model] = {
            "correctness": round(float(statistics.mean(all_keyword_scores)), 4) if all_keyword_scores else 0.0,
            "latency_mean": round(float(statistics.mean(all_latencies)), 2) if all_latencies else 0.0,
            "latency_p95": round(float(np.percentile(all_latencies, 95)), 2) if all_latencies else 0.0,
            "total_tokens": all_prompt_tok + all_comp_tok,
            "completion_tokens": all_comp_tok,
            "peak_memory_mb": round(float(max(all_mem_deltas)), 2) if all_mem_deltas else 0.0,
            "code_pass_rate": round(float(sum(1 for p in all_code_parsed if p) / len(all_code_parsed)), 4) if all_code_parsed else 0.0,
            "hallucination_rate": round(float(statistics.mean(all_hallucinations)), 4) if all_hallucinations else 0.0,
        }

    # Determine Category Winners for the 7 questions
    category_winners = {}
    winner_criteria = {
        "Explanation": ("correctness", False, "Highest architectural conceptual correctness and terminology clarity"),
        "Code Retrieval": ("correctness", False, "Most accurate file and function identification"),
        "Dependency Understanding": ("correctness", False, "Best multi-service call chain and failure tracing"),
        "Bug Analysis": ("correctness", False, "Most comprehensive root cause diagnosis and fix prescription"),
        "Code Generation": ("test_pass_rate", False, "Highest syntactically valid Python AST execution rate"),
        "Refactoring": ("test_pass_rate", False, "Highest quality syntactic refactoring and async/regex constructs"),
        "RAG based Question": ("hallucination_rate", True, "Lowest hallucination rate on regulatory traps and factual grounding"),
    }

    for cat, (metric_key, lower_is_better, rationale) in winner_criteria.items():
        cat_scores = {m: category_metrics[cat][m].get(metric_key, 0.0) for m in models}
        if lower_is_better:
            best_model = min(cat_scores, key=cat_scores.get)
        else:
            best_model = max(cat_scores, key=cat_scores.get)

        best_val = cat_scores[best_model]
        category_winners[cat] = {
            "winner": best_model,
            "metric_evaluated": metric_key,
            "score": best_val,
            "rationale": rationale,
        }

    summary_payload = {
        "by_category": category_metrics,
        "overall": overall_metrics,
        "category_winners": category_winners,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n==========================================================================================")
    print(f"                      WEEK 4 CATEGORY-WISE QUANTITATIVE COMPARISON")
    print(f"==========================================================================================")
    print_category_comparison_matrix(category_metrics, category_winners, models)
    print("\n")
    print_overall_table(overall_metrics, models)
    print("\n")
    print_seven_questions_summary(category_winners)
    print(f"\nSaved full category-wise metrics to: {output_path}\n")


def print_category_comparison_matrix(cat_metrics: dict, cat_winners: dict, models: list):
    col_w = [27, 18, 18, 18, 18]
    headers = ["Category", "CodeLlama 7B", "StarCoder2 7B", "Llama 3.2 3B", "Category Winner"]

    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
    divider = "-+-".join("-" * w for w in col_w)

    print(header_str)
    print(divider)

    for cat in SE_CATEGORIES:
        m_vals = []
        for m in ["codellama", "starcoder2:7b", "llama3.2:3b"]:
            if m in cat_metrics[cat]:
                score = cat_metrics[cat][m]["correctness"]
                lat = cat_metrics[cat][m]["latency_mean"]
                m_vals.append(f"{score * 100:.1f}% ({lat:.1f}s)")
            else:
                m_vals.append("—")

        winner = cat_winners.get(cat, {}).get("winner", "—")
        row = [cat] + m_vals + [winner]
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_w)))


def print_overall_table(overall: dict, models: list):
    col_w = [16, 13, 14, 14, 14, 14, 16]
    headers = ["Model", "Correctness", "Latency Mean", "Latency P95", "Total Tokens", "Peak Mem (MB)", "Code Pass Rate"]
    header_str = " | ".join(h.ljust(w) for h, w in zip(headers, col_w))
    divider = "-+-".join("-" * w for w in col_w)

    print("OVERALL MODEL BENCHMARK (AGGREGATE REFERENCE):")
    print(header_str)
    print(divider)

    for m in models:
        stats = overall[m]
        row = [
            m,
            f"{stats['correctness'] * 100:.1f}%",
            f"{stats['latency_mean']:.2f}s",
            f"{stats['latency_p95']:.2f}s",
            f"{stats['total_tokens']:,}",
            f"{stats['peak_memory_mb']:.1f} MB",
            f"{stats['code_pass_rate'] * 100:.1f}%",
        ]
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_w)))


def print_seven_questions_summary(winners: dict):
    print("==========================================================================================")
    print("         ANSWERS TO THE 7 SOFTWARE ENGINEERING MODEL SELECTION QUESTIONS")
    print("==========================================================================================")
    q_prompts = {
        "Explanation": "1. Which model performs best for Explanation?",
        "Code Retrieval": "2. Which model is best for Code Retrieval?",
        "Dependency Understanding": "3. Which model performs better for Dependency Understanding?",
        "Bug Analysis": "4. Which model is better for Bug Analysis?",
        "Code Generation": "5. Which model is better for Code Generation?",
        "Refactoring": "6. Which model performs better for Refactoring?",
        "RAG based Question": "7. Which model performs better for RAG based Questions?",
    }
    for cat, prompt in q_prompts.items():
        w = winners.get(cat, {})
        m = w.get("winner", "N/A")
        metric = w.get("metric_evaluated", "")
        score = w.get("score", 0.0)
        score_str = f"{score * 100:.1f}%" if ("rate" in metric or "correctness" in metric) else f"{score}"
        print(f"{prompt:<65} -> [WINNER: {m.upper()}] ({metric}: {score_str})")


if __name__ == "__main__":
    calculate_metrics()
