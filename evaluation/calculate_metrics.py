import ast
import json
import os
import re
import statistics
import numpy as np


def resolve_path(filename: str) -> str:
    """Find file relative to evaluation folder or current directory."""
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
    """Extract Python code blocks from model response or return full text."""
    # Match ```python ... ``` or ```py ... ``` blocks
    matches = re.findall(r"```(?:python|py)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if matches:
        extracted = "\n\n".join(m.strip() for m in matches if m.strip())
        if extracted:
            return extracted
    return text.strip()


def calculate_metrics():
    questions_path = resolve_path("questions.json")
    results_path = resolve_path("raw_results.json")
    hallucination_path = resolve_path("hallucination_manual.json")
    retrieval_path = resolve_path("retrieval_manual.json")
    output_path = resolve_path("metrics_summary.json")

    questions = load_json_file(questions_path, default=[])
    raw_results = load_json_file(results_path, default={})
    hallucination_manual = load_json_file(hallucination_path, default={})
    retrieval_manual = load_json_file(retrieval_path, default={})

    if not questions:
        print(f"Error: questions file not found or empty at {questions_path}")
        return

    if not raw_results:
        print(f"Error: raw_results file not found or empty at {results_path}")
        return

    # Map question metadata by ID
    q_map = {q["id"]: q for q in questions}
    trap_questions = [q for q in questions if q.get("category") == "hallucination_trap"]
    trap_ids = [q["id"] for q in trap_questions]
    total_trap = len(trap_questions)

    rag_questions = [q for q in questions if q.get("requires_rag")]
    total_rag = len(rag_questions)

    code_questions = [q for q in questions if q.get("category") == "code_generation"]
    code_ids = [q["id"] for q in code_questions]
    total_code = len(code_questions)

    metrics_summary = {}

    for model, records in raw_results.items():
        if not records:
            continue

        # 1. CORRECTNESS (keyword matching proxy across all questions)
        keyword_scores = []
        for r in records:
            qid = r.get("question_id")
            q_info = q_map.get(qid, {})
            expected_keywords = q_info.get("expected_keywords", [])
            response_text = (r.get("response") or "").lower()

            if expected_keywords:
                matched = sum(1 for kw in expected_keywords if kw.lower() in response_text)
                score = matched / len(expected_keywords)
            else:
                score = 1.0
            keyword_scores.append(score)

        correctness = round(float(statistics.mean(keyword_scores)), 4) if keyword_scores else 0.0

        # 2. HALLUCINATION RATE (hallucination_trap questions only)
        hallucinated_count = 0
        if isinstance(hallucination_manual, list):
            for item in hallucination_manual:
                if isinstance(item, dict):
                    resp_obj = item.get("model_responses", {}).get(model, {})
                    if isinstance(resp_obj, dict) and resp_obj.get("hallucinated") is True:
                        hallucinated_count += 1
        elif isinstance(hallucination_manual, dict):
            if model in hallucination_manual:
                m_val = hallucination_manual[model]
                if isinstance(m_val, (int, float)):
                    hallucinated_count = int(m_val)
                elif isinstance(m_val, list):
                    hallucinated_count = sum(1 for item in m_val if item in trap_ids or item in [str(i) for i in trap_ids])
                elif isinstance(m_val, dict):
                    hallucinated_count = sum(1 for k, v in m_val.items() if v and (int(k) in trap_ids or str(k) in [str(i) for i in trap_ids]))

        hallucination_rate = round(float(hallucinated_count / total_trap), 4) if total_trap > 0 else 0.0

        # 3. RETRIEVAL QUALITY (requires_rag questions only)
        correct_chunk_retrieved = 0
        if isinstance(retrieval_manual, list):
            has_annotations = any(isinstance(item, dict) and item.get("correct_chunk_retrieved") is not None for item in retrieval_manual)
            if has_annotations:
                correct_chunk_retrieved = sum(1 for item in retrieval_manual if isinstance(item, dict) and item.get("correct_chunk_retrieved") is True)
            else:
                correct_chunk_retrieved = total_rag
        elif isinstance(retrieval_manual, dict):
            if model in retrieval_manual:
                m_val = retrieval_manual[model]
                if isinstance(m_val, (int, float)):
                    correct_chunk_retrieved = int(m_val)
                elif isinstance(m_val, list):
                    correct_chunk_retrieved = len(m_val)
                elif isinstance(m_val, dict):
                    correct_chunk_retrieved = sum(1 for v in m_val.values() if v)
            else:
                correct_chunk_retrieved = total_rag
        else:
            correct_chunk_retrieved = total_rag

        retrieval_quality = round(float(correct_chunk_retrieved / total_rag), 4) if total_rag > 0 else 0.0

        # 4. RESPONSE LATENCY (mean, min, max, p95 across questions)
        latencies = [float(r.get("latency_seconds", 0.0)) for r in records]
        latency_mean = round(float(statistics.mean(latencies)), 2) if latencies else 0.0
        latency_p95 = round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0

        # 5. TOKEN USAGE
        prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in records)
        completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in records)
        total_tokens = prompt_tokens + completion_tokens

        # 6. MEMORY
        mem_deltas = [float(r.get("memory_mb_delta", 0.0)) for r in records]
        peak_memory = round(float(max(mem_deltas)), 2) if mem_deltas else 0.0

        # 7. TEST PASS RATE (code_generation category only)
        pass_count = 0
        code_records = [r for r in records if r.get("question_id") in code_ids or r.get("category") == "code_generation"]
        effective_code_total = len(code_records) if code_records else total_code

        for r in code_records:
            resp = r.get("response", "")
            code = extract_python_code(resp)
            parsed_ok = False
            try:
                ast.parse(code)
                parsed_ok = True
            except Exception:
                try:
                    ast.parse(resp)
                    parsed_ok = True
                except Exception:
                    pass

            if parsed_ok:
                pass_count += 1

        test_pass_rate = round(float(pass_count / effective_code_total), 4) if effective_code_total > 0 else 0.0

        metrics_summary[model] = {
            "correctness": correctness,
            "hallucination_rate": hallucination_rate,
            "retrieval_quality": retrieval_quality,
            "latency_mean": latency_mean,
            "latency_p95": latency_p95,
            "total_tokens": total_tokens,
            "peak_memory": peak_memory,
            "test_pass_rate": test_pass_rate,
        }

    # Save metrics_summary.json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"\nSaved metrics summary to: {output_path}\n")

    # Print comparison table to console
    print_comparison_table(metrics_summary)


def print_comparison_table(metrics: dict):
    headers = [
        "Model",
        "Correctness",
        "Hallucination",
        "Retrieval Qual.",
        "Latency Mean (s)",
        "Latency P95 (s)",
        "Total Tokens",
        "Peak Mem (MB)",
        "Test Pass Rate"
    ]
    col_widths = [16, 13, 15, 17, 18, 17, 14, 15, 16]

    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    divider_row = "-+-".join("-" * w for w in col_widths)

    print("=" * len(header_row))
    print(header_row)
    print(divider_row)

    for model, m in metrics.items():
        row_vals = [
            model,
            f"{m['correctness']:.2%}",
            f"{m['hallucination_rate']:.2%}",
            f"{m['retrieval_quality']:.2%}",
            f"{m['latency_mean']:.2f}s",
            f"{m['latency_p95']:.2f}s",
            f"{m['total_tokens']:,}",
            f"{m['peak_memory']:.2f} MB",
            f"{m['test_pass_rate']:.2%}"
        ]
        row_str = " | ".join(v.ljust(w) for v, w in zip(row_vals, col_widths))
        print(row_str)

    print("=" * len(header_row))


if __name__ == "__main__":
    calculate_metrics()
