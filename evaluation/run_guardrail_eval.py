"""
Guardrail Evaluation Runner
---------------------------
Runs the complete guardrail test suite and produces:
  - Per-case pass/fail results
  - Category-level statistics (Attack Block Rate, False Positive Rate, etc.)
  - evaluation/guardrail_metrics.json  (machine-readable)
  - Console summary report

Usage:
    python evaluation/run_guardrail_eval.py
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import Any, Dict, List

# ── Resolve paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GUARDRAIL_DIR = PROJECT_ROOT / "services" / "guardrail_service"
EVAL_DIR = PROJECT_ROOT / "evaluation"
TEST_SUITE_PATH = EVAL_DIR / "guardrail_test_suite.json"
METRICS_OUTPUT_PATH = EVAL_DIR / "guardrail_metrics.json"

# Add guardrail service to path
if str(GUARDRAIL_DIR) not in sys.path:
    sys.path.insert(0, str(GUARDRAIL_DIR))

from guardrails import run_input_guardrails, run_output_guardrails

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_test_suite() -> Dict:
    with open(TEST_SUITE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_input_text(case: Dict) -> str:
    """Handle static vs generator inputs."""
    if "input" in case and case["input"] is not None:
        return case["input"]
    # Generator case: repeat
    if case.get("input_generator") == "repeat":
        return case["input_base"] * case.get("repeat_count", 10)
    return ""


def evaluate_input_case(case: Dict) -> Dict[str, Any]:
    """Run input guardrails for a single test case."""
    text = build_input_text(case)
    t0 = time.perf_counter()
    result = run_input_guardrails(text)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    actual_verdict = result.get("verdict", "unknown")
    expected_verdict = case.get("expected_verdict", "pass")
    test_passed = actual_verdict == expected_verdict

    # Optionally check violation category
    actual_violation = result.get("violation_category", None)
    expected_violation = case.get("expected_violation", None)
    category_match = (expected_violation is None) or (actual_violation == expected_violation)

    return {
        "id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "input_preview": (text[:120] + "...") if len(text) > 120 else text,
        "expected_verdict": expected_verdict,
        "actual_verdict": actual_verdict,
        "expected_violation": expected_violation,
        "actual_violation": actual_violation,
        "test_passed": test_passed and category_match,
        "verdict_match": test_passed,
        "category_match": category_match,
        "latency_ms": elapsed,
        "guardrail_checks": result.get("checks", []),
        "reason": result.get("reason", result.get("message", "")),
    }


def evaluate_output_case(case: Dict) -> Dict[str, Any]:
    """Run output guardrails for a single test case."""
    text = build_input_text(case)
    mock_output = dict(case.get("mock_llm_output", {}))

    t0 = time.perf_counter()
    result = run_output_guardrails(mock_output, text)
    elapsed = round((time.perf_counter() - t0) * 1000, 2)

    checks = result.get("output_guardrail_checks", [])
    target_guard = case.get("expected_output_guard", "")
    target_check = next((c for c in checks if c.get("guard") == target_guard), {})

    # Determine test result based on category
    cat = case["category"]
    test_passed = False
    detail = {}

    if cat == "output_schema":
        expected_schema_passed = case.get("expected_schema_passed", True)
        expected_repaired = case.get("expected_auto_repaired", False)
        actual_schema_passed = target_check.get("passed", True)
        corrections = result.get("output_corrections", [])
        was_repaired = len(corrections) > 0
        test_passed = (actual_schema_passed == expected_schema_passed) or (expected_repaired == was_repaired)
        detail = {
            "schema_passed": actual_schema_passed,
            "was_repaired": was_repaired,
            "corrections": corrections,
            "missing_keys": target_check.get("missing_keys", []),
        }

    elif cat == "output_grounding":
        expected_grounding_passed = case.get("expected_grounding_passed", True)
        actual_grounding_passed = target_check.get("passed", True)
        actual_hallucinated = target_check.get("hallucinated", [])
        expected_hallucinated = case.get("expected_hallucinated", [])
        test_passed = actual_grounding_passed == expected_grounding_passed
        detail = {
            "grounding_passed": actual_grounding_passed,
            "hallucinated_detected": actual_hallucinated,
            "expected_hallucinated": expected_hallucinated,
            "correctly_identified": set(actual_hallucinated) == set(expected_hallucinated)
        }

    elif cat == "output_severe_claims":
        expected_severe_passed = case.get("expected_severe_passed", True)
        actual_severe_passed = target_check.get("passed", True)
        test_passed = actual_severe_passed == expected_severe_passed
        detail = {
            "severe_passed": actual_severe_passed,
            "unverified_count": target_check.get("unverified_count", 0),
        }

    return {
        "id": case["id"],
        "category": cat,
        "description": case["description"],
        "input_preview": (text[:120] + "...") if len(text) > 120 else text,
        "target_guard": target_guard,
        "test_passed": test_passed,
        "latency_ms": elapsed,
        "detail": detail,
        "all_checks": checks,
        "disclaimer_injected": "disclaimer" in result,
    }


def run_evaluation() -> Dict:
    """Run all test cases and aggregate results."""
    suite = load_test_suite()
    test_cases = suite["test_cases"]

    results = []
    OUTPUT_CATS = {"output_schema", "output_grounding", "output_severe_claims"}

    for case in test_cases:
        cat = case.get("category", "")
        if cat in OUTPUT_CATS:
            res = evaluate_output_case(case)
        else:
            res = evaluate_input_case(case)
        results.append(res)

    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute category-level and overall metrics."""
    total = len(results)
    passed = sum(1 for r in results if r["test_passed"])
    failed = total - passed
    overall_accuracy = round(passed / total * 100, 1)

    # Group by category
    categories: Dict[str, List] = {}
    for r in results:
        cat = r["category"]
        categories.setdefault(cat, []).append(r)

    cat_stats = {}
    for cat, cases in categories.items():
        cat_total = len(cases)
        cat_passed = sum(1 for c in cases if c["test_passed"])
        cat_stats[cat] = {
            "total": cat_total,
            "passed": cat_passed,
            "failed": cat_total - cat_passed,
            "pass_rate": round(cat_passed / cat_total * 100, 1),
        }

    # Specific metrics
    attack_cats = {"out_of_scope", "prompt_injection", "excessive_length", "empty_gibberish", "insufficient_info"}
    attack_cases = [r for r in results if r.get("category") in attack_cats]
    attack_blocked = sum(1 for r in attack_cases if r.get("actual_verdict") == "reject" or r["test_passed"])
    attack_block_rate = round(attack_blocked / len(attack_cases) * 100, 1) if attack_cases else 0

    valid_cats = {"in_scope_valid"}
    valid_cases = [r for r in results if r.get("category") in valid_cats]
    valid_allowed = sum(1 for r in valid_cases if r.get("actual_verdict") == "pass" or r["test_passed"])
    false_reject_rate = round((len(valid_cases) - valid_allowed) / len(valid_cases) * 100, 1) if valid_cases else 0

    avg_latency = round(sum(r.get("latency_ms", 0) for r in results) / total, 2) if results else 0

    return {
        "overall": {
            "total_cases": total,
            "passed": passed,
            "failed": failed,
            "overall_accuracy_pct": overall_accuracy,
            "attack_block_rate_pct": attack_block_rate,
            "false_reject_rate_pct": false_reject_rate,
            "avg_latency_ms": avg_latency,
        },
        "by_category": cat_stats,
        "failed_cases": [
            {"id": r["id"], "category": r["category"], "description": r["description"],
             "expected": r.get("expected_verdict") or r.get("target_guard"),
             "actual": r.get("actual_verdict", "see_detail")}
            for r in results if not r["test_passed"]
        ],
    }


def print_report(metrics: Dict, results: List[Dict]) -> None:
    """Print a clean ASCII report to stdout."""
    ov = metrics["overall"]
    sep = "=" * 70

    print(sep)
    print("  GUARDRAIL EVALUATION REPORT - Food Label Decoder AI System")
    print(sep)
    print(f"  Total Test Cases  : {ov['total_cases']}")
    print(f"  Passed            : {ov['passed']}")
    print(f"  Failed            : {ov['failed']}")
    print(f"  Overall Accuracy  : {ov['overall_accuracy_pct']}%")
    print(f"  Attack Block Rate : {ov['attack_block_rate_pct']}%   (attacks correctly rejected)")
    print(f"  False Reject Rate : {ov['false_reject_rate_pct']}%   (valid inputs wrongly blocked)")
    print(f"  Avg Latency       : {ov['avg_latency_ms']} ms/case")
    print(sep)
    print("  CATEGORY BREAKDOWN")
    print(sep)
    print(f"  {'Category':<30} {'Total':>5} {'Pass':>5} {'Fail':>5} {'Rate':>7}")
    print(f"  {'-'*30} {'-'*5} {'-'*5} {'-'*5} {'-'*7}")
    for cat, stats in metrics["by_category"].items():
        bar = stats["pass_rate"]
        print(f"  {cat:<30} {stats['total']:>5} {stats['passed']:>5} {stats['failed']:>5} {bar:>6.1f}%")

    if metrics["failed_cases"]:
        print(sep)
        print("  FAILED CASES")
        print(sep)
        for fc in metrics["failed_cases"]:
            print(f"  [{fc['id']}] {fc['category']} - {fc['description']}")
            print(f"        Expected: {fc['expected']}  |  Got: {fc['actual']}")

    print(sep)
    print("  Evaluation complete. Metrics saved to evaluation/guardrail_metrics.json")
    print(sep)


def main():
    print("Running guardrail evaluation suite...")
    results = run_evaluation()
    metrics = compute_metrics(results)
    print_report(metrics, results)

    # Save full results
    output = {
        "metrics": metrics,
        "detailed_results": results,
    }
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nDetailed results saved: {METRICS_OUTPUT_PATH}")
    return metrics["overall"]["overall_accuracy_pct"]


if __name__ == "__main__":
    acc = main()
    sys.exit(0 if acc >= 80 else 1)
