"""evaluation/generate_report.py

Generates evaluation/report.md from:
  - evaluation/metrics_summary.json   (output of calculate_metrics.py)
  - evaluation/rag_analysis_results.json (output of rag_analysis.py)
  - evaluation/hallucination_manual.json (manual annotations)

Run after all evaluation steps are complete.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR              = Path(__file__).resolve().parent
METRICS_PATH          = EVAL_DIR / "metrics_summary.json"
RAG_ANALYSIS_PATH     = EVAL_DIR / "rag_analysis_results.json"
HALLUCINATION_PATH    = EVAL_DIR / "hallucination_manual.json"
OUTPUT_PATH           = EVAL_DIR / "report.md"

# Display names for known model keys
MODEL_DISPLAY = {
    "codellama":    "CodeLlama 7B",
    "starcoder2:7b": "StarCoder2 7B",
    "llama3.2:3b":  "Llama 3.2 3B",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_json(path: Path, default: Any = None) -> Any:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"  ⚠️  Not found: {path.name} — using default.")
    return default


def display(model_key: str) -> str:
    return MODEL_DISPLAY.get(model_key, model_key)


def pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def sec(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.2f}s"


def tok(val: Optional[int]) -> str:
    if val is None:
        return "—"
    return f"{val:,}"


def mb(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{val:.1f} MB"


def _best(metrics: Dict, key: str, lower_is_better: bool = False) -> str:
    """Return the model key with the best value for `key`."""
    candidates = {m: v.get(key) for m, v in metrics.items() if v.get(key) is not None}
    if not candidates:
        return "N/A"
    return min(candidates, key=candidates.__getitem__) if lower_is_better \
        else max(candidates, key=candidates.__getitem__)


def _val(metrics: Dict, model: str, key: str) -> Optional[float]:
    return metrics.get(model, {}).get(key)


# ── Section builders ──────────────────────────────────────────────────────────
def section_models(models: List[str]) -> str:
    lines = ["## 1. Models Evaluated\n"]
    for m in models:
        lines.append(f"- **{display(m)}** (`{m}`)")
    lines += [
        "",
        "All models were tested on the **same 25 questions**, with the **same knowledge base** "
        "and **identical prompt templates** to ensure a fair comparison.",
        "",
        f"*Evaluation run: {datetime.now().strftime('%Y-%m-%d')}*",
    ]
    return "\n".join(lines)


def section_metrics_table(metrics: Dict, models: List[str]) -> str:
    cols = ["Metric"] + [display(m) for m in models]

    def row(label: str, vals: List[str]) -> str:
        return "| " + " | ".join([label] + vals) + " |"

    header  = "| " + " | ".join(cols) + " |"
    divider = "| " + " | ".join("---" for _ in cols) + " |"

    rows = [
        row("Correctness",       [pct(_val(metrics, m, "correctness"))         for m in models]),
        row("Hallucination Rate",[pct(_val(metrics, m, "hallucination_rate"))  for m in models]),
        row("Retrieval Quality", [pct(_val(metrics, m, "retrieval_quality"))   for m in models]),
        row("Latency (mean)",    [sec(_val(metrics, m, "latency_mean"))        for m in models]),
        row("Latency (P95)",     [sec(_val(metrics, m, "latency_p95"))         for m in models]),
        row("Total Tokens",      [tok(_val(metrics, m, "total_tokens"))        for m in models]),
        row("Peak Memory",       [mb(_val(metrics, m, "peak_memory"))          for m in models]),
        row("Code Test Pass Rate",[pct(_val(metrics, m, "test_pass_rate"))     for m in models]),
    ]

    lines = ["## 2. Metrics Comparison Table\n", header, divider] + rows
    return "\n".join(lines)


def section_rag_analysis(rag_results: List[Dict]) -> str:
    lines = ["## 3. RAG Pipeline Analysis\n",
             "Six questions were selected that test the pipeline's ability to ground "
             "answers in the FSSAI regulatory knowledge base.\n"]

    case_emoji = {
        "good_retrieval_correct_response":      "✅",
        "bad_retrieval_hallucination":          "🔴",
        "good_retrieval_still_hallucinated":    "⚠️",
        "no_retrieval_confident_wrong":         "❌",
        None:                                   "—",
    }

    for r in rag_results:
        qid      = r.get("question_id", "?")
        question = r.get("question", "")
        chunks   = r.get("retrieved_chunks", [])
        scores   = r.get("similarity_scores", [])
        case     = r.get("analysis", {}).get("case_type")
        icon     = case_emoji.get(case, "—")

        lines.append(f"### Q{qid} — {question}\n")

        # Retrieved chunks
        lines.append("**Retrieved chunks:**\n")
        if chunks:
            for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
                preview = chunk[:120].replace("\n", " ")
                lines.append(f"{i}. *(score: {score:.3f})* {preview}…")
        else:
            lines.append("*No chunks retrieved.*")

        lines.append("")

        # Responses
        with_rag    = r.get("response_with_rag", "—")
        without_rag = r.get("response_without_rag", "—")
        lat_w  = r.get("latency_with_rag_s",   "—")
        lat_nw = r.get("latency_no_rag_s", "—")

        lines.append(f"**Response WITH RAG** *(latency: {lat_w}s)*\n")
        lines.append(f"> {with_rag[:300].replace(chr(10), ' ')}{'…' if len(with_rag) > 300 else ''}\n")

        lines.append(f"**Response WITHOUT RAG** *(latency: {lat_nw}s)*\n")
        lines.append(f"> {without_rag[:300].replace(chr(10), ' ')}{'…' if len(without_rag) > 300 else ''}\n")

        # Analysis
        analysis = r.get("analysis", {})
        lines.append(f"**Case type:** {icon} `{case or 'not annotated'}`\n")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Retrieval relevant? | {_bool_md(analysis.get('retrieval_relevant'))} |")
        lines.append(f"| RAG response correct? | {_bool_md(analysis.get('rag_response_correct'))} |")
        lines.append(f"| No-RAG response correct? | {_bool_md(analysis.get('no_rag_response_correct'))} |")
        lines.append(f"| Hallucination WITHOUT RAG? | {_bool_md(analysis.get('hallucination_without_rag'))} |")
        lines.append(f"| Hallucination WITH RAG? | {_bool_md(analysis.get('hallucination_with_rag'))} |")
        lines.append("")

    return "\n".join(lines)


def _bool_md(val) -> str:
    if val is True:
        return "✅ Yes"
    if val is False:
        return "❌ No"
    return "— *(not annotated)*"


def section_key_findings(metrics: Dict, models: List[str]) -> str:
    lines = ["## 4. Key Findings\n"]

    if not metrics:
        lines.append("*Run `calculate_metrics.py` first to generate metrics_summary.json.*")
        return "\n".join(lines)

    # Best accuracy
    best_acc = _best(metrics, "correctness")
    acc_val  = pct(_val(metrics, best_acc, "correctness"))
    lines.append(f"- 🏆 **Best accuracy:** {display(best_acc)} at **{acc_val}**")

    # Lowest hallucination
    best_hall = _best(metrics, "hallucination_rate", lower_is_better=True)
    hall_val  = pct(_val(metrics, best_hall, "hallucination_rate"))
    lines.append(f"- 🛡️  **Lowest hallucination rate:** {display(best_hall)} at **{hall_val}**")

    # Fastest model
    fastest    = _best(metrics, "latency_mean", lower_is_better=True)
    fast_val   = sec(_val(metrics, fastest, "latency_mean"))
    lines.append(f"- ⚡ **Fastest model:** {display(fastest)} at **{fast_val}** mean latency")

    # Most tokens efficient
    eff        = _best(metrics, "total_tokens", lower_is_better=True)
    eff_val    = tok(_val(metrics, eff, "total_tokens"))
    lines.append(f"- 💡 **Most token-efficient:** {display(eff)} with **{eff_val}** total tokens")

    # Best code generation
    best_code  = _best(metrics, "test_pass_rate")
    code_val   = pct(_val(metrics, best_code, "test_pass_rate"))
    lines.append(f"- 🧑‍💻 **Best code generation:** {display(best_code)} with **{code_val}** test pass rate")

    # Best retrieval quality
    best_ret   = _best(metrics, "retrieval_quality")
    ret_val    = pct(_val(metrics, best_ret, "retrieval_quality"))
    lines.append(f"- 🔍 **Best retrieval quality:** {display(best_ret)} at **{ret_val}**")

    # Quality-latency tradeoff narrative
    lines.append("")
    lines.append("### Quality-Latency Tradeoff\n")

    for m in models:
        lat  = _val(metrics, m, "latency_mean")
        acc  = _val(metrics, m, "correctness")
        if lat is not None and acc is not None:
            lines.append(f"- **{display(m)}:** {pct(acc)} correctness @ {sec(lat)} mean latency")

    return "\n".join(lines)


def section_routing_recommendation(metrics: Dict) -> str:
    lines = ["## 5. Routing Recommendation\n",
             "Based on the evaluation evidence, the following routing strategy is recommended:\n",
             "| Task Type | Recommended Model | Rationale |",
             "|---|---|---|"]

    if not metrics:
        lines.append("*Metrics not available — run calculate_metrics.py first.*")
        return "\n".join(lines)

    # Simple factual tasks → fastest model with acceptable accuracy
    fastest     = _best(metrics, "latency_mean", lower_is_better=True)
    most_acc    = _best(metrics, "correctness")
    best_code   = _best(metrics, "test_pass_rate")
    safest      = _best(metrics, "hallucination_rate", lower_is_better=True)

    lines.append(f"| **Simple** (allergen, ingredient lookup) | {display(fastest)} | Lowest latency, sufficient for factual queries |")
    lines.append(f"| **Complex** (safety flags, regulatory)   | {display(most_acc)} | Highest accuracy, best regulatory reasoning |")
    lines.append(f"| **Generative** (recipe, code)            | {display(best_code)} | Best code test pass rate |")
    lines.append(f"| **Hallucination-sensitive** (trap Q)     | {display(safest)} | Lowest hallucination rate |")

    lines += [
        "",
        "> **Note:** This recommendation is derived automatically from the evaluation metrics.",
        "> Override the ROUTING_RULES in `orchestrator/app.py` if your use-case priorities differ.",
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────
def generate_report():
    print("\n📊 Loading evaluation data…")
    metrics      = load_json(METRICS_PATH,       default={})
    rag_results  = load_json(RAG_ANALYSIS_PATH,  default=[])
    _halluc      = load_json(HALLUCINATION_PATH, default=[])   # loaded for completeness

    models = list(metrics.keys()) if metrics else list(MODEL_DISPLAY.keys())
    print(f"   Models found: {models}")
    print(f"   RAG analysis entries: {len(rag_results)}")

    sections = [
        f"# Food Label Decoder — Model Evaluation Report\n",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n",
        "---\n",
        section_models(models),
        "\n---\n",
        section_metrics_table(metrics, models),
        "\n---\n",
        section_rag_analysis(rag_results),
        "\n---\n",
        section_key_findings(metrics, models),
        "\n---\n",
        section_routing_recommendation(metrics),
        "\n---\n",
        "## Appendix — File Index\n",
        "| File | Description |",
        "|---|---|",
        "| `evaluation/questions.json` | 25 evaluation questions across 13 categories |",
        "| `evaluation/raw_results.json` | Raw model responses + latency per question |",
        "| `evaluation/metrics_summary.json` | Aggregated 8-metric summary per model |",
        "| `evaluation/rag_analysis_results.json` | RAG vs No-RAG comparison for 6 questions |",
        "| `evaluation/hallucination_manual.json` | Manual hallucination annotations |",
        "| `evaluation/retrieval_manual.json` | Manual retrieval quality annotations |",
        "| `evaluation/report.md` | This report |",
    ]

    report = "\n".join(sections)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n✅ Report written to: {OUTPUT_PATH}")
    print(f"   Sections: Models | Metrics Table | RAG Analysis | Key Findings | Routing Recommendation")


if __name__ == "__main__":
    generate_report()
