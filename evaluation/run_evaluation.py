import json
import os
import time
import psutil
import requests

MODELS = ["codellama", "starcoder2:7b", "llama3.2:3b"]
OLLAMA_URL = "http://localhost:11434/api/generate"


def query_model(model, question, context=""):
    prompt = f"""
    You are a food safety AI assistant.
    Context: {context}
    Question: {question}
    Answer clearly and concisely.
    """
    start = time.time()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    latency = time.time() - start
    mem_after = process.memory_info().rss / 1024 / 1024
    data = response.json()

    return {
        "response": data["response"],
        "latency_seconds": round(latency, 2),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "memory_mb_delta": round(mem_after - mem_before, 2),
        "cpu_percent": psutil.cpu_percent(interval=1)
    }


def get_retrieval_context(question):
    r = requests.post(
        "http://localhost:8004/retrieve",
        json={"query": question, "collection": "both"}
    )
    return r.json().get("top_context", "")


def run_all():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    questions_path = (
        "evaluation/questions.json"
        if os.path.exists("evaluation/questions.json")
        else os.path.join(script_dir, "questions.json")
    )
    results_path = (
        "evaluation/raw_results.json"
        if os.path.exists("evaluation")
        else os.path.join(script_dir, "raw_results.json")
    )

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = {}

    for model in MODELS:
        results[model] = []
        for q in questions:
            context = ""
            if q.get("requires_rag", False):
                context = get_retrieval_context(q["question"])
            result = query_model(model, q["question"], context)
            result["question_id"] = q["id"]
            result["category"] = q["category"]
            results[model].append(result)
            print(f"✅ {model} | Q{q['id']} | {result['latency_seconds']}s")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_all()
