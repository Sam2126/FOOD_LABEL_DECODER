# Note: GPU recommended for starcoder2:7b and codellama

import requests

MODELS = ["codellama", "starcoder2:7b", "llama3.2:3b"]


def pull_models():
    for model in MODELS:
        print(f"Pulling {model}...")
        response = requests.post(
            "http://localhost:11434/api/pull",
            json={"name": model},
            stream=True
        )
        for line in response.iter_lines():
            if line:
                print(line.decode())


def verify_models():
    response = requests.get("http://localhost:11434/api/tags")
    available = [m["name"] for m in response.json().get("models", [])]
    for model in MODELS:
        status = "✅" if any(model in a for a in available) else "❌"
        print(f"{status} {model}")


if __name__ == "__main__":
    pull_models()
    verify_models()
