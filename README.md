# 🥫 Food Label Decoder · Professional RAG AI Studio

An end-to-end **Retrieval-Augmented Generation (RAG)** platform that analyzes packaged food ingredient lists, E-numbers, additives, and allergens with strict local knowledge grounding.

Built with **Ollama** + **Llama 3.2 / Llama 3 / Code Llama / Mistral** + **FastAPI** + **FAISS** + **Sentence-Transformers (MiniLM-L6-v2)** + **Docker**.

---

## 🌟 Key Highlights & Platform Capabilities

- ⚔️ **Multi-Model Comparison Arena & AI Referee:** Dynamically benchmarks all locally installed Ollama models (`llama3.2:1b`, `llama3:latest`, `codellama:latest`) on the same food label query. Calculates real-time telemetry (Latency, Throughput, Grounding %, Allergen Recall %, Hallucination Rate, Composite Score) and synthesizes an intelligent AI Referee evaluation crowning the optimal model.
- 🚨 **Ingredient-to-Allergen Breakdown Mapping:** Maps each raw label ingredient directly to its allergen group, clinical severity level (`CRITICAL`, `SEVERE`, `MODERATE-HIGH`, `SAFE`), and clinical/dietary guidance.
- 🌿 **Grounded RAG Intelligence:** Prevents LLM hallucinations by retrieving exact Codex Alimentarius and FDA food additive records.
- 📦 **Curated 44-Record Knowledge Base:** Features comprehensive data on emulsifiers, acidity regulators, stabilizers (INS 407, 412, 415, 466), thickeners (INS 440), colorants (INS 102, 127, 160c), oils, and allergen classes.
- ⚖️ **Side-by-Side Evaluation:** Direct comparison of **RAG Grounded Response** vs **Baseline (Without RAG)** to demonstrate grounding efficacy.
- 🔬 **Query & Vector Inspector:** Interactive ingredient pills with 384-dimensional vector activation charts and FAISS cosine similarity scoring.
- 🧮 **2D Vector Embedding Matrix Heatmap:** High-performance HTML5 Canvas matrix visualization of the dense vector embedding space.
- 🛡️ **Prompt & Grounding Transparency:** Inspect the exact system instructions, constraints, and retrieved context injected into the LLM.
- 🌓 **Modern Glassmorphism UI:** Rich dark mode interface with rendered markdown typography, high-contrast badges, fluid animations, and responsive layout.

---

## 📂 Detailed Project File Structure

```text
food-label-decoder/
├── app/                                 # Monolithic Application & API
│   ├── main.py                          # FastAPI backend orchestrator & API endpoints
│   ├── ollama_client.py                 # Isolated Ollama gateway with dynamic model discovery
│   └── static/
│       └── index.html                   # Professional RAG AI Studio frontend UI
│
├── rag/                                 # Vector Database & Retrieval Pipeline
│   ├── vector_store.py                  # FAISS IndexFlatIP (Inner Product) wrapper
│   ├── retriever.py                     # Recursive balanced-parenthesis tokenizer & cosine search
│   ├── rag_pipeline.py                  # Prompt synthesis, context assembly & allergen classifier
│   └── model_comparator.py              # Multi-model arena benchmarking & AI Referee engine
│
├── ingestion/                           # Knowledge Ingestion & Embedding Pipeline
│   ├── chunker.py                       # Converts raw knowledge base records into text chunks
│   └── embedder.py                      # Generates 384-d MiniLM embeddings & FAISS files
│
├── knowledge_base/
│   └── data.json                        # Curated database of 44 food additives, allergens & E-numbers
│
├── embeddings/                          # Generated vector store artifacts
│   ├── chunks.json                      # Serialized chunk metadata & text passages
│   └── vectors.npy                      # 384-dimensional normalized float32 embedding matrix
│
├── eval/                                # Evaluation & Quality Metrics Suite
│   ├── metrics.py                       # Grounding accuracy, allergen recall & hallucination scoring
│   ├── questions.json                   # Curated test benchmark queries
│   └── test_retrieval.py                # Automated retrieval & vector search unit tests
│
├── services/                            # Microservices Architecture
│   ├── app_service/                     # Public Web API & Frontend Host (Port 8000)
│   ├── rag_service/                     # FAISS Vector Search Microservice (Port 8001)
│   ├── llm_service/                     # Ollama / LLM Communication Service (Port 8002)
│   └── data_service/                    # Knowledge Base & Index Builder Service (Port 8003)
│
├── docker-compose.yml                   # Multi-container orchestration specification
├── requirements.txt                     # Dependencies for local development
├── SUBMISSION_GUIDE.md                  # Project submission documentation & exercise mapping
└── README.md                            # Main project documentation
```

---

## 🚀 How to Run the Project

### Option 1: Local Development Server (Recommended for Fast Local Testing)

1. **Activate your Python environment & install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running locally:**
   ```powershell
   ollama serve
   ```

3. **Verify installed models:**
   ```powershell
   ollama list
   # To pull lightweight 1B model for ultra-fast multi-model comparison:
   ollama pull llama3.2:1b
   ```

4. **Build the vector store:**
   ```powershell
   python -m ingestion.embedder
   ```

5. **Start the FastAPI backend with live reloading:**
   ```powershell
   uvicorn app.main:app --reload --port 8000
   ```

6. **Open in browser:**  
   Navigate to **[http://localhost:8000](http://localhost:8000)**.

---

### Option 2: Full Docker Microservices Stack

Runs the complete 5-container architecture (App, RAG, Data, LLM services + Ollama) in isolated containers connected over an internal Docker network.

1. **Build and start the services:**
   ```powershell
   docker compose up --build -d
   ```

2. **Download the default LLM model (First time only):**
   ```powershell
   docker compose run --rm ollama-pull
   ```

3. **Open the App:**  
   Navigate to **[http://localhost:8000](http://localhost:8000)**.

---

## 🔌 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive RAG AI Studio Web UI |
| `GET` | `/api/health` | System health, Ollama connection status & discovered models |
| `GET` | `/api/models` | List of installed Ollama models |
| `GET` | `/api/chunks` | Returns all 44 knowledge base chunks and metadata |
| `GET` | `/api/vectors` | Returns vector dimensions and index information |
| `POST` | `/api/query-embed` | Embeds an ingredient text query and computes similarity scores |
| `POST` | `/api/decode` | **RAG Grounded Decode:** Retrieves context, detects allergens, and generates grounded report |
| `POST` | `/api/decode_no_rag` | **Baseline Decode:** Generates analysis from LLM parametric memory without retrieval |
| `POST` | `/api/compare_models` | **Multi-Model Arena:** Benchmarks all models, measures telemetry & generates AI Referee verdict |

---

## ⚔️ Multi-Model Comparison Arena Example

When evaluating a packaged food label (e.g. `Wheat flour, sugar, palm oil, E322, E621, milk solids`):

```json
{
  "ingredient_list": "Wheat flour, sugar, palm oil, E322, E621, milk solids",
  "champion_model": "llama3.2:1b",
  "verdict_title": "🏆 llama3.2:1b — Optimal Production Choice",
  "referee_rationale": "llama3.2:1b delivered 6.8x faster latency (0.72s) with 94.5% Grounding Accuracy and 100% Allergen Recall.",
  "leaderboard": [
    {
      "rank": 1,
      "model": "llama3.2:1b",
      "latency_sec": 0.72,
      "throughput_tokens_sec": 312.4,
      "grounding_score": 94.5,
      "allergen_recall": 100.0,
      "hallucination_rate": 4.2,
      "composite_score": 96.2
    },
    {
      "rank": 2,
      "model": "codellama:latest",
      "latency_sec": 4.88,
      "throughput_tokens_sec": 46.2,
      "grounding_score": 91.0,
      "allergen_recall": 100.0,
      "hallucination_rate": 6.8,
      "composite_score": 88.4
    }
  ]
}
```

---

## 🧪 Automated Testing

Run the test suite to verify vector retrieval accuracy and similarity thresholds:

```powershell
python -m unittest eval/test_retrieval.py
```
