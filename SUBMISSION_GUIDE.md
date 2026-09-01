# Food Label Decoder - Assignment Submission Guide

## 1. Application Proposal

**Application / use case:** Food Label Decoder & RAG AI Studio

**Problem addressed:** Packaged-food labels contain unfamiliar ingredient names, E-numbers, chemical additives, and hidden allergens. Consumers need an instant, easy-to-understand explanation of what those ingredients mean without needing to research multiple websites.

**Knowledge / data source:** `knowledge_base/data.json` is a curated local knowledge base of food ingredients, additives, E-numbers, functional roles, allergen warnings, and cautious health considerations based on Codex Alimentarius and FDA reference guides. Each record retains its verified citation source.

**How RAG is used:** The application converts the knowledge-base records into independent chunks, creates 384-dimensional dense embeddings using `all-MiniLM-L6-v2`, and stores the vectors in a FAISS index. For every ingredient submitted by a user, it embeds the query independently, performs FAISS cosine-similarity search (`IndexFlatIP`), and injects only relevant chunks (scoring $\ge 0.30$) as context to the LLM.

**Role of the LLM:** The chosen Ollama model (`codellama`, `llama3`, `llama3.2`, `mistral`, etc.) generates a clear, ingredient-by-ingredient consumer explanation. The RAG prompt instructs it strictly to ground claims in the retrieved context, identify allergens, and avoid medical diagnoses.

**Microservice Architecture:**

```text
Browser / Frontend Studio (Port 8000)
  -> Application Service (FastAPI Orchestrator, Port 8000)
      -> RAG Service (FAISS Retrieval & Vector Search, Port 8001)
      -> LLM Service (Ollama Model Gateway, Port 8002)
          -> Ollama Local LLM Runtime (Port 11434)
      -> Data Service (Knowledge Base Ingestion & Vector Builder, Port 8003)
```

Replace the following before final submission:

```text
Group ID: G___
Member 1: ____________________
Member 2: ____________________
Member 3: ____________________
```

---

## 2. Requirement Mapping Across Exercises

| Assignment Exercise | Implementation in this Project | How to Demonstrate It |
| :--- | :--- | :--- |
| **Exercise 1: Application + API + Ollama** | `app/main.py` exposes `POST /api/decode_no_rag`; `app/ollama_client.py` communicates with Ollama's `/api/generate`. | Submit an ingredient list and demonstrate the ungrounded baseline response without retrieval. |
| **Exercise 2: Knowledge Base + Chunking + Embeddings** | `knowledge_base/data.json`, `ingestion/chunker.py`, and `ingestion/embedder.py` create 384-dimensional embeddings using `all-MiniLM-L6-v2`. | Show `embeddings/vectors.npy` and `embeddings/chunks.json`. Inspect the **Chunk Explorer** and **Vector Heatmap** tabs in the UI. |
| **Exercise 3: Vector Similarity + Retrieval + RAG** | `rag/vector_store.py` (FAISS `IndexFlatIP`), `rag/retriever.py` (cosine similarity search), and `rag/rag_pipeline.py` (context formatting & grounded prompts). | Enable the **Compare with Non-RAG Baseline** toggle in the UI to demonstrate side-by-side grounded vs baseline answers, allergen badges, and retrieved sources. |
| **Exercise 4: APIs + Microservices + Orchestration** | `services/app_service`, `services/rag_service`, `services/llm_service`, and `services/data_service` run as isolated FastAPI services with REST communication. | Open `http://localhost:8000/api/health` and verify that the application service checks downstream RAG and LLM services. |
| **Exercise 5: Docker + Complete Architecture** | `docker-compose.yml` runs all 4 microservices, Ollama, shared named volume (`shared_embeddings`), health checks, and service networking. | Run `docker compose ps` and interact with the live system on `http://localhost:8000`. |

---

## 3. End-to-End Request Journey

```text
User enters ingredient list (e.g. "Wheat flour, sugar, palm oil, E322, E621, milk solids")
  │
  ├──> 1. Application Service API receives request & parses ingredients
  │
  ├──> 2. RAG Service embeds each ingredient using SentenceTransformer (384-d)
  │
  ├──> 3. FAISS Index computes inner-product cosine similarity against all stored chunks
  │
  ├──> 4. Top-K chunks scoring >= 0.30 are filtered and assembled into a context block
  │
  ├──> 5. Application Service constructs grounded prompt with strict instructions & allergen rules
  │
  ├──> 6. LLM Service sends prompt to Ollama (selected model)
  │
  └──> 7. Structured, grounded explanation with source citations is returned to the user
```

---

## 4. Live Demonstration Steps

1. **Start the complete stack:**
   ```powershell
   docker compose up --build -d
   docker compose run --rm ollama-pull
   ```
   *(Or for local Python dev mode: `python -m ingestion.embedder` followed by `uvicorn app.main:app --reload --port 8000`)*

2. **Verify system health & status:**
   ```powershell
   docker compose ps
   Invoke-RestMethod http://localhost:8000/api/health
   ```

3. **Open the Web Studio:**
   Navigate to **`http://localhost:8000`** in your browser.

4. **Demonstrate Model Selection & Parameter Control:**
   - Show the dynamic **LLM Model** dropdown (populated from installed Ollama models).
   - Adjust **Temperature**, **Max Tokens**, or **Similarity Threshold** sliders.

5. **Test Sample Ingredients:**
   - Click one of the preset chips (e.g., *🍪 Biscuit / Snack* or *⚠️ High-Allergen Blend*).
   - Click **⚡ Decode Ingredients**.

6. **Demonstrate RAG vs Non-RAG Comparison:**
   - Compare the **Grounded Analysis (With RAG)** column with the **Baseline Analysis (Without RAG)** column.
   - Point out the **Retrieved Evidence & Sources** citations and **🚨 Allergen Alert Banner**.

7. **Demonstrate Transparency Tabs:**
   - **Query & Vector Inspector:** Click ingredient pills to see real-time 384-d vector bars and FAISS scores.
   - **Knowledge Base Explorer:** Search for an E-number (e.g., `E322`, `E621`) or ingredient name.
   - **Vector Embedding Heatmap:** View the 2D matrix visualization of chunk embeddings.
   - **Prompt & Grounding Inspector:** Inspect the exact context block injected into the LLM.

---

## 5. API Endpoints Reference

| Service | Endpoint | Method | Purpose |
| :--- | :--- | :--- | :--- |
| **Application** | `GET /` | `GET` | Serves the interactive Food Label Decoder Web Studio. |
| **Application** | `GET /api/health` | `GET` | Reports application, RAG, and LLM health status. |
| **Application** | `GET /api/models` | `GET` | Lists all model tags pulled in the local Ollama instance. |
| **Application** | `GET /api/chunks` | `GET` | Returns stored knowledge base chunks for the explorer. |
| **Application** | `GET /api/vectors` | `GET` | Returns embedding matrix snapshot for the heatmap. |
| **Application** | `POST /api/query-embed` | `POST` | Embeds a query and returns similarity matches with vector preview. |
| **Application** | `POST /api/decode` | `POST` | Complete RAG-powered decoding with verified evidence. |
| **Application** | `POST /api/decode_no_rag` | `POST` | Exercise 1 baseline generation without retrieval. |
| **RAG** | `POST /retrieve` | `POST` | Returns retrieved chunks and formatted context for ingredients. |
| **Data** | `GET /health/ready` | `GET` | Reports whether embeddings and vector index are built and ready. |
| **LLM** | `POST /generate` | `POST` | Dispatches prompt to Ollama with model and parameter overrides. |
