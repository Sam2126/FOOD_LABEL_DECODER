# Repository & Codebase Understanding Notes (Pre-Sourcegraph Groundwork)

**Course / Project:** Food Label Decoder · Professional RAG AI Studio  
**Date:** September 2026  
**Deliverable:** Week 4 – Exercise 6  

---

## 1. Objective & Scope

In preparation for advanced code intelligence tools (Sourcegraph / Cody) in Week 5, this exercise evaluates how the current LLM and retrieval architecture performs when queried about **its own codebase and architecture**, rather than food ingredients.

We posed 5 critical architectural questions to the system to evaluate:
- **Lexical vs. Structural Code Understanding:** Does the model understand multi-file dependencies?
- **Domain Boundaries:** What happens when an LLM trained on code attempts to analyze an un-indexed project repo?
- **Failure Modes:** How does the absence of code graph indexing affect component reasoning?

---

## 2. Test Queries & Response Analysis

### Query 1: "Which service handles FAISS vector search in the microservice architecture?"

- **Expected Answer:** `services/rag_service` (specifically `services/rag_service/vector_store.py` and `services/rag_service/retriever.py`, running on port 8001).
- **LLM Output (Code Llama / Llama 3):**
  > *"In the microservice architecture, the **RAG Service (`services/rag_service`)** running on port 8001 handles FAISS vector search. It loads the shared index from `shared_embeddings` and uses `IndexFlatIP` to perform cosine similarity calculations."*
- **Assessment:** **CORRECT (100%)**. The model accurately identified the service name, file path, port (8001), and FAISS index type.

---

### Query 2: "Which file builds the embedding index on startup?"

- **Expected Answer:** In monolithic mode: `ingestion/embedder.py`. In microservices mode: `services/data_service/main.py` (which calls `embedder.py` to write to the `shared_embeddings` volume).
- **LLM Output:**
  > *"In the monolithic setup, running `python -m ingestion.embedder` builds the index files `embeddings/vectors.npy` and `embeddings/chunks.json`. In Docker Compose, the **Data Service (`services/data_service/main.py`)** runs on startup and generates the vector index."*
- **Assessment:** **CORRECT (100%)**. Clearly differentiated between the monolithic development script and the containerized `data_service`.

---

### Query 3: "What happens end-to-end when a user submits a POST request to `/api/decode`?"

- **Expected Answer:**
  1. `app_service` receives JSON payload with `ingredient_list`.
  2. `app_service` calls `rag_service` at `POST /retrieve`.
  3. `rag_service` splits ingredients, embeds each with MiniLM, runs FAISS search, and filters chunks $\ge 0.30$.
  4. `app_service` constructs the grounded prompt with injected context.
  5. `app_service` calls `llm_service` at `POST /generate`.
  6. `llm_service` forwards prompt to Ollama.
  7. Response and metadata returned to browser.
- **LLM Output:**
  > *"The request hits `app_service` (port 8000), which acts as an orchestrator. It calls `rag_service` (port 8001) to retrieve relevant chunks via FAISS. It then constructs a grounded prompt combining retrieved context and system rules, dispatches this prompt to `llm_service` (port 8002) which queries Ollama, and finally formats the response for the browser."*
- **Assessment:** **CORRECT (95%)**. Captured the multi-hop REST orchestration accurately.

---

### Query 4: "Which components would break if `all-MiniLM-L6-v2` were swapped for a 768-dimensional model like `bge-base-en`?"

- **Expected Answer:**
  - `ingestion/embedder.py` / `services/data_service/embedder.py` (needs model name change).
  - Existing `vectors.npy` must be regenerated (shape changes from $(N, 384) \to (N, 768)$).
  - `rag/vector_store.py` / `services/rag_service/vector_store.py` (FAISS index dimension changes from $d=384 \to d=768$).
  - `services/rag_service/embedding_utils.py` (query embedding model must match).
  - Frontend vector visualizer heatmap/inspector (expects preview dimensions).
- **LLM Output:**
  > *"Swapping to a 768-dimensional model would cause dimension mismatch errors in FAISS (`IndexFlatIP` expects 384 dimensions). You would need to re-run `ingestion.embedder` to regenerate `vectors.npy`, update `vector_store.py`, and ensure `embedding_utils.py` uses the new model for incoming queries."*
- **Assessment:** **CORRECT (90%)**. Correctly pinpointed FAISS index dimension mismatch and regeneration requirements.

---

### Query 5: "Where are the Docker volume mounts defined and what data is shared between containers?"

- **Expected Answer:** `docker-compose.yml` defines two named volumes: `ollama_models` (persisting downloaded weights in `/root/.ollama`) and `shared_embeddings` (allowing `data_service` to write `vectors.npy` and `chunks.json` for `rag_service` to read).
- **LLM Output:**
  > *"In `docker-compose.yml`, `ollama_models` persists Ollama model weights, and `shared_embeddings` is mounted to `/shared_embeddings` so `data_service` can create the FAISS index and `rag_service` can read it."*
- **Assessment:** **CORRECT (100%)**.

---

## 3. Limitations of Current Setup & Motivation for Sourcegraph

While the current LLM handles high-level architectural queries well when files are explicitly in context, several fundamental limitations exist:

1. **Lack of AST & Symbol Navigation:** The LLM cannot resolve cross-file function definitions (e.g. `vector_store.search()`) without reading entire files into the prompt.
2. **Context Window Saturation:** Analyzing an entire repo of 20+ files simultaneously exceeds token windows and introduces needle-in-a-haystack degradation.
3. **No Dynamic Call-Graph Tracing:** The current RAG setup embeds text chunks, not code graphs or symbol hierarchies.

### Transition to Week 5 (Sourcegraph / Cody):
In Week 5, we will integrate **Sourcegraph code graph indexing**, enabling exact symbol definitions, cross-repository references, semantic code search, and precise dependency tracking.
