# Food Label Decoder

Food Label Decoder is a microservice-based system that decodes, parses, and analyzes nutritional ingredient labels from packaged foods using OCR, LLMs, and safety guardrails. It empowers consumers to detect allergens, identify harmful additives, track ingredient drift over time, and discover healthier food alternatives.

---

## Architecture Diagram

```text
                               +-----------------------------+
                               |     Client / Frontend       |
                               |        (Port 8501)          |
                               +--------------+--------------+
                                              |
                                              | HTTP POST /process
                                              v
+------------------------------------------------------------------------------------------+
|                                  ORCHESTRATOR SERVICE                                    |
|                                       (Port 8000)                                        |
+-----+--------------+--------------+--------------+--------------+--------------+---------+
      |              |              |              |              |              |
 (1)  |         (2)  |         (3)  |         (4)  |         (5)  |         (6)  |     (7) |
      v              v              v              v              v              v         v
+----------+   +-----------+   +----------+   +-----------+   +----------+   +-------+  +------+
|   OCR    |   | Guardrail |   |  Drift   |   | Retrieval |   | Analysis |   | Alt.  |  |Recipe|
| Service  |   |  Service  |   | Service  |   |  Service  |   | Service  |   |Service|  | Serv.|
|  (:8001) |   |  (:8002)  |   | (:8007)  |   |  (:8004)  |   | (:8003)  |   |(:8005)|  |(:8006|
+----+-----+   +-----+-----+   +----+-----+   +-----+-----+   +----+-----+   +---+---+  +---+--+
     |               |              |               |              |             |          |
 [Tesseract]   [Keywords]      [SQLite DB]     [Knowledge]     [Ollama]     [Suggestions][Custom]
   Engine        Filter         scans.db         Base        codellama         Engine    Recipes
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository_url>
cd food-label-decoder
```

### 2. Install Ollama & Pull the CodeLlama Model
Download and install [Ollama](https://ollama.com/), then pull the `codellama` model:
```bash
ollama pull codellama
```
Ensure Ollama is running at `http://localhost:11434`.

### 3. Install Dependencies per Service
Install requirements for each service:
```bash
pip install -r services/ocr_service/requirements.txt
pip install -r services/guardrail_service/requirements.txt
pip install -r services/drift_service/requirements.txt
pip install -r services/retrieval_service/requirements.txt
pip install -r services/analysis_service/requirements.txt
pip install -r services/alternative_service/requirements.txt
pip install -r services/recipe_service/requirements.txt
pip install -r orchestrator/requirements.txt
pip install -r frontend/requirements.txt
```

### 4. Run Services
Launch each service in a separate terminal window:
```bash
# 1. OCR Service (Port 8001)
python services/ocr_service/app.py

# 2. Guardrail Service (Port 8002)
python services/guardrail_service/app.py

# 3. Drift Service (Port 8007)
python services/drift_service/app.py

# 4. Retrieval Service (Port 8004)
python services/retrieval_service/app.py

# 5. Analysis Service (Port 8003)
python services/analysis_service/app.py

# 6. Alternative Service (Port 8005)
python services/alternative_service/app.py

# 7. Recipe Service (Port 8006)
python services/recipe_service/app.py

# 8. Orchestrator Service (Port 8000)
python orchestrator/app.py

# 9. Frontend Dashboard (Port 8501)
python frontend/app.py
```

---

## How to Test

You can test individual endpoints or the full orchestration pipeline with `curl`:

### Test OCR Service (`POST /ocr`)
```bash
# Using raw text
curl -X POST http://localhost:8001/ocr \
  -F "text=Ingredients: Whole grain oats, cane sugar, sea salt, 140mg sodium."

# Or upload an image file
curl -X POST http://localhost:8001/ocr \
  -F "file=@label_sample.png"
```

### Test Guardrail Service (`POST /guardrail`)
```bash
curl -X POST http://localhost:8002/guardrail \
  -H "Content-Type: application/json" \
  -d '{"text": "Ingredients: Whole grain oats, sugar, sodium, per 100g 12g protein."}'
```

### Test Analysis Service (`POST /analyse`)
```bash
curl -X POST http://localhost:8003/analyse \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Ingredients: Whole grain oats, high fructose corn syrup, Tartrazine (E102), Sodium Benzoate, Milk.",
    "context": "FDA and EU Food Additive Safety Standards"
  }'
```

### Test End-to-End Orchestrator (`POST /process`)
```bash
# Using JSON payload
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "Crunchy Morning Cereal",
    "text": "Ingredients: Whole grain oats, sugar, salt, sodium, per 100g 5g fat."
  }'

# Or using multipart image upload
curl -X POST http://localhost:8000/process \
  -F "product_name=Crunchy Morning Cereal" \
  -F "file=@label_sample.png"
```

---

## Environment Variables

The services support the following environment variables for configuration:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | HTTP endpoint of the local Ollama LLM service |
| `DB_PATH` | `database/food_label.db` | File path to the SQLite database for scans and drift tracking |
| `OCR_SERVICE_URL` | `http://ocr-service/ocr` | URL of the OCR microservice |
| `GUARDRAIL_SERVICE_URL`| `http://guardrail-service/guardrail` | URL of the Guardrail microservice |
| `DRIFT_SERVICE_URL` | `http://drift-service/scan` | URL of the Drift microservice |
| `RETRIEVAL_SERVICE_URL`| `http://retrieval-service/retrieve` | URL of the Retrieval microservice |
| `ANALYSIS_SERVICE_URL` | `http://analysis-service/analyse` | URL of the Analysis microservice |
| `ALTERNATIVE_SERVICE_URL`| `http://alternative-service/alternatives` | URL of the Alternative microservice |
| `RECIPE_SERVICE_URL` | `http://recipe-service/recipe` | URL of the Recipe microservice |

---

## Future

`docker compose up` multi-container deployment coming in **Ex5**.
