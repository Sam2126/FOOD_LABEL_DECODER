FROM python:3.11-slim

WORKDIR /app

# curl is needed for the container HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion/ ./ingestion/
COPY knowledge_base/ ./knowledge_base/
COPY main.py .

EXPOSE 8003

HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=5 \
    CMD curl -f http://localhost:8003/health/ready || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
