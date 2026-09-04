FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# CHROMA_PATH can be overridden via env var (default set in docker-compose)
ENV CHROMA_PATH=/app/chroma_db
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
