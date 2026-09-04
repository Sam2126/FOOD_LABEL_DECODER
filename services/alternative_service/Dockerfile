FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV CHROMA_PATH=/app/chroma_db
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8005"]
