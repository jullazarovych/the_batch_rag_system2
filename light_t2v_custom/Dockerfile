FROM python:3.11-slim

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn torch transformers sentence-transformers
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
WORKDIR /app
COPY app_t2v.py /app/app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
