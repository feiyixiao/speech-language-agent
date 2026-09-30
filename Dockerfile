# API image for Cloud Run. ASR runs on Groq's hosted Whisper, so no torch/whisper here.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/hf TRACE_DIR=/tmp/traces

WORKDIR /app
COPY requirements.txt .
# CPU-only torch keeps the image ~1GB smaller (needed by the RAG embedding model)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt

COPY agent/ agent/
COPY api/ api/
COPY data/grammar_docs/ data/grammar_docs/
# bake the embedding model + vector index into the image: no download on cold start
RUN python -c "from agent.rag import get_vectorstore; get_vectorstore()"

RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8080
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
