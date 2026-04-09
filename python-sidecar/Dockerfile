# Stage 1: download models
FROM python:3.12-slim AS model-download

WORKDIR /models

RUN pip install --no-cache-dir sentence-transformers==3.3.1 spacy==3.8.3

# Download sentence-transformers model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Stage 2: runtime image
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy pre-downloaded models from stage 1
COPY --from=model-download /root/.cache/huggingface /root/.cache/huggingface
COPY --from=model-download /usr/local/lib/python3.12/site-packages/en_core_web_sm \
    /usr/local/lib/python3.12/site-packages/en_core_web_sm

COPY app/ ./app/

EXPOSE 8081

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
