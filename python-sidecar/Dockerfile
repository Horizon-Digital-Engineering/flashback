FROM python:3.12-slim

WORKDIR /app

# Build tools needed by some spaCy / tokenizers transitive deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch first to keep the image lean; sentence-transformers
# will detect it and skip the GPU variant.
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Download NLP models at build time so the container starts instantly.
COPY download_models.py .
RUN python download_models.py

# Copy application source last to maximise layer cache reuse.
COPY schemas.py embedder.py extractor.py main.py ./

EXPOSE 8081

CMD ["python", "main.py"]
