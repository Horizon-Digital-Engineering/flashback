#!/usr/bin/env python3
"""Download NLP models required by the sidecar.

Run once before starting the server, or bake into the Docker build step.

    python download_models.py
"""
from __future__ import annotations

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SPACY_MODEL = "en_core_web_sm"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def download_spacy_model(model: str = SPACY_MODEL) -> None:
    """Download *model* via ``python -m spacy download`` if not already installed."""
    try:
        import spacy

        spacy.load(model)
        logger.info("spaCy model '%s' already installed — skipping download", model)
    except OSError:
        logger.info("Downloading spaCy model '%s'…", model)
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(result.stdout.strip() or "Done")


def download_sentence_transformer(model: str = EMBEDDING_MODEL) -> None:
    """Pull *model* into the sentence-transformers local cache."""
    logger.info("Pre-fetching sentence-transformer model '%s'…", model)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model)
    logger.info("Model '%s' cached", model)


if __name__ == "__main__":
    download_spacy_model()
    download_sentence_transformer()
    logger.info("All models ready")
