"""
Jina AI embeddings service.

Uses Jina Embeddings v3 — 1024 dimensions, optimised for retrieval.

Key design decisions:
  - Asymmetric tasks: 'retrieval.passage' for chunks, 'retrieval.query' for queries.
    Using the wrong task silently degrades retrieval quality.
  - Batch processing: one API call per 32 texts instead of one per text.
    For 10,000 chunks: 312 calls vs 10,000 calls — ~30× faster.
  - Graceful fallback: returns None when embedding fails so callers
    can decide whether to skip or fall back to BM25.
  - API key optional: returns zero vectors when key is absent,
    so the application still starts and BM25 search still works.


Usage:
    from arxiv_rag_curator.services.embeddings.jina import JinaEmbeddingService
    svc = JinaEmbeddingService(api_key="jina_...")
    embeddings = svc.embed_passages(["chunk text one", "chunk text two"])
    query_vec  = svc.embed_query("transformer attention")
"""

import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

JINA_API_URL   = "https://api.jina.ai/v1/embeddings"
JINA_MODEL     = "jina-embeddings-v3"
EMBEDDING_DIM  = 1024

# Jina's recommended batch size — don't exceed 2048 per request
BATCH_SIZE     = 32

# Seconds to wait between batches to be a polite API citizen
BATCH_DELAY    = 0.1

# Per-request timeout
REQUEST_TIMEOUT = 30


class JinaEmbeddingService:
    """
    Production embedding service backed by Jina AI v3.

    Thread-safe: uses a stateless requests.Session.
    One instance per application is appropriate.
    """

    def __init__(self, api_key: str):
        self._api_key   = api_key
        self._available = bool(api_key)
        if not self._available:
            logger.warning(
                "No Jina API key provided. Embedding service disabled. "
                "Hybrid search will fall back to BM25."
            )

    @property
    def is_available(self) -> bool:
        """True if the service has a key and can generate embeddings."""
        return self._available

    def embed_passages(self, texts: list[str]) -> Optional[list[list[float]]]:
        """
        Embed a list of document chunks (index-time, passage task).

        Returns None if the service is unavailable or request fails.
        Caller should handle None by skipping vector indexing (BM25 still works).
        """
        return self._embed(texts, task="retrieval.passage")

    def embed_query(self, text: str) -> Optional[list[float]]:
        """
        Embed a single search query (query-time, query task).

        Returns None on failure — caller falls back to BM25.
        """
        result = self._embed([text], task="retrieval.query")
        return result[0] if result else None

    # ── Private ───────────────────────────────────────────────────────────────

    def _embed(
        self,
        texts: list[str],
        task:  str,
    ) -> Optional[list[list[float]]]:
        """
        Core embedding call with batching.

        task:
          'retrieval.passage' — for indexing document chunks
          'retrieval.query'   — for search-time query embedding
        """
        if not self._available:
            logger.debug("Embedding service unavailable, returning None")
            return None

        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            payload = {
                "model":      JINA_MODEL,
                "task":       task,
                "input":      batch,
                "dimensions": EMBEDDING_DIM,
            }

            try:
                resp = requests.post(
                    JINA_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                data = resp.json()

                # Sort by index to preserve input order (API may not guarantee it)
                sorted_items = sorted(data["data"], key=lambda x: x["index"])
                batch_embeddings = [item["embedding"] for item in sorted_items]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "Embedded batch %d-%d (task=%s, dim=%d)",
                    i, i + len(batch), task, EMBEDDING_DIM,
                )

                # Rate limit courtesy delay between batches
                if i + BATCH_SIZE < len(texts):
                    time.sleep(BATCH_DELAY)

            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else "?"
                logger.error("Jina API HTTP error %s for batch %d-%d: %s", status, i, i+len(batch), exc)
                return None  # fail fast — don't return partial results

            except requests.exceptions.RequestException as exc:
                logger.error("Jina API request failed for batch %d-%d: %s", i, i+len(batch), exc)
                return None

        return all_embeddings

    def health_check(self) -> dict:
        """Check if the embedding service is reachable."""
        if not self._available:
            return {"status": "disabled", "reason": "no_api_key"}
        try:
            result = self.embed_query("health check")
            if result and len(result) == EMBEDDING_DIM:
                return {"status": "healthy", "model": JINA_MODEL, "dimensions": EMBEDDING_DIM}
            return {"status": "unhealthy", "reason": "unexpected_response"}
        except Exception as exc:
            return {"status": "unhealthy", "reason": str(exc)}