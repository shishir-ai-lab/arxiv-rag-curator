"""
Factory for the embeddings service.

Reads JINA_API_KEY from settings. If the key is absent, returns a
JinaEmbeddingService with is_available=False — callers check this
before attempting to embed, falling back to BM25-only search.

Usage:
    from arxiv_rag_curator.services.embeddings.factory import make_embeddings_service
    svc = make_embeddings_service()
    if svc.is_available:
        vec = svc.embed_query("attention mechanism")
"""

from ...core.config import settings
from .jina import JinaEmbeddingService


def make_embeddings_service() -> JinaEmbeddingService:
    """
    Create a Jina embeddings service from application settings.

    If JINA_API_KEY is not set, returns a service instance where
    is_available=False — hybrid search will gracefully fall back to BM25.
    """
    return JinaEmbeddingService(api_key=settings.jina_api_key)