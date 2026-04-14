"""
Factory functions for OpenSearch services.

Mirrors the pattern from services/factory.py — a single place to wire
dependencies and configure each service for the current environment.

Usage:
    from arxiv_rag_curator.services.opensearch.factory import (
        make_search_service,
        make_paper_indexer,
    )

    svc     = make_search_service()
    indexer = make_paper_indexer()
"""

from opensearchpy import OpenSearch

from ...core.config import settings
from ...core.database import get_db
from .indexer import PaperIndexer
from .service import SearchService


def make_opensearch_client() -> OpenSearch:
    """Create a configured OpenSearch client."""
    return OpenSearch(
        hosts=[{
            "host": settings.opensearch_host,
            "port": settings.opensearch_port,
        }],
        http_compress=True,     # gzip request/response bodies
        use_ssl=False,          # no TLS in dev
        verify_certs=False,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )


def make_search_service() -> SearchService:
    """
    Create a fully configured SearchService.

    Ensures the papers index exists before returning.
    Safe to call on every app startup.
    """
    client = make_opensearch_client()
    service = SearchService(client=client)
    service.ensure_index()
    return service


def make_paper_indexer() -> PaperIndexer:
    """
    Create a PaperIndexer wired to the search service and DB.

    Used by the Airflow DAG's indexing stage and by MetadataFetcher
    for write-through indexing after each paper save.
    """
    return PaperIndexer(
        search_service=make_search_service(),
        db=get_db,
    )