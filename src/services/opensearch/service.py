"""
OpenSearch search service.

Responsibilities:
  - Manage the OpenSearch client lifecycle
  - Execute queries built by QueryBuilder
  - Normalise raw OpenSearch responses into clean SearchResult objects
  - Handle index initialisation (create if not exists)
  - Bulk indexing of papers from PostgreSQL

What this is NOT responsible for:
  - Building queries (that's QueryBuilder)
  - Fetching papers from PostgreSQL (that's the Indexer)
  - HTTP routing (that's the FastAPI router)

Usage:
    from arxiv_rag_curator.services.opensearch.factory import make_search_service
    svc = make_search_service()
    results = svc.search("transformer attention", categories=["cs.AI"])
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from opensearchpy import OpenSearch, OpenSearchException
from opensearchpy.helpers import bulk

from ...core.config import settings
from .index_config import INDEX_CONFIG, INDEX_NAME
from .query_builder import QueryBuilder

logger = logging.getLogger(__name__)


# ── Result schema ─────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    """A single search result, normalised from OpenSearch's raw response."""
    arxiv_id:    str
    title:       str
    abstract:    str
    authors:     list[str]
    categories:  list[str]
    published_at: Optional[str]
    score:       float
    pdf_parsed:  bool = False
    highlights:  dict[str, list[str]] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Full response from a search operation."""
    hits:      list[SearchHit]
    total:     int         # total matching docs (may be > len(hits) due to pagination)
    took_ms:   int         # OpenSearch execution time
    query:     str         # the original query text (for logging/display)
    page:      int = 0
    page_size: int = 10

    @property
    def has_more(self) -> bool:
        return (self.page + 1) * self.page_size < self.total


@dataclass
class IndexResult:
    """Result of a bulk indexing operation."""
    indexed:  int
    errors:   int
    total:    int

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.indexed / self.total * 100


# ── Service ───────────────────────────────────────────────────────────────────

class SearchService:
    """
    Production OpenSearch search service.

    Thread-safe: the OpenSearch client manages its own connection pool.
    One SearchService instance per application is fine.
    """

    def __init__(self, client: OpenSearch):
        self._client  = client
        self._builder = QueryBuilder()

    # ── Search operations ─────────────────────────────────────────────────────

    def search(
        self,
        text: str,
        categories:  Optional[list[str]] = None,
        date_from:   Optional[date] = None,
        date_to:     Optional[date] = None,
        page:        int = 0,
        page_size:   int = 10,
        sort_by:     str = "relevance",
    ) -> SearchResult:
        """
        Full-text BM25 search with optional filters.

        Args:
            text:       user's search query
            categories: restrict to these arXiv categories
            date_from:  published on or after this date
            date_to:    published on or before this date
            page:       zero-based page number
            page_size:  results per page (max 100)
            sort_by:    'relevance' | 'date_desc' | 'date_asc'

        Returns:
            SearchResult with normalised hits and pagination metadata.
        """
        query = self._builder.bm25(
            text=text,
            categories=categories,
            date_from=date_from,
            date_to=date_to,
            from_=page * page_size,
            size=page_size,
            sort_by=sort_by,
        )

        try:
            response = self._client.search(index=INDEX_NAME, body=query)
            return self._normalise_response(response, text, page, page_size)
        except OpenSearchException as exc:
            logger.error("Search failed for query '%s': %s", text, exc)
            raise

    def search_by_category(
        self,
        categories: list[str],
        page: int = 0,
        page_size: int = 20,
        sort_by: str = "date_desc",
    ) -> SearchResult:
        """Return papers in given categories, sorted by date. No text search."""
        query = self._builder.by_category(
            categories=categories,
            from_=page * page_size,
            size=page_size,
            sort_by=sort_by,
        )
        response = self._client.search(index=INDEX_NAME, body=query)
        return self._normalise_response(response, f"category:{categories}", page, page_size)

    def get_stats(self) -> dict:
        """Index statistics and category breakdown for the /health endpoint."""
        try:
            agg_query = self._builder.count_by_category()
            response  = self._client.search(index=INDEX_NAME, body=agg_query)
            aggs      = response["aggregations"]

            category_counts = {
                b["key"]: b["doc_count"]
                for b in aggs["by_category"]["buckets"]
            }

            total_resp  = self._client.count(index=INDEX_NAME)
            index_stats = self._client.indices.stats(index=INDEX_NAME)
            size_bytes  = index_stats["_all"]["primaries"]["store"]["size_in_bytes"]

            return {
                "total_documents":  total_resp["count"],
                "pdf_parsed_count": aggs["pdf_parsed_count"]["doc_count"],
                "index_size_mb":    round(size_bytes / (1024 * 1024), 2),
                "categories":       category_counts,
            }
        except Exception as exc:
            logger.error("Stats query failed: %s", exc)
            return {"error": str(exc)}

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_paper(self, paper_doc: dict) -> bool:
        """
        Index a single paper document.
        Returns True on success, False on failure.
        Uses arxiv_id as the document _id for upsert semantics.
        """
        try:
            arxiv_id = paper_doc["arxiv_id"]
            self._client.index(
                index=INDEX_NAME,
                id=arxiv_id,
                body=paper_doc,
            )
            return True
        except OpenSearchException as exc:
            logger.error("Failed to index paper %s: %s", paper_doc.get("arxiv_id"), exc)
            return False

    def bulk_index(self, paper_docs: list[dict]) -> IndexResult:
        """
        Index a batch of papers using the bulk API.

        Bulk vs individual:
          100 papers × single index → ~100 HTTP round-trips, seconds
          100 papers × bulk         → 1 HTTP round-trip, milliseconds

        Using _op_type='index' means each operation is an upsert:
        existing documents are overwritten, not duplicated.
        """
        if not paper_docs:
            return IndexResult(indexed=0, errors=0, total=0)

        actions = [
            {
                "_op_type": "index",      # upsert: safe to re-run
                "_index":   INDEX_NAME,
                "_id":      doc["arxiv_id"],
                "_source":  doc,
            }
            for doc in paper_docs
        ]

        try:
            success, failed = bulk(
                self._client,
                actions,
                raise_on_error=False,
                stats_only=False,
            )
            error_count = len(failed) if isinstance(failed, list) else int(failed)
            result = IndexResult(
                indexed=success,
                errors=error_count,
                total=len(paper_docs),
            )
            logger.info(
                "Bulk indexed %d/%d papers (%.1f%% success)",
                result.indexed, result.total, result.success_rate,
            )
            return result

        except Exception as exc:
            logger.error("Bulk index failed: %s", exc)
            return IndexResult(indexed=0, errors=len(paper_docs), total=len(paper_docs))

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def ensure_index(self) -> None:
        """
        Create the index if it does not exist.
        Safe to call on every app startup — no-op if index already exists.
        Does NOT update mappings on an existing index (requires reindex).
        """
        if self._client.indices.exists(index=INDEX_NAME):
            logger.debug("Index '%s' already exists", INDEX_NAME)
            return

        self._client.indices.create(index=INDEX_NAME, body=INDEX_CONFIG)
        logger.info("Index '%s' created", INDEX_NAME)

    def check_health(self) -> dict:
        """Return health status of the OpenSearch cluster."""
        try:
            health = self._client.cluster.health()
            return {
                "status":         "healthy",
                "cluster_status": health["status"],  # green / yellow / red
            }
        except Exception as exc:
            return {"status": "unhealthy", "error": str(exc)}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _normalise_response(
        self,
        response: dict,
        query_text: str,
        page: int,
        page_size: int,
    ) -> SearchResult:
        """Convert a raw OpenSearch response into a SearchResult."""
        raw_hits = response["hits"]["hits"]
        total    = response["hits"]["total"]["value"]
        took_ms  = response["took"]

        hits = [self._normalise_hit(h) for h in raw_hits]

        return SearchResult(
            hits=hits,
            total=total,
            took_ms=took_ms,
            query=query_text,
            page=page,
            page_size=page_size,
        )

    def _normalise_hit(self, raw: dict) -> SearchHit:
        """Convert a single OpenSearch hit into a SearchHit."""
        src = raw.get("_source", {})
        return SearchHit(
            arxiv_id    = src.get("arxiv_id", raw["_id"]),
            title       = src.get("title", ""),
            abstract    = src.get("abstract", ""),
            authors     = src.get("authors", []),
            categories  = src.get("categories", []),
            published_at= src.get("published_at"),
            score       = raw.get("_score") or 0.0,
            pdf_parsed  = src.get("pdf_parsed", False),
            highlights  = raw.get("highlight", {}),
        )