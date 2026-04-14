"""
Hybrid search service.

Combines BM25 keyword search with kNN vector search using
Reciprocal Rank Fusion (RRF) via OpenSearch's native hybrid query.

The key design: graceful degradation.
  - If embedding service is available and healthy → hybrid search
  - If embedding fails or key is missing → BM25 fallback
  - The search_mode in the response tells callers which path was taken

Search flow:
  1. Embed the user's query with Jina (retrieval.query task)
  2. Run OpenSearch hybrid query (BM25 + kNN in one request)
  3. OpenSearch applies RRF via the normalization pipeline
  4. Normalise results into HybridSearchResult
  5. On any embedding failure: run BM25-only search instead

Usage:
    from arxiv_rag_curator.services.opensearch.hybrid_service import HybridSearchService
    svc = HybridSearchService(os_client, embeddings_svc)
    result = svc.search("retrieval augmented generation", use_hybrid=True)
    print(result.search_mode)  # 'hybrid' or 'bm25'
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from opensearchpy import OpenSearch

from ..embeddings.jina import JinaEmbeddingService
from .chunks_index_config import CHUNKS_INDEX_NAME, RRF_PIPELINE_ID

logger = logging.getLogger(__name__)


# ── Result schemas ────────────────────────────────────────────────────────────

@dataclass
class ChunkHit:
    """A single chunk result from hybrid search."""
    arxiv_id:     str
    chunk_id:     str
    chunk_text:   str
    section_name: str
    chunk_index:  int
    title:        str
    abstract:     str
    authors:      list[str]
    categories:   list[str]
    published_at: Optional[str]
    score:        float


@dataclass
class HybridSearchResult:
    """Complete response from a hybrid search operation."""
    query:       str
    hits:        list[ChunkHit]
    total:       int
    took_ms:     int
    search_mode: str   # 'hybrid' | 'bm25'
    page:        int = 0
    page_size:   int = 10

    @property
    def has_more(self) -> bool:
        return (self.page + 1) * self.page_size < self.total


# ── Service ───────────────────────────────────────────────────────────────────

class HybridSearchService:
    """
    Hybrid search: BM25 + kNN with RRF, graceful BM25 fallback.
    """

    def __init__(
        self,
        os_client:      OpenSearch,
        embeddings_svc: JinaEmbeddingService,
    ):
        self._client  = os_client
        self._embed   = embeddings_svc

    def search(
        self,
        query:       str,
        use_hybrid:  bool = True,
        categories:  Optional[list[str]] = None,
        date_from:   Optional[date] = None,
        date_to:     Optional[date] = None,
        page:        int = 0,
        page_size:   int = 10,
    ) -> HybridSearchResult:
        """
        Search chunks, combining BM25 and kNN with RRF.

        Args:
            query:      the user's search query
            use_hybrid: attempt hybrid (True) or force BM25 (False)
            categories: filter to these arXiv categories
            date_from:  filter papers published on or after this date
            date_to:    filter papers published on or before this date
            page:       zero-based page number
            page_size:  results per page

        Returns:
            HybridSearchResult with search_mode indicating which path was used.
        """
        from_  = page * page_size

        if use_hybrid and self._embed.is_available:
            try:
                return self._hybrid_search(query, categories, date_from, date_to, from_, page_size, page)
            except Exception as exc:
                logger.warning("Hybrid search failed (%s), falling back to BM25", exc)

        # BM25 fallback
        return self._bm25_search(query, categories, date_from, date_to, from_, page_size, page)

    def health_check(self) -> dict:
        """Return health of both OpenSearch and embedding service."""
        try:
            count = self._client.count(index=CHUNKS_INDEX_NAME)["count"]
            os_ok = True
        except Exception:
            count = 0
            os_ok = False

        embed_health = self._embed.health_check() if hasattr(self._embed, "health_check") else {}

        return {
            "opensearch":        "healthy" if os_ok else "unhealthy",
            "chunks_indexed":    count,
            "embedding_service": embed_health.get("status", "unknown"),
        }

    # ── Private: search paths ─────────────────────────────────────────────────

    def _hybrid_search(
        self,
        query:      str,
        categories: Optional[list[str]],
        date_from:  Optional[date],
        date_to:    Optional[date],
        from_:      int,
        size:       int,
        page:       int,
    ) -> HybridSearchResult:
        """Run hybrid BM25 + kNN search via OpenSearch's hybrid query."""
        # Embed the query — use retrieval.query (NOT passage) task
        query_vec = self._embed.embed_query(query)
        if query_vec is None:
            raise RuntimeError("Embedding service returned None for query")

        # Build filter clauses
        filters = self._build_filters(categories, date_from, date_to)

        # BM25 sub-query
        bm25_clause: dict = {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query":    query,
                        "fields":   ["chunk_text^2", "title^3", "abstract"],
                        "type":     "best_fields",
                        "operator": "or",
                        "fuzziness": "AUTO",
                    }
                }],
                "filter": filters,
            }
        }

        # kNN sub-query
        knn_clause: dict = {
            "knn": {
                "embedding": {
                    "vector": query_vec,
                    "k": size * 2,   # over-fetch; RRF trims to final size
                    "filter": {"bool": {"filter": filters}} if filters else None,
                }
            }
        }
        # Clean up None filter from kNN clause
        if knn_clause["knn"]["embedding"]["filter"] is None:
            del knn_clause["knn"]["embedding"]["filter"]

        body = {
            "from":  from_,
            "size":  size,
            "query": {
                "hybrid": {"queries": [bm25_clause, knn_clause]}
            },
            "_source": {"excludes": ["embedding"]},
        }

        response = self._client.search(
            index=CHUNKS_INDEX_NAME,
            body=body,
            params={"search_pipeline": RRF_PIPELINE_ID},
        )

        return self._normalise(response, query, page, size, search_mode="hybrid")

    def _bm25_search(
        self,
        query:      str,
        categories: Optional[list[str]],
        date_from:  Optional[date],
        date_to:    Optional[date],
        from_:      int,
        size:       int,
        page:       int,
    ) -> HybridSearchResult:
        """Pure BM25 search on the chunks index."""
        filters = self._build_filters(categories, date_from, date_to)

        body = {
            "from": from_,
            "size": size,
            "query": {
                "bool": {
                    "must": [{
                        "multi_match": {
                            "query":    query,
                            "fields":   ["chunk_text^2", "title^3", "abstract"],
                            "type":     "best_fields",
                            "operator": "or",
                            "fuzziness": "AUTO",
                        }
                    }],
                    "filter": filters,
                }
            },
            "_source": {"excludes": ["embedding"]},
        }

        response = self._client.search(index=CHUNKS_INDEX_NAME, body=body)
        return self._normalise(response, query, page, size, search_mode="bm25")

    # ── Private: helpers ──────────────────────────────────────────────────────

    def _build_filters(
        self,
        categories: Optional[list[str]],
        date_from:  Optional[date],
        date_to:    Optional[date],
    ) -> list:
        filters = []
        if categories:
            filters.append({"terms": {"categories": categories}})
        if date_from or date_to:
            date_range: dict = {}
            if date_from:
                date_range["gte"] = date_from.isoformat()
            if date_to:
                date_range["lte"] = date_to.isoformat()
            filters.append({"range": {"published_at": date_range}})
        return filters

    def _normalise(
        self,
        response:    dict,
        query:       str,
        page:        int,
        page_size:   int,
        search_mode: str,
    ) -> HybridSearchResult:
        raw_hits = response["hits"]["hits"]
        total    = response["hits"]["total"]["value"]
        took_ms  = response["took"]

        hits = [self._normalise_hit(h) for h in raw_hits]
        return HybridSearchResult(
            query=query,
            hits=hits,
            total=total,
            took_ms=took_ms,
            search_mode=search_mode,
            page=page,
            page_size=page_size,
        )

    def _normalise_hit(self, raw: dict) -> ChunkHit:
        src = raw.get("_source", {})
        return ChunkHit(
            arxiv_id     = src.get("arxiv_id", ""),
            chunk_id     = src.get("chunk_id", raw["_id"]),
            chunk_text   = src.get("chunk_text", ""),
            section_name = src.get("section_name", ""),
            chunk_index  = src.get("chunk_index", 0),
            title        = src.get("title", ""),
            abstract     = src.get("abstract", ""),
            authors      = src.get("authors", []),
            categories   = src.get("categories", []),
            published_at = src.get("published_at"),
            score        = raw.get("_score") or 0.0,
        )