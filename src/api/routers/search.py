"""
FastAPI search router.

Endpoints:
  POST /api/v1/search           → BM25 full-text search
  GET  /api/v1/search/stats     → index statistics
  GET  /api/v1/search/{arxiv_id} → single paper lookup by ID

Design:
  - Router depends on SearchService via FastAPI's dependency injection
  - HTTP concerns (status codes, error responses) handled here
  - No business logic — routing and schema translation only
  - Errors from SearchService propagate as HTTP 500 with a clear message

The search endpoint uses POST (not GET) because:
  - Filters can be complex nested objects (not easily representable as query params)
  - Request body avoids URL length limits for long queries
  - Consistent with how production search APIs are typically designed
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ...services.opensearch.factory import make_search_service
from ...services.opensearch.service import SearchService
from ..schemas.search import SearchRequest, SearchResponse, StatsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


# ── Dependency ────────────────────────────────────────────────────────────────

def get_search_service() -> SearchService:
    """
    FastAPI dependency that provides a SearchService.

    In production this would be a singleton injected at startup.
    For simplicity here, make_search_service() is called per request
    (it's fast because ensure_index() is a no-op if index exists).

    A better pattern: create once in app lifespan, store on app.state.
    We'll refactor this when we wire the full app together.
    """
    return make_search_service()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=SearchResponse,
    summary="Search papers",
    description=(
        "Full-text BM25 search across paper titles, abstracts, and full text. "
        "Supports category filtering, date range filtering, and pagination."
    ),
)
async def search_papers(
    request: SearchRequest,
    svc: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Search papers using BM25 keyword search.

    Field boosting: title (3×) > abstract (2×) > full_text (1×).
    Results include highlighted snippets showing where the query matched.
    """
    logger.info(
        "Search request: query=%r categories=%s page=%d",
        request.query, request.categories, request.page,
    )

    try:
        result = svc.search(
            text=request.query,
            categories=request.categories,
            date_from=request.date_from,
            date_to=request.date_to,
            page=request.page,
            page_size=request.page_size,
            sort_by=request.sort_by,
        )

        logger.info(
            "Search completed: query=%r total=%d took=%dms",
            request.query, result.total, result.took_ms,
        )

        return SearchResponse.from_service_result(result)

    except Exception as exc:
        logger.error("Search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Index statistics",
    description="Returns document counts, index size, and category breakdown.",
)
async def get_stats(
    svc: SearchService = Depends(get_search_service),
) -> StatsResponse:
    """Return statistics about the search index."""
    stats = svc.get_stats()
    if "error" in stats:
        raise HTTPException(status_code=503, detail=stats["error"])
    return StatsResponse(**stats)


@router.get(
    "/{arxiv_id}",
    response_model=Optional[SearchResponse],
    summary="Get paper by arXiv ID",
    description="Fetch a single paper by its arXiv ID (e.g. '2301.00001').",
)
async def get_paper(
    arxiv_id: str,
    svc: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Look up a single paper by its arXiv ID."""
    try:
        # Use term query on keyword field — exact match, no scoring
        from ...services.opensearch.index_config import INDEX_NAME
        raw = svc._client.get(index=INDEX_NAME, id=arxiv_id, ignore=404)

        if not raw.get("found"):
            raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

        # Wrap in SearchResponse shape for consistency
        src = raw["_source"]
        from ...services.opensearch.service import SearchHit, SearchResult
        hit = SearchHit(
            arxiv_id    = src.get("arxiv_id", arxiv_id),
            title       = src.get("title", ""),
            abstract    = src.get("abstract", ""),
            authors     = src.get("authors", []),
            categories  = src.get("categories", []),
            published_at= src.get("published_at"),
            score       = 1.0,
            pdf_parsed  = src.get("pdf_parsed", False),
        )
        result = SearchResult(
            hits=[hit], total=1, took_ms=0,
            query=arxiv_id, page=0, page_size=1,
        )
        return SearchResponse.from_service_result(result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Paper lookup failed for %s: %s", arxiv_id, exc)
        raise HTTPException(status_code=500, detail=str(exc))