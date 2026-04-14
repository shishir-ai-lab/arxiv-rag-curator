"""
Hybrid search router.

Single endpoint: POST /api/v1/hybrid-search/
  - use_hybrid=true  → BM25 + kNN with RRF (semantic understanding)
  - use_hybrid=false → BM25 only (fast, exact term matching)

The endpoint auto-detects the search mode and returns search_mode
in the response, so clients can see which path was actually used
(hybrid may fall back to BM25 if the embedding service is unavailable).

Why POST instead of GET?
  - Request body avoids URL length limits for long queries
  - Filter objects (categories, date ranges) are cleaner in JSON than query params
  - Consistent with how production search APIs are typically designed
"""

import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from ...services.embeddings.factory import make_embeddings_service
from ...services.opensearch.factory import make_opensearch_client
from ...services.opensearch.hybrid_service import HybridSearchResult, HybridSearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/hybrid-search", tags=["hybrid-search"])


# ── Request / Response schemas ────────────────────────────────────────────────

class HybridSearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        examples=["transformer attention mechanism"],
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use BM25 + kNN hybrid search. Falls back to BM25 if embeddings unavailable.",
    )
    categories: Optional[list[str]] = Field(
        default=None,
        description="Filter to these arXiv categories e.g. ['cs.AI', 'cs.LG']",
    )
    date_from: Optional[date] = Field(
        default=None,
        description="Only return papers on or after this date (YYYY-MM-DD)",
    )
    date_to: Optional[date] = Field(
        default=None,
        description="Only return papers on or before this date (YYYY-MM-DD)",
    )
    page:      int = Field(default=0,  ge=0, description="Zero-based page number")
    page_size: int = Field(default=10, ge=1, le=100)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


class ChunkHitResponse(BaseModel):
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


class HybridSearchResponse(BaseModel):
    query:       str
    total:       int
    page:        int
    page_size:   int
    has_more:    bool
    took_ms:     int
    search_mode: str     # 'hybrid' | 'bm25' — which path was used
    hits:        list[ChunkHitResponse]

    @classmethod
    def from_service_result(cls, result: HybridSearchResult) -> "HybridSearchResponse":
        return cls(
            query       = result.query,
            total       = result.total,
            page        = result.page,
            page_size   = result.page_size,
            has_more    = result.has_more,
            took_ms     = result.took_ms,
            search_mode = result.search_mode,
            hits=[
                ChunkHitResponse(
                    arxiv_id     = h.arxiv_id,
                    chunk_id     = h.chunk_id,
                    chunk_text   = h.chunk_text,
                    section_name = h.section_name,
                    chunk_index  = h.chunk_index,
                    title        = h.title,
                    abstract     = h.abstract,
                    authors      = h.authors,
                    categories   = h.categories,
                    published_at = h.published_at,
                    score        = h.score,
                )
                for h in result.hits
            ],
        )


# ── Dependency ────────────────────────────────────────────────────────────────

def get_hybrid_service() -> HybridSearchService:
    """Provide a HybridSearchService with injected dependencies."""
    return HybridSearchService(
        os_client      = make_opensearch_client(),
        embeddings_svc = make_embeddings_service(),
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=HybridSearchResponse,
    summary="Hybrid search (BM25 + semantic)",
    description=(
        "Search paper chunks using BM25 keyword search combined with "
        "kNN vector similarity via Reciprocal Rank Fusion (RRF). "
        "Set use_hybrid=false to use BM25 only (faster, exact-term matching). "
        "The response includes search_mode indicating which path was used."
    ),
)
async def hybrid_search(
    request: HybridSearchRequest,
    svc:     HybridSearchService = Depends(get_hybrid_service),
) -> HybridSearchResponse:
    """
    Unified search endpoint supporting BM25-only and hybrid modes.

    Field boosting in hybrid mode:
      chunk_text^2, title^3, abstract^1 (BM25 component)
      kNN on 1024-dim Jina v3 embeddings (vector component)
      RRF fusion with k=60 combines both rankings
    """
    logger.info(
        "Hybrid search: query=%r use_hybrid=%s categories=%s page=%d",
        request.query, request.use_hybrid, request.categories, request.page,
    )

    try:
        result = svc.search(
            query      = request.query,
            use_hybrid = request.use_hybrid,
            categories = request.categories,
            date_from  = request.date_from,
            date_to    = request.date_to,
            page       = request.page,
            page_size  = request.page_size,
        )

        logger.info(
            "Search done: query=%r mode=%s total=%d took=%dms",
            request.query, result.search_mode, result.total, result.took_ms,
        )

        return HybridSearchResponse.from_service_result(result)

    except Exception as exc:
        logger.error("Hybrid search error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {exc}")


@router.get(
    "/health",
    summary="Hybrid search service health",
    description="Check OpenSearch and embedding service health.",
)
async def health(
    svc: HybridSearchService = Depends(get_hybrid_service),
) -> dict:
    return svc.health_check()