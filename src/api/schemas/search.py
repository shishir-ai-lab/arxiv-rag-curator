"""
Pydantic schemas for the search API.

These define the exact shape of:
  - SearchRequest:  what the client POSTs to /api/v1/search
  - SearchResponse: what the API returns

Keeping schemas separate from the service layer means:
  - API contract is versioned independently of business logic
  - FastAPI generates accurate OpenAPI docs automatically
  - Request validation happens at the HTTP boundary, not deep in services
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Request schema ────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    """
    Search request body.

    All fields except 'query' are optional — sensible defaults apply.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Full-text search query",
        examples=["transformer attention mechanism"],
    )
    categories: Optional[list[str]] = Field(
        default=None,
        description="Filter to specific arXiv categories e.g. ['cs.AI', 'cs.LG']",
        examples=[["cs.AI", "cs.LG"]],
    )
    date_from: Optional[date] = Field(
        default=None,
        description="Only return papers published on or after this date (YYYY-MM-DD)",
    )
    date_to: Optional[date] = Field(
        default=None,
        description="Only return papers published on or before this date (YYYY-MM-DD)",
    )
    page: int = Field(
        default=0,
        ge=0,
        description="Zero-based page number for pagination",
    )
    page_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results per page (1–100)",
    )
    sort_by: str = Field(
        default="relevance",
        description="Sort order: 'relevance' | 'date_desc' | 'date_asc'",
    )

    @field_validator("sort_by")
    @classmethod
    def validate_sort_by(cls, v: str) -> str:
        allowed = {"relevance", "date_desc", "date_asc"}
        if v not in allowed:
            raise ValueError(f"sort_by must be one of {allowed}")
        return v

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        return v.strip()


# ── Response schemas ──────────────────────────────────────────────────────────

class PaperHit(BaseModel):
    """A single paper in search results."""
    arxiv_id:    str
    title:       str
    abstract:    str
    authors:     list[str]
    categories:  list[str]
    published_at: Optional[str]
    score:       float
    pdf_parsed:  bool
    # Highlighted snippets: field_name → list of HTML snippets with <mark> tags
    highlights:  dict[str, list[str]] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Full search response with results and pagination metadata."""
    query:      str
    total:      int            = Field(description="Total matching documents")
    page:       int            = Field(description="Current page (zero-based)")
    page_size:  int
    has_more:   bool           = Field(description="Whether more pages are available")
    took_ms:    int            = Field(description="OpenSearch execution time in ms")
    hits:       list[PaperHit]

    @classmethod
    def from_service_result(cls, result) -> "SearchResponse":
        """Convert a SearchService SearchResult into an API SearchResponse."""
        return cls(
            query=result.query,
            total=result.total,
            page=result.page,
            page_size=result.page_size,
            has_more=result.has_more,
            took_ms=result.took_ms,
            hits=[
                PaperHit(
                    arxiv_id    = h.arxiv_id,
                    title       = h.title,
                    abstract    = h.abstract,
                    authors     = h.authors,
                    categories  = h.categories,
                    published_at= h.published_at,
                    score       = h.score,
                    pdf_parsed  = h.pdf_parsed,
                    highlights  = h.highlights,
                )
                for h in result.hits
            ],
        )


class StatsResponse(BaseModel):
    """Index statistics for the /search/stats endpoint."""
    total_documents:  int
    pdf_parsed_count: int
    index_size_mb:    float
    categories:       dict[str, int]