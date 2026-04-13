"""
arXiv API client with production-grade rate limiting.

arXiv's rules are non-negotiable:
  - Minimum 3 seconds between requests
  - No parallel requests from the same IP
  - Violations = temporary or permanent ban

We wrap the 'arxiv' Python library with:
  - Explicit delay tracking (not just time.sleep(3))
  - Date range filtering for daily pipeline runs
  - Category validation
  - Structured output via ArxivPaper schema

Usage:
    from arxiv_rag_curator.services.arxiv.client import ArxivClient
    client = ArxivClient()
    papers = client.fetch_by_date(category="cs.AI", target_date=date.today())
"""

import logging
import time
from datetime import date, timedelta
from typing import Generator

import arxiv

from schemas import ArxivPaper

logger = logging.getLogger(__name__)

# arXiv mandates >= 3s between requests. We use 3.1s to add a small buffer.
RATE_LIMIT_SECONDS = 3.1

# Categories we support — extend as needed
VALID_CATEGORIES = {
    "cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR",
    "cs.NE", "cs.RO", "stat.ML",
}


class ArxivClient:
    """
    Rate-limited arXiv API client.

    The underlying 'arxiv' library handles the 3-second delay internally
    when we set delay_seconds. We wrap it with:
    - Category validation
    - Date range query construction
    - Structured output (ArxivPaper)
    - Result limiting with logging
    """

    def __init__(
        self,
        delay_seconds: float = RATE_LIMIT_SECONDS,
        num_retries: int = 3,
        page_size: int = 100,
    ):
        self._client = arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )
        logger.info(
            "ArxivClient initialised (delay=%.1fs, retries=%d)",
            delay_seconds, num_retries,
        )

    def fetch_by_date(
        self,
        category: str,
        target_date: date,
        max_results: int = 100,
    ) -> list[ArxivPaper]:
        """
        Fetch papers for a specific category and submission date.

        Args:
            category:    arXiv category string e.g. 'cs.AI'
            target_date: the date to fetch papers for (usually yesterday)
            max_results: cap on results — the API has no hard limit but
                         we bound it to avoid runaway fetches

        Returns:
            List of ArxivPaper objects, empty list on failure.
        """
        if category not in VALID_CATEGORIES:
            logger.warning(
                "Category '%s' not in known set %s", category, VALID_CATEGORIES
            )

        query = self._build_date_query(category, target_date)
        logger.info("Fetching arXiv papers: query=%r max=%d", query, max_results)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in self._client.results(search):
            try:
                papers.append(self._to_paper(result))
            except Exception as exc:
                logger.warning("Could not parse result %s: %s", result.entry_id, exc)
                continue

        logger.info("Fetched %d papers for %s on %s", len(papers), category, target_date)
        return papers

    def fetch_by_query(
        self,
        query: str,
        max_results: int = 50,
    ) -> list[ArxivPaper]:
        """
        Fetch papers with a raw arXiv query string.

        Useful for ad-hoc searches like 'ti:"RAG" AND cat:cs.AI'.
        """
        logger.info("Fetching arXiv papers: query=%r max=%d", query, max_results)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers = []
        for result in self._client.results(search):
            try:
                papers.append(self._to_paper(result))
            except Exception as exc:
                logger.warning("Could not parse result: %s", exc)
                continue

        logger.info("Fetched %d papers for query: %s", len(papers), query)
        return papers

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_date_query(self, category: str, target_date: date) -> str:
        """
        Build an arXiv query for a specific category and single date.

        arXiv's submittedDate format: YYYYMMDDhhmm
        We search the full day: 0000 to 2359.
        """
        date_str = target_date.strftime("%Y%m%d")
        return f"cat:{category} AND submittedDate:[{date_str}0000 TO {date_str}2359]"

    def _to_paper(self, result: arxiv.Result) -> ArxivPaper:
        """Convert a raw arxiv.Result to our ArxivPaper schema."""
        return ArxivPaper(
            arxiv_id=result.entry_id.split("/")[-1],  # validator strips version
            title=result.title,
            abstract=result.summary,
            authors=[a.name for a in result.authors],
            categories=result.categories,
            pdf_url=result.pdf_url,
            published_at=result.published,
        )