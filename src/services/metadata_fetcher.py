"""
MetadataFetcher: the ingestion pipeline orchestrator.

Responsibilities:
  1. Search arXiv for papers (via ArxivClient)
  2. Download each paper's PDF (via PDFDownloader)
  3. Parse the PDF (via PDFParser)
  4. Save everything to PostgreSQL (via DB connection)
  5. Return a BatchResult so Airflow can report on what happened

Design:
  - Per-paper error isolation: one failure never aborts the batch
  - Failures are logged and counted, not raised
  - Idempotent: re-running with the same query updates rather than duplicates

Usage:
    from arxiv_rag_curator.services.metadata_fetcher import MetadataFetcher
    from arxiv_rag_curator.services.arxiv.factory import make_arxiv_client
    from arxiv_rag_curator.services.pdf_parser.factory import make_pdf_downloader, make_pdf_parser
    from arxiv_rag_curator.core.database import get_db

    fetcher = MetadataFetcher(
        arxiv_client=make_arxiv_client(),
        pdf_downloader=make_pdf_downloader(),
        pdf_parser=make_pdf_parser(),
        db=get_db,
    )
    result = fetcher.fetch_for_date(category="cs.AI", target_date=date.today())
    print(result.summary())
"""

import logging
from contextlib import contextmanager
from datetime import date
from typing import Callable, Optional

from .arxiv.client import ArxivClient
from .pdf_parser.downloader import PDFDownloader
from .pdf_parser.parser import PDFParser
from .schemas import ArxivPaper, BatchResult, IngestionResult, ParsedPaper

logger = logging.getLogger(__name__)


# ── SQL ───────────────────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO papers (
    arxiv_id, title, abstract, authors, categories,
    published_at, pdf_url, full_text, pdf_parsed, parse_error
)
VALUES (
    %(arxiv_id)s, %(title)s, %(abstract)s, %(authors)s, %(categories)s,
    %(published_at)s, %(pdf_url)s, %(full_text)s, %(pdf_parsed)s, %(parse_error)s
)
ON CONFLICT (arxiv_id) DO UPDATE SET
    title        = EXCLUDED.title,
    abstract     = EXCLUDED.abstract,
    authors      = EXCLUDED.authors,
    categories   = EXCLUDED.categories,
    pdf_url      = EXCLUDED.pdf_url,
    -- COALESCE: don't overwrite good existing text with NULL from a failed re-parse
    full_text    = COALESCE(EXCLUDED.full_text, papers.full_text),
    -- OR: once successfully parsed, stays parsed (even if a re-run fails)
    pdf_parsed   = EXCLUDED.pdf_parsed OR papers.pdf_parsed,
    parse_error  = EXCLUDED.parse_error,
    updated_at   = NOW()
RETURNING id, arxiv_id, pdf_parsed;
"""

_FETCH_UNPARSED_SQL = """
SELECT arxiv_id, pdf_url
FROM papers
WHERE pdf_parsed = FALSE
  AND pdf_url IS NOT NULL
ORDER BY created_at DESC
LIMIT %(limit)s;
"""


# ── Orchestrator ──────────────────────────────────────────────────────────────

class MetadataFetcher:
    """
    Orchestrates the full ingestion pipeline.

    Receives dependencies via constructor (dependency injection):
    - Makes the class testable without real network calls
    - Makes implementations swappable (e.g. MockParser in tests)
    """

    def __init__(
        self,
        arxiv_client: ArxivClient,
        pdf_downloader: PDFDownloader,
        pdf_parser: PDFParser,
        db: Callable,  # context manager factory: yields a psycopg2 connection
    ):
        self._arxiv = arxiv_client
        self._downloader = pdf_downloader
        self._parser = pdf_parser
        self._db = db

    def fetch_for_date(
        self,
        category: str,
        target_date: date,
        max_results: int = 100,
    ) -> BatchResult:
        """
        Fetch and store all papers for a given category and date.

        This is the main entry point for the Airflow daily DAG task.
        """
        logger.info(
            "Starting ingestion: category=%s date=%s max=%d",
            category, target_date, max_results,
        )

        papers = self._arxiv.fetch_by_date(
            category=category,
            target_date=target_date,
            max_results=max_results,
        )

        return self._process_batch(papers)

    def fetch_by_query(
        self,
        query: str,
        max_results: int = 50,
    ) -> BatchResult:
        """
        Fetch and store papers matching an arbitrary arXiv query.

        Useful for backfill runs or topic-specific ingestion.
        """
        logger.info("Starting query ingestion: %r max=%d", query, max_results)
        papers = self._arxiv.fetch_by_query(query=query, max_results=max_results)
        return self._process_batch(papers)

    def retry_failed_pdfs(self, limit: int = 20) -> BatchResult:
        """
        Re-attempt PDF parsing for papers where pdf_parsed = FALSE.

        Called by the Airflow DAG's 'retry_failed_pdfs' stage.
        Useful because:
          - Docling occasionally fails on first attempt (transient errors)
          - PDF might have been unavailable at ingestion time
          - We can tune Docling settings and retry
        """
        batch = BatchResult()

        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(_FETCH_UNPARSED_SQL, {"limit": limit})
            rows = cursor.fetchall()

        logger.info("Retrying PDF parse for %d papers", len(rows))

        for row in rows:
            arxiv_id = row["arxiv_id"]
            pdf_url = row["pdf_url"]
            batch.total += 1

            try:
                # Re-download if not cached (cache may have been cleared)
                pdf_path = self._downloader.download(arxiv_id, pdf_url)
                if not pdf_path:
                    batch.failed += 1
                    continue

                parsed = self._parser.parse(pdf_path, arxiv_id)

                with self._db() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "UPDATE papers SET full_text=%(text)s, pdf_parsed=%(ok)s, "
                        "parse_error=%(err)s, updated_at=NOW() WHERE arxiv_id=%(id)s",
                        {
                            "text": parsed.full_text or None,
                            "ok":   parsed.parse_success,
                            "err":  parsed.error_message or None,
                            "id":   arxiv_id,
                        },
                    )

                if parsed.parse_success:
                    batch.saved += 1
                    batch.parsed += 1
                    logger.info("Retry success: %s", arxiv_id)
                else:
                    batch.failed += 1
                    batch.errors.append(f"{arxiv_id}: {parsed.error_message}")

            except Exception as exc:
                batch.failed += 1
                batch.errors.append(f"{arxiv_id}: {exc}")
                logger.error("Retry failed for %s: %s", arxiv_id, exc)

        logger.info("Retry complete: %s", batch.summary())
        return batch

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_batch(self, papers: list[ArxivPaper]) -> BatchResult:
        """Process a list of papers through the full pipeline."""
        batch = BatchResult(total=len(papers))

        for paper in papers:
            result = self._process_one(paper)
            if result.success:
                batch.saved += 1
                if result.pdf_parsed:
                    batch.parsed += 1
            else:
                batch.failed += 1
                batch.errors.append(f"{paper.arxiv_id}: {result.error}")

        logger.info("Batch done: %s", batch.summary())
        return batch

    def _process_one(self, paper: ArxivPaper) -> IngestionResult:
        """
        Full pipeline for a single paper.

        Never raises — returns IngestionResult with success=False on any error.
        This is the key isolation guarantee: one bad paper can't kill the batch.
        """
        try:
            # Step 1: download PDF
            pdf_path = self._downloader.download(paper.arxiv_id, paper.pdf_url)

            # Step 2: parse (or record that we couldn't)
            if pdf_path:
                parsed = self._parser.parse(pdf_path, paper.arxiv_id)
            else:
                parsed = ParsedPaper(
                    arxiv_id=paper.arxiv_id,
                    parse_success=False,
                    error_message="PDF skipped (too large or download failed)",
                )

            # Step 3: save to DB
            self._save(paper, parsed)

            return IngestionResult(
                arxiv_id=paper.arxiv_id,
                success=True,
                pdf_parsed=parsed.parse_success,
            )

        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", paper.arxiv_id, exc, exc_info=True)
            return IngestionResult(
                arxiv_id=paper.arxiv_id,
                success=False,
                pdf_parsed=False,
                error=str(exc),
            )

    def _save(self, paper: ArxivPaper, parsed: ParsedPaper) -> None:
        """Upsert paper metadata + parsed content in one transaction."""
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(_UPSERT_SQL, {
                "arxiv_id":    paper.arxiv_id,
                "title":       paper.title,
                "abstract":    paper.abstract,
                "authors":     paper.authors,
                "categories":  paper.categories,
                "published_at": paper.published_at,
                "pdf_url":     paper.pdf_url,
                "full_text":   parsed.full_text if parsed.parse_success else None,
                "pdf_parsed":  parsed.parse_success,
                "parse_error": parsed.error_message if not parsed.parse_success else None,
            })