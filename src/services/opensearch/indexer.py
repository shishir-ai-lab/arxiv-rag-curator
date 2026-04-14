"""
Paper indexer — syncs PostgreSQL → OpenSearch.

Two sync modes:

1. Write-through (in MetadataFetcher): index immediately after saving to DB.
   Fast but can miss papers if OpenSearch was briefly down.

2. Batch sync (this module, called by Airflow DAG): read all unindexed papers
   from PostgreSQL and push them in bulk. Acts as a safety net and catch-up
   mechanism for any gaps.

The indexer reads papers from PostgreSQL and transforms them into
OpenSearch documents. It does NOT decide what to search for — that's
QueryBuilder's job.

Usage:
    from arxiv_rag_curator.services.opensearch.indexer import PaperIndexer
    from arxiv_rag_curator.services.opensearch.factory import make_search_service
    from arxiv_rag_curator.core.database import get_db

    indexer = PaperIndexer(search_service=make_search_service(), db=get_db)
    result = indexer.sync_all(batch_size=200)
    print(result)
"""

import logging
from contextlib import AbstractContextManager
from typing import Callable, Optional

from .service import IndexResult, SearchService

logger = logging.getLogger(__name__)

# How many papers to load from PostgreSQL per batch
DEFAULT_BATCH_SIZE = 200

# SQL to fetch papers for indexing — only columns needed for search
_FETCH_SQL = """
SELECT
    arxiv_id,
    title,
    abstract,
    authors,
    categories,
    published_at,
    pdf_parsed,
    full_text
FROM papers
ORDER BY created_at DESC
LIMIT %(limit)s OFFSET %(offset)s;
"""

_COUNT_SQL = "SELECT COUNT(*) AS total FROM papers;"


class PaperIndexer:
    """
    Syncs papers from PostgreSQL to OpenSearch.

    Processes in batches to avoid loading the entire table into memory.
    Each batch is a separate bulk index call.

    All indexing is idempotent — re-running produces the same result.
    arxiv_id is used as the OpenSearch document _id so re-indexing
    overwrites rather than duplicates.
    """

    def __init__(
        self,
        search_service: SearchService,
        db: Callable,   # context manager factory: yields a psycopg2 connection
    ):
        self._svc = search_service
        self._db  = db

    def sync_all(self, batch_size: int = DEFAULT_BATCH_SIZE) -> dict:
        """
        Sync all papers from PostgreSQL to OpenSearch.

        Processes in batches of batch_size to keep memory usage bounded.
        Returns a summary dict suitable for Airflow XCom.
        """
        total_in_db  = self._count_papers()
        total_indexed = 0
        total_errors  = 0
        offset        = 0

        logger.info("Starting full sync: %d papers in PostgreSQL", total_in_db)

        while True:
            batch = self._fetch_batch(limit=batch_size, offset=offset)
            if not batch:
                break  # no more papers

            docs   = [self._to_doc(row) for row in batch]
            result = self._svc.bulk_index(docs)

            total_indexed += result.indexed
            total_errors  += result.errors
            offset        += len(batch)

            logger.info(
                "Batch %d–%d: indexed=%d errors=%d",
                offset - len(batch), offset, result.indexed, result.errors,
            )

            if len(batch) < batch_size:
                break  # last page

        summary = {
            "total_in_db":    total_in_db,
            "total_indexed":  total_indexed,
            "total_errors":   total_errors,
            "success_rate":   f"{total_indexed / max(total_in_db, 1) * 100:.1f}%",
        }
        logger.info("Sync complete: %s", summary)
        return summary

    def index_one(self, arxiv_id: str) -> bool:
        """
        Index a single paper by arxiv_id.
        Called by MetadataFetcher immediately after a paper is saved to DB.
        """
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT arxiv_id, title, abstract, authors, categories, "
                "published_at, pdf_parsed, full_text FROM papers WHERE arxiv_id = %s",
                (arxiv_id,),
            )
            row = cursor.fetchone()

        if not row:
            logger.warning("Paper not found in DB: %s", arxiv_id)
            return False

        doc = self._to_doc(dict(row))
        return self._svc.index_paper(doc)

    # ── Private ───────────────────────────────────────────────────────────────

    def _fetch_batch(self, limit: int, offset: int) -> list[dict]:
        """Fetch a page of papers from PostgreSQL."""
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(_FETCH_SQL, {"limit": limit, "offset": offset})
            return [dict(row) for row in cursor.fetchall()]

    def _count_papers(self) -> int:
        """Return total paper count from PostgreSQL."""
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(_COUNT_SQL)
            return cursor.fetchone()["total"]

    def _to_doc(self, row: dict) -> dict:
        """
        Transform a PostgreSQL row into an OpenSearch document.

        Decisions:
        - published_at: convert datetime → ISO string (OpenSearch date format)
        - full_text: only include when pdf_parsed is True; avoid indexing None
        - authors/categories: ensure list type (not None)
        - Exclude: id, created_at, updated_at, parse_error (not for search)
        """
        doc: dict = {
            "arxiv_id":    row["arxiv_id"],
            "title":       (row.get("title") or "").strip(),
            "abstract":    (row.get("abstract") or "").strip(),
            "authors":     row.get("authors") or [],
            "categories":  row.get("categories") or [],
            "pdf_parsed":  bool(row.get("pdf_parsed")),
            "published_at": (
                row["published_at"].isoformat()
                if row.get("published_at") else None
            ),
        }

        # Only include full_text when it's actually parsed content.
        # Indexing an empty or None full_text wastes space and can hurt scoring.
        if row.get("pdf_parsed") and row.get("full_text"):
            doc["full_text"] = row["full_text"]

        return doc