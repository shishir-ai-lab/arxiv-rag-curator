"""
ChunkIndexer: the pipeline that turns papers into searchable chunks.

Pipeline per paper:
  1. Read paper content from PostgreSQL
  2. Chunk with TextChunker (section-aware)
  3. Embed chunks with JinaEmbeddingService (batched)
  4. Bulk-index chunks into the `chunks` OpenSearch index

Also responsible for:
  - Creating the chunks index if it doesn't exist
  - Creating the RRF normalization search pipeline (one-time setup)
  - Deleting old chunks for a paper before re-indexing (clean update)

Usage:
    from arxiv_rag_curator.services.opensearch.chunk_indexer import ChunkIndexer
    from arxiv_rag_curator.services.opensearch.factory import make_opensearch_client
    from arxiv_rag_curator.services.embeddings.factory import make_embeddings_service
    from arxiv_rag_curator.core.database import get_db

    indexer = ChunkIndexer(
        os_client=make_opensearch_client(),
        embeddings_svc=make_embeddings_service(),
        db=get_db,
    )
    result = indexer.index_paper(arxiv_id="2301.00001")
"""

import logging
from typing import Callable

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from ..embeddings.jina import JinaEmbeddingService
from ..indexing.text_chunker import TextChunker
from ..indexing.schemas import ChunkIndexResult
from .chunks_index_config import CHUNKS_INDEX_CONFIG, CHUNKS_INDEX_NAME, RRF_PIPELINE_ID

logger = logging.getLogger(__name__)

_FETCH_PAPER_SQL = """
SELECT
    arxiv_id, title, abstract, authors, categories,
    published_at, full_text, pdf_parsed,
    -- sections stored as JSON string in the DB (set by Docling parser)
    sections
FROM papers
WHERE arxiv_id = %s;
"""

_FETCH_ALL_SQL = """
SELECT
    arxiv_id, title, abstract, authors, categories,
    published_at, full_text, pdf_parsed, sections
FROM papers
WHERE pdf_parsed = TRUE
  AND full_text IS NOT NULL
ORDER BY created_at DESC
LIMIT %(limit)s OFFSET %(offset)s;
"""

# RRF normalization pipeline — required for OpenSearch native hybrid query
_RRF_PIPELINE = {
    "description": "RRF normalization for hybrid BM25 + kNN search",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {"technique": "rrf"},
                "combination": {
                    "technique":  "arithmetic_mean",
                    "parameters": {"weights": [0.5, 0.5]},
                },
            }
        }
    ],
}


class ChunkIndexer:
    """
    End-to-end pipeline: paper → chunks → embeddings → OpenSearch.

    Requires:
    - os_client:       OpenSearch client (from opensearch-py)
    - embeddings_svc:  JinaEmbeddingService (may be unavailable)
    - db:              context-manager factory yielding a psycopg2 connection
    """

    def __init__(
        self,
        os_client:      OpenSearch,
        embeddings_svc: JinaEmbeddingService,
        db:             Callable,
    ):
        self._client  = os_client
        self._embed   = embeddings_svc
        self._db      = db
        self._chunker = TextChunker()

    # ── Public API ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """
        One-time setup: create the chunks index and RRF pipeline.
        Safe to call on every startup — no-ops if already present.
        """
        self._ensure_index()
        self._ensure_rrf_pipeline()

    def index_paper(self, arxiv_id: str) -> ChunkIndexResult:
        """
        Chunk and index a single paper by its arXiv ID.

        Fetches from PostgreSQL, chunks, embeds, and bulk-indexes.
        Deletes existing chunks for this paper first (clean update).
        """
        paper = self._fetch_paper(arxiv_id)
        if not paper:
            logger.warning("Paper not found in DB: %s", arxiv_id)
            return ChunkIndexResult(arxiv_id=arxiv_id, n_chunks=0, indexed=0, errors=1)

        return self._process_paper(paper)

    def index_all_papers(self, batch_size: int = 50) -> dict:
        """
        Index all parsed papers in PostgreSQL into the chunks index.

        Processes in batches to keep memory bounded.
        Returns a summary dict suitable for Airflow XCom.
        """
        total_indexed = 0
        total_errors  = 0
        offset        = 0

        logger.info("Starting full chunk indexing run")

        while True:
            papers = self._fetch_batch(limit=batch_size, offset=offset)
            if not papers:
                break

            for paper in papers:
                result = self._process_paper(paper)
                total_indexed += result.indexed
                total_errors  += result.errors

            offset += len(papers)
            if len(papers) < batch_size:
                break

        summary = {
            "total_papers_processed": offset,
            "total_chunks_indexed":   total_indexed,
            "total_errors":           total_errors,
        }
        logger.info("Chunk indexing complete: %s", summary)
        return summary

    # ── Private pipeline ──────────────────────────────────────────────────────

    def _process_paper(self, paper: dict) -> ChunkIndexResult:
        """Full pipeline for one paper: chunk → embed → index."""
        arxiv_id = paper["arxiv_id"]

        # Step 1: chunk
        import json
        sections_raw = paper.get("sections")
        sections = None
        if sections_raw:
            try:
                sections = json.loads(sections_raw) if isinstance(sections_raw, str) else sections_raw
            except (json.JSONDecodeError, TypeError):
                sections = None

        chunks = self._chunker.chunk_paper(
            arxiv_id  = arxiv_id,
            title     = paper.get("title") or "",
            abstract  = paper.get("abstract") or "",
            full_text = paper.get("full_text") or "",
            sections  = sections,
        )

        if not chunks:
            logger.warning("No chunks produced for %s", arxiv_id)
            return ChunkIndexResult(arxiv_id=arxiv_id, n_chunks=0, indexed=0, errors=0)

        # Step 2: embed (skip if service unavailable)
        embeddings = None
        if self._embed.is_available:
            chunk_texts = [c.chunk_text for c in chunks]
            embeddings  = self._embed.embed_passages(chunk_texts)
            if embeddings is None:
                logger.warning("Embedding failed for %s, indexing without vectors", arxiv_id)

        # Step 3: delete old chunks for this paper (clean update)
        self._delete_paper_chunks(arxiv_id)

        # Step 4: bulk-index
        paper_meta = {
            "title":        paper.get("title", ""),
            "abstract":     paper.get("abstract", ""),
            "authors":      paper.get("authors") or [],
            "categories":   paper.get("categories") or [],
            "published_at": paper["published_at"].isoformat() if paper.get("published_at") else None,
        }

        indexed, errors = self._bulk_index(chunks, embeddings, paper_meta)

        logger.info(
            "Indexed %s: %d chunks, %d indexed, %d errors",
            arxiv_id, len(chunks), indexed, errors,
        )
        return ChunkIndexResult(
            arxiv_id=arxiv_id,
            n_chunks=len(chunks),
            indexed=indexed,
            errors=errors,
        )

    def _bulk_index(self, chunks, embeddings, paper_meta) -> tuple[int, int]:
        """Build and execute a bulk index request."""
        actions = []
        for i, chunk in enumerate(chunks):
            doc = {
                "chunk_id":     chunk.chunk_id,
                "arxiv_id":     chunk.arxiv_id,
                "chunk_index":  chunk.chunk_index,
                "section_name": chunk.section_name,
                "word_count":   chunk.word_count,
                "chunk_text":   chunk.chunk_text,
                **paper_meta,
            }
            # Only add embedding if we have one for this chunk
            if embeddings and i < len(embeddings) and embeddings[i]:
                doc["embedding"] = embeddings[i]

            actions.append({
                "_op_type": "index",
                "_index":   CHUNKS_INDEX_NAME,
                "_id":      chunk.chunk_id,
                "_source":  doc,
            })

        if not actions:
            return 0, 0

        success, failed = bulk(self._client, actions, raise_on_error=False, stats_only=False)
        error_count = len(failed) if isinstance(failed, list) else int(failed)
        return success, error_count

    def _delete_paper_chunks(self, arxiv_id: str) -> None:
        """Delete all existing chunks for a paper before re-indexing."""
        try:
            self._client.delete_by_query(
                index=CHUNKS_INDEX_NAME,
                body={"query": {"term": {"arxiv_id": arxiv_id}}},
                params={"refresh": "true"},
            )
        except Exception as exc:
            logger.warning("Could not delete old chunks for %s: %s", arxiv_id, exc)

    # ── DB fetch helpers ──────────────────────────────────────────────────────

    def _fetch_paper(self, arxiv_id: str) -> dict | None:
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(_FETCH_PAPER_SQL, (arxiv_id,))
            row = cursor.fetchone()
        return dict(row) if row else None

    def _fetch_batch(self, limit: int, offset: int) -> list[dict]:
        with self._db() as conn:
            cursor = conn.cursor()
            cursor.execute(_FETCH_ALL_SQL, {"limit": limit, "offset": offset})
            return [dict(row) for row in cursor.fetchall()]

    # ── Index / pipeline lifecycle ────────────────────────────────────────────

    def _ensure_index(self) -> None:
        if not self._client.indices.exists(index=CHUNKS_INDEX_NAME):
            self._client.indices.create(index=CHUNKS_INDEX_NAME, body=CHUNKS_INDEX_CONFIG)
            logger.info("Created chunks index '%s'", CHUNKS_INDEX_NAME)
        else:
            logger.debug("Chunks index '%s' already exists", CHUNKS_INDEX_NAME)

    def _ensure_rrf_pipeline(self) -> None:
        """Create the RRF normalization pipeline if it doesn't exist."""
        try:
            self._client.transport.perform_request(
                method="PUT",
                url=f"/_search/pipeline/{RRF_PIPELINE_ID}",
                body=_RRF_PIPELINE,
            )
            logger.info("RRF pipeline '%s' created/updated", RRF_PIPELINE_ID)
        except Exception as exc:
            logger.warning("Could not create RRF pipeline: %s", exc)
            logger.warning("Hybrid search may fall back to BM25 if pipeline is unavailable")