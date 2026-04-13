"""
PDF downloader with local disk cache.

Design principles:
  1. Cache-first: never re-download the same PDF
  2. Stream to disk: constant memory regardless of PDF size
  3. Size guard: skip unusually large PDFs (corrupted or book-length)
  4. Raise on network errors: let the caller (MetadataFetcher) handle retries

Cache structure:
  {PDF_CACHE_DIR}/{arxiv_id}.pdf

The cache persists across pipeline runs, so crashed runs restart fast.
You can clear it manually:  rm -rf /tmp/arxiv_pdf_cache/

Usage:
    from arxiv_rag_curator.services.pdf_parser.downloader import PDFDownloader
    downloader = PDFDownloader()
    path = downloader.download("2301.00001", "https://arxiv.org/pdf/2301.00001")
"""

import logging
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("/tmp/arxiv_pdf_cache")
MAX_PDF_SIZE_MB = 50
DOWNLOAD_TIMEOUT_SECONDS = 60
CHUNK_SIZE_BYTES = 8192


class PDFDownloader:
    """
    Downloads arXiv PDFs to a local cache directory.

    Thread-safe: each download writes to a unique path (by arxiv_id),
    so concurrent downloads for different papers are fine.
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        max_size_mb: float = MAX_PDF_SIZE_MB,
        timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    ):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._max_size_bytes = int(max_size_mb * 1024 * 1024)
        self._timeout = timeout
        logger.info("PDFDownloader cache: %s", self._cache_dir)

    def download(self, arxiv_id: str, pdf_url: str) -> Optional[Path]:
        """
        Download a PDF to the local cache.

        Returns:
            Path to the cached PDF file, or None if the file was skipped
            (e.g. too large). Raises httpx exceptions on network failure —
            callers should handle retries.

        The caller is responsible for deciding whether None is acceptable
        (e.g. skip this paper) or should trigger a retry.
        """
        cached_path = self._cache_path(arxiv_id)

        # Cache hit — return immediately, zero network calls
        if cached_path.exists() and cached_path.stat().st_size > 0:
            logger.debug("Cache hit: %s (%s)", arxiv_id, self._human_size(cached_path))
            return cached_path

        logger.info("Downloading PDF: %s → %s", arxiv_id, pdf_url)

        with httpx.Client(
            timeout=self._timeout,
            follow_redirects=True,
            headers={"User-Agent": "arxiv-rag-curator/0.1 (academic research tool)"},
        ) as client:
            with client.stream("GET", pdf_url) as response:
                response.raise_for_status()

                # Check content-length header if server provides it
                content_length = int(response.headers.get("content-length", 0))
                if content_length > self._max_size_bytes:
                    size_mb = content_length / (1024 * 1024)
                    logger.warning(
                        "PDF too large (%.1f MB > %.0f MB limit), skipping: %s",
                        size_mb, MAX_PDF_SIZE_MB, arxiv_id,
                    )
                    return None

                # Write in chunks: low constant memory regardless of PDF size
                bytes_written = 0
                with cached_path.open("wb") as f:
                    for chunk in response.iter_bytes(chunk_size=CHUNK_SIZE_BYTES):
                        f.write(chunk)
                        bytes_written += len(chunk)
                        # Guard against servers lying about content-length
                        if bytes_written > self._max_size_bytes:
                            cached_path.unlink(missing_ok=True)
                            logger.warning(
                                "PDF exceeded size limit during download, skipped: %s",
                                arxiv_id,
                            )
                            return None

        logger.info(
            "Downloaded %s (%s)",
            arxiv_id,
            self._human_size(cached_path),
        )
        return cached_path

    def is_cached(self, arxiv_id: str) -> bool:
        """Check if a PDF is already in the local cache."""
        path = self._cache_path(arxiv_id)
        return path.exists() and path.stat().st_size > 0

    def cache_path(self, arxiv_id: str) -> Path:
        """Return the expected cache path for a given arxiv_id."""
        return self._cache_path(arxiv_id)

    def clear_cache(self) -> int:
        """Delete all cached PDFs. Returns number of files deleted."""
        deleted = 0
        for pdf in self._cache_dir.glob("*.pdf"):
            pdf.unlink()
            deleted += 1
        logger.info("Cache cleared: %d files deleted", deleted)
        return deleted

    # ── Private ───────────────────────────────────────────────────────────────

    def _cache_path(self, arxiv_id: str) -> Path:
        return self._cache_dir / f"{arxiv_id}.pdf"

    def _human_size(self, path: Path) -> str:
        size = path.stat().st_size
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f}KB"
        else:
            return f"{size/(1024*1024):.1f}MB"