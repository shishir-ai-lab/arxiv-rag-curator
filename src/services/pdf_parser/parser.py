"""
PDF parser: abstract interface + Docling implementation.

Why an abstract interface?
  - Tests use MockParser (instant, no real PDF needed)
  - Swap Docling → PyMuPDF without touching MetadataFetcher
  - Add parser-specific configs without leaking into callers

Why Docling over PyPDF2/pdfplumber?
  - Handles multi-column academic layouts correctly
  - Preserves section structure as Markdown headings
  - Consistent output: same PDF → same result (no randomness)
  - Tables extracted as structured data, not word soup
  - Future: OCR and VLM support for figures/charts

Usage:
    from arxiv_rag_curator.services.pdf_parser.parser import DoclingParser, MockParser
    parser = DoclingParser()
    result = parser.parse(Path("paper.pdf"), "2301.00001")
    if result.parse_success:
        print(result.full_text[:1000])
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .schemas import ParsedPaper

logger = logging.getLogger(__name__)


# ── Abstract interface ────────────────────────────────────────────────────────

class PDFParser(ABC):
    """
    Abstract base for all PDF parsers.

    Contract:
      - parse() NEVER raises. It always returns a ParsedPaper.
      - Callers check parse_success, not try/except.
      - This enables per-paper failure isolation in the orchestrator.
    """

    @abstractmethod
    def parse(self, pdf_path: Path, arxiv_id: str) -> ParsedPaper:
        """
        Parse a PDF into structured content.

        Args:
            pdf_path: local path to the PDF file
            arxiv_id: used for logging and as the identifier in the result

        Returns:
            ParsedPaper — always. Check parse_success for outcome.
        """
        ...


# ── Mock implementation (for tests) ──────────────────────────────────────────

class MockParser(PDFParser):
    """
    Fast in-memory parser for testing.

    Returns synthetic content immediately — no real PDF needed.
    Use this in unit tests and notebook exploration.
    """

    def __init__(self, should_fail: bool = False):
        """
        Args:
            should_fail: set True to simulate a parse failure in tests
        """
        self._should_fail = should_fail

    def parse(self, pdf_path: Path, arxiv_id: str) -> ParsedPaper:
        if self._should_fail:
            return ParsedPaper(
                arxiv_id=arxiv_id,
                parse_success=False,
                error_message="MockParser configured to fail",
            )
        return ParsedPaper(
            arxiv_id=arxiv_id,
            full_text=f"[Mock full text for {arxiv_id}] Introduction... Methods... Results...",
            sections={
                "Introduction": f"Introduction content for {arxiv_id}",
                "Methods": "Methodology section",
                "Results": "Results and evaluation",
                "Conclusion": "Conclusion",
            },
            parse_success=True,
        )


# ── Docling implementation (for production) ───────────────────────────────────

class DoclingParser(PDFParser):
    """
    Production PDF parser backed by Docling.

    Docling is a scientific document understanding library developed by IBM.
    It understands academic paper structure: multi-column layouts, mathematical
    notation, section headers, tables, references.

    First instantiation loads ML models (~10s). Subsequent parses reuse them.
    Each paper takes 20-60s depending on length and CPU.

    do_ocr=False:          faster; OCR only needed for scanned/image PDFs
    do_table_structure=True: extract table contents as structured markdown
    """

    def __init__(self, do_ocr: bool = False, do_table_structure: bool = True):
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )
        # Converter is expensive to initialise; keep it as an instance attribute
        # so it's created once per DoclingParser instance (not per parse call)
        self._converter = DocumentConverter()
        logger.info(
            "DoclingParser initialised (ocr=%s, table_structure=%s)",
            do_ocr, do_table_structure,
        )

    def parse(self, pdf_path: Path, arxiv_id: str) -> ParsedPaper:
        """
        Parse a PDF file into structured Markdown content.

        Returns ParsedPaper with parse_success=False on any error —
        never raises, so callers can process failures gracefully.
        """
        try:
            logger.info("Parsing %s with Docling...", arxiv_id)
            start = time.perf_counter()

            result = self._converter.convert(str(pdf_path))
            doc = result.document

            # Docling exports to Markdown, preserving:
            # - # Heading structure → section boundaries
            # - | Table | cells | → readable tables
            # - Paragraph breaks → paragraph separation
            full_text = doc.export_to_markdown()

            if not full_text.strip():
                logger.warning("Docling returned empty text for %s", arxiv_id)
                return ParsedPaper(
                    arxiv_id=arxiv_id,
                    parse_success=False,
                    error_message="Empty content after parsing",
                )

            sections = self._split_into_sections(full_text)
            elapsed = time.perf_counter() - start

            logger.info(
                "Parsed %s in %.1fs: %d chars, %d sections",
                arxiv_id, elapsed, len(full_text), len(sections),
            )

            return ParsedPaper(
                arxiv_id=arxiv_id,
                full_text=full_text,
                sections=sections,
                parse_success=True,
            )

        except Exception as exc:
            logger.error("Docling failed for %s: %s", arxiv_id, exc, exc_info=True)
            return ParsedPaper(
                arxiv_id=arxiv_id,
                parse_success=False,
                error_message=str(exc),
            )

    def _split_into_sections(self, markdown_text: str) -> dict[str, str]:
        """
        Split Markdown text into {heading: content} pairs.

        Handles both '# Heading' and '## Sub-heading' styles.
        Strips reference sections (usually just citation lists, low value for RAG).
        """
        sections: dict[str, str] = {}
        current_heading = "Preamble"   # content before first heading
        current_lines: list[str] = []

        for line in markdown_text.splitlines():
            if line.startswith("#"):
                # Save previous section (if it has content)
                content = "\n".join(current_lines).strip()
                if content:
                    sections[current_heading] = content
                # Start new section
                current_heading = line.lstrip("# ").strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Save the final section
        content = "\n".join(current_lines).strip()
        if content:
            sections[current_heading] = content

        # Drop reference/bibliography sections — they're citation lists,
        # not useful prose for RAG retrieval
        skip_headings = {"references", "bibliography", "acknowledgments", "acknowledgements"}
        sections = {
            k: v for k, v in sections.items()
            if k.lower() not in skip_headings
        }

        return sections