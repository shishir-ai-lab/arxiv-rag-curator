"""
Section-aware text chunker for academic papers.

The core insight: academic papers have semantic structure (sections).
Chunking along section boundaries preserves coherent ideas in each chunk.
Fixed-size token chunking ignores structure and fragments ideas mid-sentence.

Algorithm (4 cases):
  1. Small section   (< min_chunk_size words): merge with the next section
  2. Perfect section (min..chunk_size words):  use as a single chunk
  3. Large section   (> chunk_size words):     split with sliding window + overlap
  4. No sections available:                   paragraph-based greedy grouping

Context header:
  Every chunk is prefixed with:
    Title: {paper title}
    Abstract: {first 80 words of abstract}

    Section: {section_name}

  This means the embedding model always sees paper-level context, not just
  an isolated fragment. It's the single biggest quality improvement over
  naive fixed-size chunking.

Usage:
    chunker = TextChunker()
    chunks = chunker.chunk_paper(
        arxiv_id="2301.00001",
        title="Attention Is All You Need",
        abstract="We propose...",
        full_text="...",
        sections={"Introduction": "...", "Methods": "..."},
    )
"""

import re
import uuid
import logging
from typing import Optional

from .schemas import TextChunk

logger = logging.getLogger(__name__)

# Tuneable defaults — match what the blog post uses
DEFAULT_CHUNK_SIZE    = 600   # target words per chunk
DEFAULT_OVERLAP_SIZE  = 100   # overlap words between sliding-window sub-chunks
DEFAULT_MIN_CHUNK_SIZE = 100  # sections smaller than this get merged with next

# How many abstract words to include in every context header
ABSTRACT_HEADER_WORDS = 80

# Section names to always skip — they're citation lists, not useful prose
SKIP_SECTIONS = frozenset({
    "references", "bibliography",
    "acknowledgments", "acknowledgements",
    "appendix",
})


class TextChunker:
    """
    Section-aware chunker for academic papers.

    Thread-safe: all state is in local variables during chunk_paper().
    One instance can be shared across threads.
    """

    def __init__(
        self,
        chunk_size:     int = DEFAULT_CHUNK_SIZE,
        overlap_size:   int = DEFAULT_OVERLAP_SIZE,
        min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    ):
        self.chunk_size     = chunk_size
        self.overlap_size   = overlap_size
        self.min_chunk_size = min_chunk_size

    def chunk_paper(
        self,
        arxiv_id: str,
        title:    str,
        abstract: str,
        full_text: str,
        sections: Optional[dict[str, str]] = None,
    ) -> list[TextChunk]:
        """
        Chunk a paper into semantically coherent pieces.

        Args:
            arxiv_id:  the paper's arXiv ID
            title:     paper title
            abstract:  paper abstract
            full_text: complete parsed text (used if sections is empty)
            sections:  {section_name: section_content} from Docling

        Returns:
            List of TextChunk objects, ordered by chunk_index.
        """
        # Build the context header — prepended to every chunk
        abstract_words   = abstract.split()[:ABSTRACT_HEADER_WORDS]
        abstract_snippet = " ".join(abstract_words)
        header = f"Title: {title}\nAbstract: {abstract_snippet}\n\n"

        # Filter out low-value sections
        usable_sections = {
            k: v for k, v in (sections or {}).items()
            if k.lower() not in SKIP_SECTIONS and v.strip()
        } if sections else {}

        if len(usable_sections) >= 2:
            chunks = self._chunk_by_sections(arxiv_id, header, usable_sections)
        elif full_text.strip():
            chunks = self._chunk_by_paragraphs(arxiv_id, header, full_text)
        else:
            # Edge case: no text at all (parse failed or paper is empty)
            logger.warning("No usable content for chunking: %s", arxiv_id)
            chunks = []

        logger.info(
            "Chunked %s into %d chunks (strategy: %s)",
            arxiv_id, len(chunks),
            "sections" if len(usable_sections) >= 2 else "paragraphs",
        )
        return chunks

    # ── Private: section-aware path ──────────────────────────────────────────

    def _chunk_by_sections(
        self,
        arxiv_id: str,
        header:   str,
        sections: dict[str, str],
    ) -> list[TextChunk]:
        """Apply the 4-case algorithm to structured sections."""
        chunks: list[TextChunk] = []
        chunk_index = 0
        sections_list = list(sections.items())
        i = 0

        while i < len(sections_list):
            name, content = sections_list[i]
            words = content.split()
            wc    = len(words)

            # Case 1: Too small — merge with next section and retry
            if wc < self.min_chunk_size and i + 1 < len(sections_list):
                next_name, next_content = sections_list[i + 1]
                merged_name    = f"{name} + {next_name}"
                merged_content = content.rstrip() + "\n\n" + next_content.lstrip()
                sections_list[i + 1] = (merged_name, merged_content)
                logger.debug("Merged small section '%s' into '%s'", name, merged_name)
                i += 1
                continue

            # Case 2: Perfect size — single chunk
            if wc <= self.chunk_size:
                chunk_text = f"{header}Section: {name}\n\n{content}"
                chunks.append(self._make_chunk(arxiv_id, chunk_text, name, chunk_index))
                chunk_index += 1

            # Case 3: Too large — sliding window split
            else:
                sub_texts = self._sliding_window(words)
                total     = len(sub_texts)
                for j, sub_text in enumerate(sub_texts):
                    section_label = f"{name} (part {j+1}/{total})"
                    chunk_text    = f"{header}Section: {section_label}\n\n{sub_text}"
                    chunks.append(self._make_chunk(arxiv_id, chunk_text, section_label, chunk_index))
                    chunk_index += 1

            i += 1

        return chunks

    # ── Private: paragraph fallback ───────────────────────────────────────────

    def _chunk_by_paragraphs(
        self,
        arxiv_id: str,
        header:   str,
        full_text: str,
    ) -> list[TextChunk]:
        """
        Fallback: group paragraphs greedily until chunk_size is reached.

        Splits on double newlines (paragraph boundaries). When a group
        reaches chunk_size, the last overlap_size words seed the next chunk.
        """
        paragraphs = [p.strip() for p in re.split(r"\n\n+", full_text) if p.strip()]

        combined: list[str] = []
        current_words: list[str] = []

        for para in paragraphs:
            para_words = para.split()
            if current_words and len(current_words) + len(para_words) > self.chunk_size:
                combined.append(" ".join(current_words))
                # Seed next chunk with overlap from end of current
                current_words = current_words[-self.overlap_size:]
            current_words.extend(para_words)

        if current_words:
            combined.append(" ".join(current_words))

        chunks: list[TextChunk] = []
        for i, text in enumerate(combined):
            chunk_text = f"{header}Section: Content\n\n{text}"
            chunks.append(self._make_chunk(arxiv_id, chunk_text, "Content", i))

        return chunks

    # ── Private: utilities ────────────────────────────────────────────────────

    def _sliding_window(self, words: list[str]) -> list[str]:
        """Split a word list into overlapping windows of chunk_size words."""
        step   = self.chunk_size - self.overlap_size
        result = []
        start  = 0
        while start < len(words):
            window = words[start : start + self.chunk_size]
            result.append(" ".join(window))
            if start + self.chunk_size >= len(words):
                break
            start += step
        return result

    def _make_chunk(
        self,
        arxiv_id:     str,
        chunk_text:   str,
        section_name: str,
        chunk_index:  int,
    ) -> TextChunk:
        return TextChunk(
            chunk_id     = str(uuid.uuid4()),
            arxiv_id     = arxiv_id,
            chunk_text   = chunk_text,
            section_name = section_name,
            chunk_index  = chunk_index,
            word_count   = len(chunk_text.split()),
        )