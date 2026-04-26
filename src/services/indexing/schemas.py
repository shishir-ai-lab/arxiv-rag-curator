"""
Schemas for the chunking and indexing pipeline.

TextChunk is the central data structure: it flows from
TextChunker → EmbeddingsService → ChunkIndexer, picking up
an embedding vector along the way.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextChunk:
    """
    A single chunk of a paper, ready for embedding and indexing.

    chunk_text is what gets embedded — it includes the context header
    (title + abstract snippet) so the vector encodes paper-level context,
    not just the isolated section content.
    """
    chunk_id:     str         # UUID — used as OpenSearch _id
    arxiv_id:     str         # back-reference to the source paper
    chunk_text:   str         # full text including context header
    section_name: str         # which section this chunk came from
    chunk_index:  int         # 0-based position within the paper
    word_count:   int


@dataclass
class EmbeddedChunk:
    """A TextChunk paired with its embedding vector. Ready to index."""
    chunk:     TextChunk
    embedding: list[float]    # 1024-dimensional vector from Jina v3


@dataclass
class ChunkIndexResult:
    """Result of indexing one paper's worth of chunks."""
    arxiv_id:  str
    n_chunks:  int
    indexed:   int
    errors:    int

    @property
    def success(self) -> bool:
        return self.errors == 0