"""
Pydantic schemas for data flowing through the ingestion pipeline.

Schemas serve as the contract between services:
  - ArxivPaper: what we get from the arXiv API
  - ParsedPaper: what comes out of the PDF parser
  - IngestionResult: per-paper outcome for the orchestrator
  - BatchResult: summary of a full ingestion run

Using Pydantic here (not plain dataclasses) gives us:
  - Automatic type coercion (e.g. string → datetime)
  - Validation at construction time, not at use time
  - .model_dump() for easy DB serialization
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ArxivPaper(BaseModel):
    """A paper as returned by the arXiv API, normalized to our schema."""

    arxiv_id: str = Field(description="e.g. '2301.00001' — no version suffix")
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    pdf_url: str
    published_at: datetime

    @field_validator("arxiv_id")
    @classmethod
    def strip_version(cls, v: str) -> str:
        """Ensure we never store a version-suffixed ID like '2301.00001v2'."""
        # Handle both raw IDs and full URLs
        raw = v.split("/")[-1]   # handles 'http://arxiv.org/abs/2301.00001v2'
        return raw.split("v")[0]  # drops 'v2'

    @field_validator("title", "abstract")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class ParsedPaper(BaseModel):
    """Structured output from a PDF parser."""

    arxiv_id: str
    full_text: str = ""
    sections: dict[str, str] = Field(default_factory=dict)
    parse_success: bool
    error_message: str = ""

    @property
    def char_count(self) -> int:
        return len(self.full_text)

    @property
    def section_count(self) -> int:
        return len(self.sections)


class IngestionResult(BaseModel):
    """Outcome of processing a single paper through the full pipeline."""

    arxiv_id: str
    success: bool
    pdf_parsed: bool = False
    error: str = ""


class BatchResult(BaseModel):
    """Aggregated result of a full ingestion batch run."""

    total: int = 0
    saved: int = 0
    parsed: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.saved / self.total * 100

    @property
    def parse_rate(self) -> float:
        if self.saved == 0:
            return 0.0
        return self.parsed / self.saved * 100

    def summary(self) -> str:
        return (
            f"Batch: {self.total} total | {self.saved} saved ({self.success_rate:.1f}%) | "
            f"{self.parsed} parsed ({self.parse_rate:.1f}%) | {self.failed} failed"
        )