"""
PostgreSQL database connection and schema management.

This version extends the base schema with columns needed for PDF ingestion:
  - pdf_url:    where to download the paper's PDF
  - full_text:  Docling-parsed content (Markdown)
  - pdf_parsed: boolean flag — False means we need to (re)parse
  - parse_error: error message if parsing failed

All migrations use IF NOT EXISTS / IF EXISTS — safe to re-run on restart.
"""

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from .config import settings

logger = logging.getLogger(__name__)

_pool: ThreadedConnectionPool | None = None


def init_connection_pool() -> None:
    global _pool
    _pool = ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        cursor_factory=RealDictCursor,
    )
    logger.info("PostgreSQL connection pool initialised")


def close_connection_pool() -> None:
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


@contextmanager
def get_db() -> Generator:
    """
    Yield a database connection from the pool.

    Commits on success, rolls back on error, always returns to pool.
    """
    if _pool is None:
        raise RuntimeError("Connection pool not initialised")
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_PAPERS_TABLE = """
CREATE TABLE IF NOT EXISTS papers (
    id           SERIAL PRIMARY KEY,
    arxiv_id     VARCHAR(50) UNIQUE NOT NULL,
    title        TEXT NOT NULL,
    abstract     TEXT,
    authors      TEXT[],
    categories   TEXT[],
    published_at TIMESTAMP WITH TIME ZONE,

    -- PDF ingestion columns (added in arxiv-ingestion phase)
    pdf_url      TEXT,
    full_text    TEXT,
    pdf_parsed   BOOLEAN DEFAULT FALSE,
    parse_error  TEXT,

    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

_CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id
    ON papers (arxiv_id);

CREATE INDEX IF NOT EXISTS idx_papers_categories
    ON papers USING GIN (categories);

CREATE INDEX IF NOT EXISTS idx_papers_published_at
    ON papers (published_at DESC);

-- Partial index: efficiently find papers still needing PDF parsing
-- Only indexes rows where pdf_parsed=FALSE, so it stays small
CREATE INDEX IF NOT EXISTS idx_papers_needs_parsing
    ON papers (created_at DESC)
    WHERE pdf_parsed = FALSE;
"""

_CREATE_TRIGGER = """
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_papers_updated_at ON papers;
CREATE TRIGGER update_papers_updated_at
    BEFORE UPDATE ON papers
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
"""


def init_schema() -> None:
    """
    Create or migrate the database schema.

    Safe to call on every startup — all DDL uses IF NOT EXISTS.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(_CREATE_PAPERS_TABLE)
        cursor.execute(_CREATE_INDEXES)
        cursor.execute(_CREATE_TRIGGER)
        logger.info("Database schema initialised")


def check_health() -> dict:
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 AS ok")
            cursor.fetchone()
        return {"status": "healthy"}
    except Exception as exc:
        logger.error("PostgreSQL health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}