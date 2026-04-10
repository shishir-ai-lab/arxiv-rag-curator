"""
PostgreSQL database connection and schema management.

Responsibilities:
- Connection pooling via psycopg2
- Schema creation (tables + indexes)
- Async-ready session factory (SQLAlchemy for future use)

Design principle: PostgreSQL is the source of truth. OpenSearch is a
search-optimised projection. Always write to Postgres first.
"""

import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from .config import settings

logger = logging.getLogger(__name__)

# ── Connection pool ──────────────────────────────────────────────────────────
# A pool reuses connections instead of opening a new socket per query.
# minconn=1: always keep 1 connection alive
# maxconn=10: never open more than 10 simultaneous connections
_pool: ThreadedConnectionPool | None = None


def init_connection_pool() -> None:
    """Initialise the PostgreSQL connection pool. Called on app startup."""
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
    """Close all connections in the pool. Called on app shutdown."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


@contextmanager
def get_db() -> Generator:
    """
    Context manager that yields a database connection from the pool.

    Usage:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")

    Automatically returns the connection to the pool when done.
    Rolls back on error; commits on success.
    """
    if _pool is None:
        raise RuntimeError("Connection pool not initialised. Call init_connection_pool() first.")

    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


# ── Schema ───────────────────────────────────────────────────────────────────

_CREATE_PAPERS_TABLE = """
CREATE TABLE IF NOT EXISTS papers (
    id           SERIAL PRIMARY KEY,
    arxiv_id     VARCHAR(50) UNIQUE NOT NULL,
    title        TEXT NOT NULL,
    abstract     TEXT,
    authors      TEXT[],          -- PostgreSQL native array; GIN-indexed
    categories   TEXT[],          -- e.g. ['cs.LG', 'cs.CL']
    published_at TIMESTAMP WITH TIME ZONE,
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
"""

_CREATE_INDEXES = """
-- Fast lookup by arXiv ID (used in upsert logic)
CREATE INDEX IF NOT EXISTS idx_papers_arxiv_id
    ON papers (arxiv_id);

-- GIN index for array containment queries:
--   WHERE 'cs.AI' = ANY(categories)
CREATE INDEX IF NOT EXISTS idx_papers_categories
    ON papers USING GIN (categories);

-- Range queries on published date
CREATE INDEX IF NOT EXISTS idx_papers_published_at
    ON papers (published_at DESC);
"""

_CREATE_UPDATE_TRIGGER = """
-- Auto-update updated_at on any row change
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
    Create tables, indexes, and triggers if they don't exist.

    Safe to call on every startup — all statements use IF NOT EXISTS.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(_CREATE_PAPERS_TABLE)
        cursor.execute(_CREATE_INDEXES)
        cursor.execute(_CREATE_UPDATE_TRIGGER)
        logger.info("Database schema initialised")


# ── Health check ─────────────────────────────────────────────────────────────

def check_health() -> dict:
    """Return health status of the PostgreSQL connection."""
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 AS ok")
            cursor.fetchone()
        return {"status": "healthy"}
    except Exception as exc:
        logger.error("PostgreSQL health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}