"""
arXiv RAG Curator — FastAPI application entry point.

Lifespan pattern:
  - Everything in the `startup` block runs before the first request
  - Everything in the `shutdown` block runs after the last request
  - This is the modern replacement for @app.on_event("startup")
"""

import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.database import check_health as db_health
from ..core.database import close_connection_pool, init_connection_pool, init_schema
from ..core.search import check_health as search_health
from ..core.search import get_opensearch_client, init_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown lifecycle.

    Startup order matters: DB must exist before schema creation,
    OpenSearch must be reachable before index creation.
    """
    logger.info("Starting arXiv RAG Curator API...")

    # 1. PostgreSQL
    logger.info("Connecting to PostgreSQL...")
    init_connection_pool()
    init_schema()

    # 2. OpenSearch
    logger.info("Connecting to OpenSearch...")
    init_index()

    logger.info("All services initialised. API ready.")
    yield  # <── app is running here

    # Shutdown
    logger.info("Shutting down...")
    close_connection_pool()
    logger.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="arXiv RAG Curator API",
    description=(
        "Production RAG system for arXiv papers. "
        "Combines BM25 keyword search and semantic vector search to answer "
        "questions about academic papers."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "arXiv RAG Curator",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    Comprehensive health check across all downstream services.

    Returns HTTP 200 if all services are healthy.
    Returns HTTP 503 if any service is unhealthy.
    (HTTP status code enforcement added in a later commit.)
    """
    postgres = db_health()
    opensearch = search_health()

    # Ollama check — lightweight HTTP ping
    try:
        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        ollama = {"status": "healthy"} if r.status_code == 200 else {"status": "unhealthy"}
    except Exception as exc:
        ollama = {"status": "unhealthy", "error": str(exc)}

    all_healthy = all(
        s["status"] == "healthy"
        for s in [postgres, opensearch, ollama]
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": {
            "postgresql": postgres,
            "opensearch": opensearch,
            "ollama": ollama,
        },
    }


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "arxiv_rag_curator.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )