"""
arXiv RAG Curator — FastAPI application entry point.

Updated to register the search router added in this phase.
All other lifespan behaviour (DB pool, OpenSearch index init) is unchanged.
"""

import logging
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.database import check_health as db_health
from ..core.database import close_connection_pool, init_connection_pool, init_schema
from ..core.search import check_health as os_health
from ..core.search import init_index
from .routers.search import router as search_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting arXiv RAG Curator API...")

    init_connection_pool()
    init_schema()
    init_index()

    logger.info("All services initialised. API ready.")
    yield

    close_connection_pool()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="arXiv RAG Curator API",
    description=(
        "Production RAG system for arXiv papers. "
        "BM25 keyword search, semantic search, and LLM-powered Q&A."
    ),
    version="0.2.0",
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

# ── Routers ───────────────────────────────────────────────────────────────────

app.include_router(search_router)

# ── Root routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "arXiv RAG Curator",
        "version": "0.2.0",
        "docs":    "/docs",
        "health":  "/health",
        "search":  "/api/v1/search",
    }


@app.get("/health")
async def health_check():
    postgres    = db_health()
    opensearch_ = os_health()

    try:
        r = requests.get(f"{settings.ollama_url}/api/tags", timeout=3)
        ollama = {"status": "healthy"} if r.status_code == 200 else {"status": "unhealthy"}
    except Exception as exc:
        ollama = {"status": "unhealthy", "error": str(exc)}

    all_healthy = all(
        s["status"] == "healthy"
        for s in [postgres, opensearch_, ollama]
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": {
            "postgresql": postgres,
            "opensearch": opensearch_,
            "ollama":     ollama,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "arxiv_rag_curator.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )