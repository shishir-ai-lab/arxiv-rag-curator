"""
arXiv RAG Curator — FastAPI entry point.

Updated to register the hybrid search router and run chunk indexer setup
on startup (creates chunks index + RRF pipeline if not already present).
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
from .routers.hybrid_search import router as hybrid_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting arXiv RAG Curator API...")

    # Core infrastructure
    init_connection_pool()
    init_schema()
    init_index()   # BM25 papers index

    # Chunk index + RRF pipeline (for hybrid search)
    try:
        from ..services.opensearch.factory import make_opensearch_client
        from ..services.embeddings.factory import make_embeddings_service
        from ..services.opensearch.chunk_indexer import ChunkIndexer
        from ..core.database import get_db

        indexer = ChunkIndexer(
            os_client      = make_opensearch_client(),
            embeddings_svc = make_embeddings_service(),
            db             = get_db,
        )
        indexer.setup()   # creates chunks index + RRF pipeline if absent
        logger.info("Chunk indexer setup complete")
    except Exception as exc:
        logger.warning("Chunk indexer setup failed (hybrid search may not work): %s", exc)

    if settings.jina_api_key:
        logger.info("Jina API key detected — hybrid search enabled")
    else:
        logger.info("No Jina API key — hybrid search will fall back to BM25")

    logger.info("API ready.")
    yield

    close_connection_pool()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="arXiv RAG Curator API",
    description=(
        "Production RAG system for arXiv papers. "
        "BM25 keyword search, hybrid semantic search, and LLM-powered Q&A."
    ),
    version="0.3.0",
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
app.include_router(search_router)    # /api/v1/search  (BM25 on papers index)
app.include_router(hybrid_router)    # /api/v1/hybrid-search (chunks index)


@app.get("/")
async def root():
    return {
        "service":       "arXiv RAG Curator",
        "version":       "0.3.0",
        "docs":          "/docs",
        "endpoints": {
            "bm25_search":   "/api/v1/search",
            "hybrid_search": "/api/v1/hybrid-search",
            "health":        "/health",
        },
        "hybrid_enabled": bool(settings.jina_api_key),
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

    embedding = {
        "status":   "enabled" if settings.jina_api_key else "disabled",
        "reason":   None if settings.jina_api_key else "JINA_API_KEY not set",
        "fallback": "bm25",
    }

    all_critical = all(
        s["status"] == "healthy"
        for s in [postgres, opensearch_]
    )

    return {
        "status":   "healthy" if all_critical else "degraded",
        "version":  "0.3.0",
        "services": {
            "postgresql":  postgres,
            "opensearch":  opensearch_,
            "ollama":      ollama,
            "embeddings":  embedding,
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