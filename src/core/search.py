"""
OpenSearch client and index management.

Responsibilities:
- Client singleton with connection pooling
- Index creation with proper field mappings
- Health check

Design principle: OpenSearch holds a search-optimised *projection* of data
that lives in PostgreSQL. Fields are typed deliberately:
  - `keyword`: exact match, filtering, aggregations (not analyzed)
  - `text`:    full-text search with tokenization (analyzed)
  - `date`:    range queries, sorting by date
"""

import logging

from opensearchpy import OpenSearch, OpenSearchException

from .config import settings

logger = logging.getLogger(__name__)

# ── Index configuration ───────────────────────────────────────────────────────

PAPERS_INDEX = "papers"

# Index settings + field mappings.
# Mappings are set at index creation; changing them on an existing index
# requires a reindex operation — so get them right early.
PAPERS_INDEX_CONFIG = {
    "settings": {
        "number_of_shards": 1,      # single node in dev; scale up in prod
        "number_of_replicas": 0,    # 0 replicas fine for single-node dev
        "analysis": {
            "analyzer": {
                # Custom analyzer for academic text:
                # lowercase → remove stopwords → stem (snowball)
                "paper_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            # keyword: exact match only — used in filters & aggregations
            "arxiv_id":     {"type": "keyword"},
            "authors":      {"type": "keyword"},
            "categories":   {"type": "keyword"},

            # text: tokenized for full-text BM25 search
            "title": {
                "type": "text",
                "analyzer": "paper_analyzer",
                # 'keyword' sub-field lets us sort/aggregate by exact title
                "fields": {"keyword": {"type": "keyword", "ignore_above": 512}},
            },
            "abstract": {
                "type": "text",
                "analyzer": "paper_analyzer",
            },

            # date: ISO-8601 string or epoch millis
            "published_at": {"type": "date"},

            # Later we will add this for semantic (vector) search:
            # "embedding": {
            #     "type": "knn_vector",
            #     "dimension": 768,
            #     "method": {"name": "hnsw", "space_type": "cosinesimil"}
            # },
        }
    },
}


# ── Client factory ────────────────────────────────────────────────────────────

_client: OpenSearch | None = None


def get_opensearch_client() -> OpenSearch:
    """
    Return a shared OpenSearch client (created once, reused).

    The OpenSearch Python client handles connection pooling internally,
    so a single client instance is safe to share across threads.
    """
    global _client
    if _client is None:
        _client = OpenSearch(
            hosts=[{
                "host": settings.opensearch_host,
                "port": settings.opensearch_port,
            }],
            http_compress=True,     # gzip compression on requests
            use_ssl=False,          # dev only
            verify_certs=False,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        logger.info("OpenSearch client initialised → %s", settings.opensearch_url)
    return _client


# ── Index lifecycle ───────────────────────────────────────────────────────────

def init_index() -> None:
    """
    Create the papers index if it doesn't exist.

    Safe to call on every startup. If the index exists with a different
    mapping, this is a no-op — use a migration script to handle changes.
    """
    client = get_opensearch_client()

    if client.indices.exists(index=PAPERS_INDEX):
        logger.info("OpenSearch index '%s' already exists", PAPERS_INDEX)
        return

    client.indices.create(index=PAPERS_INDEX, body=PAPERS_INDEX_CONFIG)
    logger.info("OpenSearch index '%s' created", PAPERS_INDEX)


# ── Health check ─────────────────────────────────────────────────────────────

def check_health() -> dict:
    """Return health status of the OpenSearch cluster."""
    try:
        client = get_opensearch_client()
        cluster_health = client.cluster.health()
        return {
            "status": "healthy",
            "cluster_status": cluster_health["status"],   # green / yellow / red
        }
    except OpenSearchException as exc:
        logger.error("OpenSearch health check failed: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}
    except Exception as exc:
        logger.error("OpenSearch health check unexpected error: %s", exc)
        return {"status": "unhealthy", "error": str(exc)}