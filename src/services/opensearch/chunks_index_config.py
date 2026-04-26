"""
OpenSearch index configuration for the chunks corpus.

Unlike the `papers` index (BM25 only), the `chunks` index:
  - Has `index.knn: True` to enable approximate nearest-neighbour search
  - Has an `embedding` field of type `knn_vector` (1024 dims, HNSW, cosine)
  - Stores one document per chunk — flat structure, simple kNN queries

Why a separate index from `papers`?
  A paper with 18 chunks would require 18 nested knn_vector fields if we
  stored chunks inside the paper document. Nested kNN queries are complex
  and slower. Flat structure (one doc = one chunk) is simpler and faster.

  The `arxiv_id` field in each chunk doc is the join key back to `papers`.

HNSW parameters:
  m=16:              number of bi-directional links per node during graph build.
                     Higher = better recall, more memory. 16 is a solid default.
  ef_construction=128: how thoroughly to explore the graph during index build.
                     Higher = better quality graph, slower initial indexing.
                     At query time this doesn't apply — see ef_search instead.
  space_type=cosinesimil: cosine similarity as the distance metric.
                     Best for text embeddings that are L2-normalised.
"""

CHUNKS_INDEX_NAME = "chunks"

# Hybrid search pipeline ID — must be created separately (see chunk_indexer.py)
RRF_PIPELINE_ID = "hybrid-rrf-pipeline"

CHUNKS_INDEX_CONFIG = {
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
        "index.knn":          True,   # Required to activate kNN plugin
        "analysis": {
            "analyzer": {
                "paper_analyzer": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter":    ["lowercase", "stop", "snowball"],
                }
            }
        },
    },
    "mappings": {
        "properties": {
            # ── Chunk identity ────────────────────────────────────────────────
            "chunk_id":     {"type": "keyword"},   # UUID — used as _id
            "chunk_index":  {"type": "integer"},   # 0-based position in paper
            "section_name": {"type": "keyword"},   # which section
            "word_count":   {"type": "integer"},

            # ── Back-reference to parent paper ────────────────────────────────
            "arxiv_id":     {"type": "keyword"},

            # ── Content field ─────────────────────────────────────────────────
            # Analysed for BM25 text search within hybrid queries
            "chunk_text": {
                "type":     "text",
                "analyzer": "paper_analyzer",
            },

            # ── Paper metadata (denormalised for result display) ───────────────
            # Avoids a DB round-trip to get paper title/authors after search
            "title":        {"type": "text",    "analyzer": "paper_analyzer"},
            "abstract":     {"type": "text",    "analyzer": "paper_analyzer"},
            "authors":      {"type": "keyword"},
            "categories":   {"type": "keyword"},
            "published_at": {"type": "date"},

            # ── Vector field ─────────────────────────────────────────────────
            # knn_vector: dense float vector for approximate nearest-neighbour
            # dimension MUST exactly match the embedding model output (Jina v3 = 1024)
            "embedding": {
                "type":      "knn_vector",
                "dimension": 1024,
                "method": {
                    "name":       "hnsw",          # Hierarchical Navigable Small World
                    "space_type": "cosinesimil",   # cosine similarity distance
                    "engine":     "lucene",        # available in all OpenSearch versions
                    "parameters": {
                        "m":               16,
                        "ef_construction": 128,
                    },
                },
            },
        }
    },
}