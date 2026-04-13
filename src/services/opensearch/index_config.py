"""
OpenSearch index configuration for the papers corpus.

Why configuration-as-code?
  - Index mappings are immutable after creation. Getting them wrong means a
    full reindex operation. Having them in code makes them reviewable,
    version-controlled, and reproducible across environments.
  - BM25 parameters (k1, b) are tunable — explicit values make it clear
    these are deliberate engineering decisions, not defaults we fell into.

Mapping decisions:
  - title/abstract/full_text → 'text' with paper_analyzer (BM25 search)
  - arxiv_id/authors/categories → 'keyword' (exact filters/aggregations)
  - published_at → 'date' (range queries, recency sort)
  - title also has a .keyword sub-field for exact sorting

Analyzer decisions:
  - standard tokenizer: splits on whitespace and punctuation
  - lowercase filter: 'Transformer' and 'transformer' are the same query
  - stop filter: removes 'the', 'is', 'at' — they add noise, not signal
  - snowball filter: stems 'running' → 'run', 'papers' → 'paper'
    Conservative choice — aggressive stemmers mangle technical terms
"""

INDEX_NAME = "papers"

# BM25 parameters
# k1 = 1.2: TF saturation. Diminishing returns after ~4 occurrences of a term.
#            Higher values keep rewarding repetition. 1.2 is the OpenSearch default.
# b  = 0.75: Length normalisation. Academic papers vary widely in length.
#            0.75 applies moderate penalisation for longer documents.
#            Set to 0.0 if you don't want length to matter at all.
BM25_K1 = 1.2
BM25_B  = 0.75

INDEX_CONFIG = {
    "settings": {
        "number_of_shards":   1,   # single node dev setup; increase in prod
        "number_of_replicas": 0,   # no replicas on a single node

        # Named similarity profile — referenced by field mappings below
        "similarity": {
            "bm25_tuned": {
                "type": "BM25",
                "k1": BM25_K1,
                "b":  BM25_B,
                # Don't count overlap tokens (stop-words, synonyms) in doc length
                "discount_overlaps": True,
            }
        },

        "analysis": {
            "analyzer": {
                "paper_analyzer": {
                    "type":      "custom",
                    "tokenizer": "standard",
                    "filter": [
                        "lowercase",   # case-insensitive matching
                        "stop",        # remove English stop words
                        "snowball",    # light stemming (English)
                    ],
                }
            }
        },
    },

    "mappings": {
        "properties": {

            # ── Exact-match fields ────────────────────────────────────────────
            # keyword type: stored as-is, no tokenisation
            # Used for: term filters, aggregations, sorting
            "arxiv_id":   {"type": "keyword"},
            "authors":    {"type": "keyword"},
            "categories": {"type": "keyword"},
            "pdf_parsed": {"type": "boolean"},

            # ── Full-text search fields ───────────────────────────────────────
            # text type: analysed with paper_analyzer before indexing
            # Used for: BM25 relevance search
            "title": {
                "type":       "text",
                "analyzer":   "paper_analyzer",
                "similarity": "bm25_tuned",
                # .keyword sub-field: allows exact sort/aggregation on title
                # without re-indexing. Two fields for the price of one.
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 512}
                },
            },
            "abstract": {
                "type":       "text",
                "analyzer":   "paper_analyzer",
                "similarity": "bm25_tuned",
            },
            "full_text": {
                "type":       "text",
                "analyzer":   "paper_analyzer",
                "similarity": "bm25_tuned",
                # Store term offsets for fast highlighting.
                # Costs ~20% more disk space on this field; worth it for UX.
                "index_options": "offsets",
            },

            # ── Date field ───────────────────────────────────────────────────
            # Accepts ISO 8601 strings or epoch millis
            "published_at": {"type": "date"},
        }
    },
}