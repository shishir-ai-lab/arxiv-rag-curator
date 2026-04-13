"""
Query builder for BM25 keyword search.

Responsibility: translate human-readable search parameters into
OpenSearch Query DSL dicts. Nothing else — no network calls, no state.

This separation means:
  - Query logic is unit-testable without a running OpenSearch
  - Query shapes are inspectable and debuggable in a notebook
  - The search service stays thin — it just sends what the builder produces

Field boosting rationale:
  title^3    — a title match is the strongest signal; the author named
               the paper to summarise its contribution
  abstract^2 — dense signal; authors carefully chose every word
  full_text^1 — broader signal; might contain tangential mentions

Query type — 'best_fields':
  Uses the score from the single highest-scoring field, with a
  tie_breaker for partial credit from other fields. Good for queries
  where the user is looking for the main topic, not a property spread
  across all fields.

  Contrast with 'most_fields': sums scores across fields. Risks
  double-counting when the same concept appears in title AND abstract.
"""

from datetime import date
from typing import Optional


# Default fields and boosts — override via build_query(fields=...)
DEFAULT_FIELDS = ["title^3", "abstract^2", "full_text^1"]

# How many highlight fragments to return per field
HIGHLIGHT_CONFIG = {
    "pre_tags":  ["<mark>"],
    "post_tags": ["</mark>"],
    "fields": {
        "title":    {"number_of_fragments": 0},        # always return full title
        "abstract": {"number_of_fragments": 2, "fragment_size": 200},
    },
}

# Never return full_text in search results — it can be megabytes
SOURCE_EXCLUDES = ["full_text"]

# Hard cap on result size — prevents accidental full-index dumps
MAX_RESULTS = 100


class QueryBuilder:
    """
    Builds OpenSearch Query DSL dicts from search parameters.

    All methods return plain dicts — no OpenSearch SDK dependency.
    The caller (SearchService) is responsible for executing them.

    Usage:
        builder = QueryBuilder()
        query = builder.bm25(
            text="transformer attention",
            categories=["cs.AI"],
            size=10,
        )
        # Pass query dict directly to opensearch_client.search(body=query)
    """

    def bm25(
        self,
        text: str,
        categories:  Optional[list[str]] = None,
        date_from:   Optional[date] = None,
        date_to:     Optional[date] = None,
        fields:      Optional[list[str]] = None,
        from_:       int = 0,
        size:        int = 10,
        highlight:   bool = True,
        sort_by:     str = "relevance",   # 'relevance' | 'date_desc' | 'date_asc'
    ) -> dict:
        """
        Build a BM25 multi-field search query.

        The bool query structure:
          must   → multi_match on text fields  (drives relevance score)
          filter → category and date filters   (binary, cached, no score effect)
        """
        size = min(size, MAX_RESULTS)
        search_fields = fields or DEFAULT_FIELDS

        # Core relevance query
        multi_match = {
            "multi_match": {
                "query":       text,
                "fields":      search_fields,
                "type":        "best_fields",
                "operator":    "or",         # any word in query can match
                "fuzziness":   "AUTO",       # handles 1–2 char typos automatically
                "tie_breaker": 0.3,          # partial credit for secondary fields
            }
        }

        # Filter clauses: pure binary inclusion, not scored, cached by OpenSearch
        filters = self._build_filters(categories, date_from, date_to)

        bool_query: dict = {"must": [multi_match]}
        if filters:
            bool_query["filter"] = filters

        query: dict = {
            "from":    from_,
            "size":    size,
            "query":   {"bool": bool_query},
            "_source": {"excludes": SOURCE_EXCLUDES},
        }

        if highlight:
            query["highlight"] = HIGHLIGHT_CONFIG

        if sort_by == "date_desc":
            query["sort"] = [{"published_at": {"order": "desc"}}, "_score"]
        elif sort_by == "date_asc":
            query["sort"] = [{"published_at": {"order": "asc"}}, "_score"]
        # default: pure relevance — no explicit sort needed

        return query

    def phrase(
        self,
        phrase: str,
        field: str = "abstract",
        slop: int = 1,
    ) -> dict:
        """
        Exact phrase search — words must appear in order (±slop words apart).

        Use when users search for specific named concepts like
        'chain of thought' or 'mixture of experts'.
        """
        return {
            "size": 10,
            "query": {
                "match_phrase": {
                    field: {"query": phrase, "slop": slop}
                }
            },
            "_source": {"excludes": SOURCE_EXCLUDES},
        }

    def by_category(
        self,
        categories: list[str],
        from_: int = 0,
        size: int = 20,
        sort_by: str = "date_desc",
    ) -> dict:
        """
        Return all papers in given categories, sorted by date.
        No text relevance — pure filter + sort.
        """
        size = min(size, MAX_RESULTS)
        sort_order = "desc" if sort_by == "date_desc" else "asc"

        return {
            "from": from_,
            "size": size,
            "query": {
                "bool": {
                    "filter": [{"terms": {"categories": categories}}]
                }
            },
            "sort": [{"published_at": {"order": sort_order}}],
            "_source": {"excludes": SOURCE_EXCLUDES},
        }

    def count_by_category(self) -> dict:
        """
        Aggregation query: count papers per category.
        Used for search analytics and the API's stats endpoint.
        size=0: don't return documents, only aggregation results.
        """
        return {
            "size": 0,
            "aggs": {
                "by_category": {
                    "terms": {"field": "categories", "size": 20}
                },
                "pdf_parsed_count": {
                    "filter": {"term": {"pdf_parsed": True}}
                },
                "date_histogram": {
                    "date_histogram": {
                        "field":              "published_at",
                        "calendar_interval": "month",
                    }
                },
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_filters(
        self,
        categories: Optional[list[str]],
        date_from:  Optional[date],
        date_to:    Optional[date],
    ) -> list:
        """Build filter clauses. Returns empty list if no filters requested."""
        filters = []

        if categories:
            # 'terms' = match any value in the list
            # Works because 'categories' is mapped as 'keyword' (exact match)
            filters.append({"terms": {"categories": categories}})

        if date_from or date_to:
            date_range: dict = {}
            if date_from:
                date_range["gte"] = date_from.isoformat()
            if date_to:
                date_range["lte"] = date_to.isoformat()
            filters.append({"range": {"published_at": date_range}})

        return filters