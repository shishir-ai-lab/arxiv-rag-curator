"""
Airflow DAG: daily arXiv ingestion + OpenSearch indexing.

Extended from the previous version to add a 5th stage:
  sync_to_opensearch — bulk-index all papers from PostgreSQL into OpenSearch.

Pipeline stages (updated):
  1. verify_services    → pre-flight health check (PostgreSQL + arXiv + OpenSearch)
  2. fetch_papers       → arXiv API → PDF download → Docling parse → PostgreSQL
  3. retry_failures     → re-attempt failed PDF parses
  4. sync_to_opensearch → PostgreSQL → OpenSearch bulk sync (safety net)
  5. daily_report       → summary stats for monitoring

Why a separate sync stage?
  MetadataFetcher already does write-through indexing (index on save).
  This stage is the safety net: if OpenSearch was briefly unreachable
  during ingestion, this catch-up sync ensures nothing is missed.
  It's also the right place to re-index after mapping changes.

Schedule: unchanged — 6 AM UTC, weekdays only.
"""

import logging
from datetime import datetime, timedelta
from datetime import date as date_type

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner":            "data-engineering",
    "depends_on_past":  False,
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=30),
    "execution_timeout": timedelta(hours=3),
}

CATEGORIES      = ["cs.AI", "cs.LG", "cs.CL"]
MAX_PER_CATEGORY = 50


# ── Task 1: verify services ───────────────────────────────────────────────────

def verify_services(**context) -> dict:
    """Pre-flight: check PostgreSQL, arXiv API, and OpenSearch are reachable."""
    import psycopg2
    import requests

    from ...src.core.config import settings

    issues = []

    try:
        conn = psycopg2.connect(
            host=settings.postgres_host, port=settings.postgres_port,
            dbname=settings.postgres_db, user=settings.postgres_user,
            password=settings.postgres_password,
        )
        conn.close()
        logger.info("PostgreSQL: healthy")
    except Exception as exc:
        issues.append(f"PostgreSQL: {exc}")

    try:
        r = requests.get("https://export.arxiv.org/api/query?max_results=1", timeout=10)
        r.raise_for_status()
        logger.info("arXiv API: reachable")
    except Exception as exc:
        issues.append(f"arXiv API: {exc}")

    try:
        r = requests.get(f"http://{settings.opensearch_host}:{settings.opensearch_port}/_cluster/health", timeout=10)
        r.raise_for_status()
        logger.info("OpenSearch: healthy")
    except Exception as exc:
        issues.append(f"OpenSearch: {exc}")

    if issues:
        raise RuntimeError(f"Pre-flight check failed: {issues}")

    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ── Task 2: fetch papers ──────────────────────────────────────────────────────

def fetch_daily_papers(**context) -> dict:
    """Fetch yesterday's papers for all configured categories."""
    from arxiv_rag_curator.services.factory import make_fetcher

    execution_date = context["execution_date"]
    target_date    = execution_date.date() - timedelta(days=1)
    logger.info("Fetching papers for date: %s", target_date)

    fetcher = make_fetcher()
    totals  = {"total": 0, "saved": 0, "parsed": 0, "failed": 0, "errors": []}

    for category in CATEGORIES:
        result = fetcher.fetch_for_date(
            category=category,
            target_date=target_date,
            max_results=MAX_PER_CATEGORY,
        )
        totals["total"]  += result.total
        totals["saved"]  += result.saved
        totals["parsed"] += result.parsed
        totals["failed"] += result.failed
        totals["errors"].extend(result.errors)

    logger.info("Fetch done: %s", totals)
    return totals


# ── Task 3: retry failed PDFs ─────────────────────────────────────────────────

def retry_failed_pdfs(**context) -> dict:
    """Re-attempt PDF parsing for papers where pdf_parsed = FALSE."""
    from arxiv_rag_curator.services.factory import make_fetcher

    fetcher = make_fetcher()
    result  = fetcher.retry_failed_pdfs(limit=20)
    return {"recovered": result.parsed, "still_failing": result.failed}


# ── Task 4: sync to OpenSearch ────────────────────────────────────────────────

def sync_to_opensearch(**context) -> dict:
    """
    Bulk-sync all papers from PostgreSQL to OpenSearch.

    This is a safety-net sync — MetadataFetcher already does write-through
    indexing during ingestion. This stage catches anything that slipped
    through (e.g. OpenSearch was briefly down during ingestion).

    Running a full sync daily is safe and fast because:
    - _op_type='index' is an upsert — existing docs are overwritten, not duplicated
    - 200-paper batches keep memory usage bounded
    - The sync scans latest papers first (ORDER BY created_at DESC)
      so today's new papers are indexed first
    """
    from arxiv_rag_curator.services.opensearch.factory import make_paper_indexer

    indexer = make_paper_indexer()
    result  = indexer.sync_all(batch_size=200)

    logger.info("OpenSearch sync complete: %s", result)
    return result


# ── Task 5: daily report ──────────────────────────────────────────────────────

def generate_report(**context) -> dict:
    """Build and log a summary report. Runs even if earlier tasks failed."""
    from airflow.models import TaskInstance
    ti: TaskInstance = context["ti"]

    fetch_result = ti.xcom_pull(task_ids="fetch_daily_papers") or {}
    retry_result = ti.xcom_pull(task_ids="retry_failed_pdfs")  or {}
    index_result = ti.xcom_pull(task_ids="sync_to_opensearch") or {}

    total     = fetch_result.get("total", 0)
    saved     = fetch_result.get("saved", 0)
    parsed    = fetch_result.get("parsed", 0)
    recovered = retry_result.get("recovered", 0)
    indexed   = index_result.get("total_indexed", 0)

    report = {
        "run_date":       context["execution_date"].date().isoformat(),
        "categories":     CATEGORIES,
        "papers_found":   total,
        "papers_saved":   saved,
        "pdfs_parsed":    parsed,
        "pdfs_recovered": recovered,
        "os_indexed":     indexed,
        "success_rate":   f"{saved / max(total, 1) * 100:.1f}%",
        "parse_rate":     f"{(parsed + recovered) / max(saved, 1) * 100:.1f}%",
        "index_rate":     index_result.get("success_rate", "N/A"),
    }

    logger.info("=" * 50)
    logger.info("DAILY PIPELINE REPORT")
    logger.info("=" * 50)
    for k, v in report.items():
        logger.info("  %-22s %s", k + ":", v)
    logger.info("=" * 50)

    return report


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="arxiv_daily_ingestion",
    description="Fetch, parse, store, and index daily arXiv papers",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1-5",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["ingestion", "arxiv", "opensearch", "production"],
) as dag:

    t1_verify = PythonOperator(
        task_id="verify_services",
        python_callable=verify_services,
    )
    t2_fetch = PythonOperator(
        task_id="fetch_daily_papers",
        python_callable=fetch_daily_papers,
    )
    t3_retry = PythonOperator(
        task_id="retry_failed_pdfs",
        python_callable=retry_failed_pdfs,
    )
    t4_index = PythonOperator(
        task_id="sync_to_opensearch",
        python_callable=sync_to_opensearch,
    )
    t5_report = PythonOperator(
        task_id="daily_report",
        python_callable=generate_report,
        trigger_rule="all_done",  # report even if indexing had issues
    )

    # Task graph:
    # verify → fetch → retry → index → report
    t1_verify >> t2_fetch >> t3_retry >> t4_index >> t5_report