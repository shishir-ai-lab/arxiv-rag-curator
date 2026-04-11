"""
Airflow DAG: daily arXiv paper ingestion.

Schedule: 6 AM UTC, Monday–Friday
  - arXiv publishes new papers every weekday morning
  - We fetch yesterday's papers (the most recently published batch)
  - Weekends excluded: arXiv doesn't publish on Sat/Sun

Pipeline stages:
  1. verify_services  → pre-flight health check
  2. fetch_papers     → arXiv API → PDF download → Docling parse → PostgreSQL
  3. retry_failures   → re-attempt papers that failed PDF parsing
  4. daily_report     → log summary stats for monitoring

Production patterns:
  - max_active_runs=1: never two runs at the same time
  - retries=2, retry_delay=30m: transient network issues handled automatically
  - Each task is independent: one failure doesn't block the report stage
  - catchup=False: if Airflow was down for 3 days, don't try to backfill

Trigger manually from Airflow UI or CLI:
  airflow dags trigger arxiv_daily_ingestion
"""

import logging
from datetime import date, datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── DAG default arguments ─────────────────────────────────────────────────────

DEFAULT_ARGS = {
    "owner": "data-engineering",
    "depends_on_past": False,        # each run is independent
    "email_on_failure": False,       # set True + configure SMTP for prod alerting
    "email_on_retry": False,
    "retries": 2,                    # retry the *task* on failure
    "retry_delay": timedelta(minutes=30),  # wait 30m between retries
    "execution_timeout": timedelta(hours=3),  # kill runaway tasks
}

# ── Ingestion config ──────────────────────────────────────────────────────────

CATEGORIES = ["cs.AI", "cs.LG", "cs.CL"]  # AI, ML, NLP
MAX_RESULTS_PER_CATEGORY = 50


# ── Task functions ────────────────────────────────────────────────────────────

def verify_services(**context) -> dict:
    """
    Stage 1: Pre-flight health check.

    Fail fast here rather than mid-pipeline. If PostgreSQL is down,
    there's no point downloading 50 PDFs we can't store.
    """
    import psycopg2
    import requests

    from arxiv_rag_curator.core.config import settings

    issues = []

    # Check PostgreSQL
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        conn.close()
        logger.info("PostgreSQL: healthy")
    except Exception as exc:
        issues.append(f"PostgreSQL: {exc}")
        logger.error("PostgreSQL: unhealthy — %s", exc)

    # Check arXiv API reachability (lightweight ping)
    try:
        r = requests.get("https://export.arxiv.org/api/query?max_results=1", timeout=10)
        r.raise_for_status()
        logger.info("arXiv API: reachable")
    except Exception as exc:
        issues.append(f"arXiv API: {exc}")
        logger.error("arXiv API: unreachable — %s", exc)

    if issues:
        raise RuntimeError(f"Pre-flight check failed: {issues}")

    result = {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    logger.info("All pre-flight checks passed")
    return result


def fetch_daily_papers(**context) -> dict:
    """
    Stage 2: Fetch and store yesterday's papers for all configured categories.

    Uses the Airflow execution_date to determine target_date so that
    manual backfill runs fetch the right date.
    """
    from arxiv_rag_curator.services.factory import make_fetcher

    # execution_date is the DAG's logical run date (yesterday for a daily dag)
    execution_date = context["execution_date"]
    target_date = execution_date.date() - timedelta(days=1)

    logger.info("Fetching papers for date: %s", target_date)

    fetcher = make_fetcher()
    totals = {"total": 0, "saved": 0, "parsed": 0, "failed": 0, "errors": []}

    for category in CATEGORIES:
        logger.info("Processing category: %s", category)
        result = fetcher.fetch_for_date(
            category=category,
            target_date=target_date,
            max_results=MAX_RESULTS_PER_CATEGORY,
        )
        totals["total"]  += result.total
        totals["saved"]  += result.saved
        totals["parsed"] += result.parsed
        totals["failed"] += result.failed
        totals["errors"].extend(result.errors)

    logger.info(
        "Fetch complete: total=%d saved=%d parsed=%d failed=%d",
        totals["total"], totals["saved"], totals["parsed"], totals["failed"],
    )
    return totals


def retry_failed_pdfs(**context) -> dict:
    """
    Stage 3: Re-attempt PDF parsing for papers that failed earlier.

    Docling occasionally fails on first attempt due to:
      - Unusual PDF encoding
      - Memory pressure during large batches
      - Transient conversion errors

    A second attempt often succeeds. We limit to 20 retries per run
    to avoid consuming the full execution window on retries.
    """
    from arxiv_rag_curator.services.factory import make_fetcher

    fetcher = make_fetcher()
    result = fetcher.retry_failed_pdfs(limit=20)

    logger.info("Retry complete: recovered=%d still_failing=%d", result.parsed, result.failed)
    return {"recovered": result.parsed, "still_failing": result.failed}


def generate_report(**context) -> dict:
    """
    Stage 4: Build a summary report for this run.

    This runs regardless of whether earlier stages succeeded or failed
    (trigger_rule="all_done") so you always get a status report.
    Visible in Airflow's task logs and XCom.
    """
    from airflow.models import TaskInstance

    # Pull results from upstream tasks via XCom
    # XCom = cross-task communication; each task's return value is stored here
    ti: TaskInstance = context["ti"]

    fetch_result = ti.xcom_pull(task_ids="fetch_daily_papers") or {}
    retry_result = ti.xcom_pull(task_ids="retry_failed_pdfs") or {}

    total   = fetch_result.get("total", 0)
    saved   = fetch_result.get("saved", 0)
    parsed  = fetch_result.get("parsed", 0)
    failed  = fetch_result.get("failed", 0)
    recovered = retry_result.get("recovered", 0)

    success_rate = f"{saved / max(total, 1) * 100:.1f}%"
    parse_rate   = f"{(parsed + recovered) / max(saved, 1) * 100:.1f}%"

    report = {
        "run_date":      context["execution_date"].date().isoformat(),
        "categories":    CATEGORIES,
        "papers_found":  total,
        "papers_saved":  saved,
        "pdfs_parsed":   parsed,
        "pdfs_recovered": recovered,
        "papers_failed": failed,
        "success_rate":  success_rate,
        "parse_rate":    parse_rate,
    }

    logger.info("=" * 50)
    logger.info("DAILY INGESTION REPORT")
    logger.info("=" * 50)
    for key, val in report.items():
        logger.info("  %-20s %s", key + ":", val)
    logger.info("=" * 50)

    return report


# ── DAG definition ────────────────────────────────────────────────────────────

with DAG(
    dag_id="arxiv_daily_ingestion",
    description="Fetch, parse, and store daily arXiv papers for cs.AI, cs.LG, cs.CL",
    default_args=DEFAULT_ARGS,
    schedule="0 6 * * 1-5",     # 6 AM UTC, Mon-Fri
    start_date=datetime(2025, 1, 1),
    catchup=False,               # don't backfill missed days on restart
    max_active_runs=1,           # never run two instances simultaneously
    tags=["ingestion", "arxiv", "production"],
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

    t4_report = PythonOperator(
        task_id="daily_report",
        python_callable=generate_report,
        trigger_rule="all_done",  # run even if earlier tasks failed
    )

    # Task dependency graph:
    # verify → fetch → retry → report
    t1_verify >> t2_fetch >> t3_retry >> t4_report