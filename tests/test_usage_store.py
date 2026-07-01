import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from usage_store import MonthlyUsageStore


def test_monthly_usage_is_aggregated_and_company_isolated(tmp_path):
    store = MonthlyUsageStore(tmp_path / "usage.sqlite3")
    try:
        store.record(
            month_utc="2026-07",
            company_id="alpha",
            provider="jina",
            model="jina-reranker-v2-base-multilingual",
            operation="reranking",
            input_tokens=1200,
            total_tokens=1200,
        )
        store.record(
            month_utc="2026-07",
            company_id="alpha",
            provider="jina",
            model="jina-reranker-v2-base-multilingual",
            operation="reranking",
            input_tokens=800,
            total_tokens=800,
        )
        store.record(
            month_utc="2026-07",
            company_id="beta",
            provider="jina",
            model="jina-reranker-v2-base-multilingual",
            operation="reranking",
            input_tokens=500,
            total_tokens=500,
        )

        alpha = store.summary("alpha", "2026-07")
        beta = store.summary("beta", "2026-07")
    finally:
        store.close()

    assert alpha["model_requests"] == 2
    assert alpha["total_tokens"] == 2000
    assert alpha["breakdown"][0]["requests"] == 2
    assert beta["model_requests"] == 1
    assert beta["total_tokens"] == 500
