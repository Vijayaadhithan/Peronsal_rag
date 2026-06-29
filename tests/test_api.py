import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from api import (
    ExpiredCursorError,
    ProductSearchService,
    SearchRequest,
    SearchSessionStore,
    create_app,
)


class FakeBM25Index:
    def count(self):
        return 12


class FakeEngine:
    def __init__(self):
        self.bm25_index = FakeBM25Index()
        self.calls = []
        self.ranker = None
        self.reranker_loads = 0

    def ensure_reranker(self):
        if self.ranker is not None:
            return 0.0
        self.reranker_loads += 1
        self.ranker = object()
        return 0.123

    def search(self, query, limit=None):
        self.calls.append((query, limit))
        return {
            "query_plan": {
                "semantic_query": query,
                "keyword_query": query,
                "target_ad_type": "offer",
                "sort_order": "price_asc",
            },
            "resolved_filters": {"categorical": {"city_name": "Chennai"}},
            "unresolved_filters": {},
            "seconds": 0.01,
            "vector_seconds": 0.02,
            "bm25_seconds": 0.003,
            "reranker_load_seconds": 0.0,
            "reranker_seconds": 0.04,
            "products": [
                {
                    "id": number,
                    "title": f"Product {number}",
                    "mobile": "private",
                }
                for number in range(1, 6)
            ],
        }


def test_cursor_pages_one_stable_search_without_repeating_engine_work():
    engine = FakeEngine()
    service = ProductSearchService(engine, max_results=20)

    first = service.search(SearchRequest(query="camera", page_size=2))
    second = service.search(
        SearchRequest(cursor=first.pagination.next_cursor, page_size=2)
    )
    third = service.search(
        SearchRequest(cursor=second.pagination.next_cursor, page_size=2)
    )

    assert [item["id"] for item in first.items] == [1, 2]
    assert "mobile" not in first.items[0]
    assert [item["id"] for item in second.items] == [3, 4]
    assert [item["id"] for item in third.items] == [5]
    assert first.search_id == second.search_id == third.search_id
    assert first.pagination.total_results == 5
    assert third.pagination.has_more is False
    assert third.pagination.next_cursor is None
    assert engine.calls == [("camera", 20)]


def test_default_api_page_contains_twenty_products():
    assert SearchRequest(query="camera").page_size == 20


def test_ranked_results_fill_first_three_pages_before_related_tail():
    class TieredEngine(FakeEngine):
        def search(self, query, limit=None):
            result = super().search(query, limit)
            result["products"] = [
                {
                    "id": number,
                    "title": f"Product {number}",
                    "result_tier": "ranked" if number <= 60 else "related",
                }
                for number in range(1, 66)
            ]
            return result

    service = ProductSearchService(TieredEngine(), max_results=65)
    pages = []
    response = service.search(SearchRequest(query="bike", page_size=20))
    pages.append(response)
    while response.pagination.next_cursor:
        response = service.search(
            SearchRequest(
                cursor=response.pagination.next_cursor,
                page_size=20,
            )
        )
        pages.append(response)

    assert len(pages) == 4
    assert all(
        item["result_tier"] == "ranked"
        for page in pages[:3]
        for item in page.items
    )
    assert [item["id"] for item in pages[3].items] == [61, 62, 63, 64, 65]
    assert all(
        item["result_tier"] == "related"
        for item in pages[3].items
    )


def test_expired_cursor_requires_a_new_query():
    now = [100.0]
    sessions = SearchSessionStore(ttl_seconds=5, clock=lambda: now[0])
    service = ProductSearchService(FakeEngine(), sessions=sessions)
    first = service.search(SearchRequest(query="camera", page_size=2))

    now[0] = 106.0

    try:
        service.search(
            SearchRequest(cursor=first.pagination.next_cursor, page_size=2)
        )
    except ExpiredCursorError as exc:
        assert "expired" in str(exc)
    else:
        raise AssertionError("Expected the cursor to expire")


def test_http_contract_and_validation():
    service = ProductSearchService(FakeEngine(), max_results=20)
    with TestClient(create_app(service=service)) as client:
        health = client.get("/api/v1/health")
        response = client.post(
            "/api/v1/search",
            json={"query": "camera in Chennai", "page_size": 2},
        )
        invalid = client.post(
            "/api/v1/search",
            json={"query": "camera", "cursor": "also-set"},
        )

    assert health.status_code == 200
    assert health.json()["indexed_products"] == 12
    assert health.json()["reranker_loaded"] is True
    assert health.json()["reranker_load_ms"] == 123.0
    assert health.json()["redis_enabled"] is False
    assert health.json()["redis_connected"] is False
    assert health.json()["query_plan_cache_backend"] == "memory"
    assert health.json()["result_cache_enabled"] is False
    assert service.engine.reranker_loads == 1
    assert response.status_code == 200
    assert response.json()["pagination"]["returned"] == 2
    assert response.json()["pagination"]["has_more"] is True
    assert response.json()["interpreted_query"]["query_corrections"] == []
    assert response.json()["interpreted_query"]["sort_order"] == "price_asc"
    assert response.json()["interpreted_query"]["result_cache_hit"] is False
    assert invalid.status_code == 422


def test_repeated_query_result_cache_is_reported_by_api():
    class ResultCachingEngine(FakeEngine):
        def search(self, query, limit=None):
            result = super().search(query, limit)
            result["result_cache_hit"] = len(self.calls) > 1
            result["result_cache_seconds"] = 0.002
            return result

    service = ProductSearchService(ResultCachingEngine(), max_results=20)

    first = service.search(SearchRequest(query="camera", page_size=2))
    repeated = service.search(SearchRequest(query="camera", page_size=2))

    assert first.cached is False
    assert first.interpreted_query["result_cache_hit"] is False
    assert repeated.cached is True
    assert repeated.interpreted_query["result_cache_hit"] is True
    assert repeated.timings_ms["result_cache"] == 2.0


def test_api_keeps_wanted_status_two_and_excludes_soft_deleted_products():
    engine = FakeEngine()
    original_search = engine.search

    def search_with_hidden_products(query, limit=None):
        result = original_search(query, limit)
        result["products"] = [
            {
                "id": 1,
                "type": "1",
                "title": "Offer",
                "status": "1",
                "deleted_at": None,
            },
            {
                "id": 2,
                "type": "2",
                "title": "Wanted",
                "status": "2",
                "deleted_at": None,
            },
            {
                "id": 3,
                "type": "2",
                "title": "Deleted",
                "status": "2",
                "deleted_at": "2025-01-01",
            },
        ]
        return result

    engine.search = search_with_hidden_products
    service = ProductSearchService(engine)

    response = service.search(SearchRequest(query="camera"))

    assert response.items == [
        {"id": 1, "type": "1", "title": "Offer"},
        {"id": 2, "type": "2", "title": "Wanted"},
    ]
