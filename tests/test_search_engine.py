import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import search_engine
from bm25_index import PersistentBM25Index
from query_planner import (
    deterministic_filter_query_plan,
    extract_sort_order,
    query_filter_value_index,
)
from search_engine import ProductSearchEngine


class FakeCollection:
    pass


class CountingQueryProvider:
    def __init__(self):
        self.calls = 0

    def structured_chat(self, *_args):
        self.calls += 1
        return json.dumps(
            {
                "semantic_query": "red bike",
                "keyword_query": "red bike",
                "target_ad_type": "offer",
                "filters": {},
            }
        )


class CapturingQueryProvider(CountingQueryProvider):
    def __init__(self):
        super().__init__()
        self.system_prompt = ""

    def structured_chat(
        self,
        _model,
        system_prompt,
        _user_prompt,
        _schema,
        _temperature,
    ):
        self.calls += 1
        self.system_prompt = system_prompt
        return json.dumps(
            {
                "semantic_query": "portable recording equipment",
                "keyword_query": "camera recorder",
                "target_ad_type": "offer",
                "filters": {},
            }
        )


class DictSharedCache:
    def __init__(self):
        self.connected = True
        self.values = {}

    def get_json(self, namespace, key):
        return self.values.get((namespace, key))

    def set_json(self, namespace, key, value, _ttl_seconds):
        self.values[(namespace, key)] = value
        return True


def product_row(doc_id, **metadata):
    return {
        "doc_id": doc_id,
        "product_id": doc_id,
        "content": doc_id,
        **metadata,
    }


def build_index(path):
    index = PersistentBM25Index(path)
    index.upsert(
        [
            product_row(
                "bike-chennai",
                main_category_name="Automobiles",
                subcategory_name="Bike",
                state_name="Tamil Nadu",
                city_name="Chennai",
                rental_duration="Per Day",
                rental_fee=900,
            )
        ]
    )
    return index


def test_deterministic_filter_plan_accepts_simple_explicit_queries(tmp_path):
    index = build_index(tmp_path / "fast-plan.sqlite3")
    value_index = query_filter_value_index(index)

    bike = deterministic_filter_query_plan("bike", value_index)
    filtered = deterministic_filter_query_plan(
        "bikes in Chennai under 1000",
        value_index,
    )
    wanted = deterministic_filter_query_plan(
        "someone looking for bikes",
        value_index,
    )

    assert bike["execution_path"] == "deterministic_filter"
    assert bike["filters"]["subcategory"] == "Bike"
    assert filtered["filters"]["subcategory"] == "Bike"
    assert filtered["filters"]["city"] == "Chennai"
    assert filtered["filters"]["state"] == "Tamil Nadu"
    assert filtered["filters"]["max_rental_fee"] == 1000
    assert wanted["target_ad_type"] == "wanted"
    index.close()


def test_tenant_prompt_context_is_added_only_to_llm_planning(tmp_path):
    index = build_index(tmp_path / "tenant-prompt.sqlite3")
    provider = CapturingQueryProvider()
    engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        query_provider=provider,
        planner_prompt_context="This tenant rents professional event equipment.",
    )
    try:
        result = engine.plan("equipment for recording a distant wedding")
    finally:
        engine.close()
        index.close()

    assert result["query_plan"]["execution_path"] == "semantic"
    assert provider.calls == 1
    assert (
        "This tenant rents professional event equipment."
        in provider.system_prompt
    )


def test_disabled_tenant_llm_planner_keeps_semantic_retrieval_available(tmp_path):
    index = build_index(tmp_path / "disabled-planner.sqlite3")
    provider = CountingQueryProvider()
    engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        query_provider=provider,
        planner_enabled=False,
    )
    try:
        result = engine.plan("equipment for recording a distant wedding")
    finally:
        engine.close()
        index.close()

    assert result["query_plan"]["execution_path"] == "semantic"
    assert result["query_plan"]["semantic_query"] == (
        "equipment for recording a distant wedding"
    )
    assert provider.calls == 0


def test_deterministic_filter_plan_corrects_category_and_city_typos(tmp_path):
    index = build_index(tmp_path / "fuzzy-fast-plan.sqlite3")
    value_index = query_filter_value_index(index)

    category = deterministic_filter_query_plan("bke", value_index)
    combined = deterministic_filter_query_plan(
        "bkes in chni under 1000",
        value_index,
    )

    assert category["filters"]["subcategory"] == "Bike"
    assert category["query_corrections"] == [
        {"field": "subcategory", "input": "bke", "value": "Bike"}
    ]
    assert combined["execution_path"] == "deterministic_filter"
    assert combined["filters"]["subcategory"] == "Bike"
    assert combined["filters"]["city"] == "Chennai"
    assert combined["filters"]["state"] == "Tamil Nadu"
    assert combined["filters"]["max_rental_fee"] == 1000
    assert combined["query_corrections"] == [
        {"field": "city", "input": "chni", "value": "Chennai"},
        {"field": "subcategory", "input": "bkes", "value": "Bike"},
    ]
    index.close()


def test_deterministic_filter_plan_accepts_reordered_bare_budget_query(
    tmp_path,
):
    index = build_index(tmp_path / "reordered-fast-plan.sqlite3")
    index.upsert(
        [
            product_row(
                "car-chennai",
                main_category_name="Automobiles",
                subcategory_name="Car",
                state_name="Tamil Nadu",
                city_name="Chennai",
                rental_fee=800,
            )
        ]
    )
    value_index = query_filter_value_index(index)

    compact = deterministic_filter_query_plan(
        "1000 rent car",
        value_index,
    )
    reordered = deterministic_filter_query_plan(
        "car rent 1000 in Chennai",
        value_index,
    )
    typo = deterministic_filter_query_plan(
        "1000 bke rent in chni",
        value_index,
    )

    assert compact["execution_path"] == "deterministic_filter"
    assert compact["filters"]["subcategory"] == "Car"
    assert compact["filters"]["max_rental_fee"] == 1000
    assert reordered["filters"]["subcategory"] == "Car"
    assert reordered["filters"]["city"] == "Chennai"
    assert reordered["filters"]["max_rental_fee"] == 1000
    assert typo["filters"]["subcategory"] == "Bike"
    assert typo["filters"]["city"] == "Chennai"
    assert typo["filters"]["max_rental_fee"] == 1000
    index.close()


def test_lowest_price_query_uses_sorted_filter_path_and_corrects_retail_typo(
    tmp_path,
):
    index = build_index(tmp_path / "sorted-fast-plan.sqlite3")
    index.upsert(
        [
            product_row(
                "car-coimbatore",
                main_category_name="Automobiles",
                subcategory_name="Car",
                state_name="Tamil Nadu",
                city_name="Coimbatore",
                rental_fee=250,
            )
        ]
    )
    value_index = query_filter_value_index(index)

    plan = deterministic_filter_query_plan(
        "lowest price car retail in coimbatore",
        value_index,
    )

    assert plan["execution_path"] == "deterministic_filter"
    assert plan["sort_order"] == "price_asc"
    assert plan["filters"]["subcategory"] == "Car"
    assert plan["filters"]["city"] == "Coimbatore"
    assert plan["query_corrections"] == [
        {"field": "intent", "input": "retail", "value": "rental"}
    ]
    index.close()


def test_price_sort_fast_path_is_category_agnostic(tmp_path):
    index = PersistentBM25Index(tmp_path / "generic-price-sort.sqlite3")
    index.upsert(
        [
            product_row(
                "bike-chennai",
                main_category_name="Automobiles",
                subcategory_name="Bike",
                state_name="Tamil Nadu",
                city_name="Chennai",
                rental_duration="Per Day",
                rental_fee=100,
            ),
            product_row(
                "camera-chennai",
                main_category_name="Audio & Video Equipments",
                subcategory_name="Camera",
                state_name="Tamil Nadu",
                city_name="Chennai",
                rental_duration="Per Day",
                rental_fee=250,
            ),
            product_row(
                "room-chennai",
                main_category_name="Accommodation & Spaces",
                subcategory_name="Room",
                state_name="Tamil Nadu",
                city_name="Chennai",
                rental_duration="Per Month",
                rental_fee=5000,
            ),
        ]
    )
    value_index = query_filter_value_index(index)
    cases = (
        ("cheapest daily bike in Chennai", "Bike", "Per Day"),
        ("lowest price camera per day in Chennai", "Camera", "Per Day"),
        ("most affordable room per month in Chennai", "Room", "Per Month"),
    )

    for query, subcategory, duration in cases:
        plan = deterministic_filter_query_plan(query, value_index)
        assert plan["execution_path"] == "deterministic_filter"
        assert plan["sort_order"] == "price_asc"
        assert plan["filters"]["subcategory"] == subcategory
        assert plan["filters"]["rental_duration"] == duration

    index.close()


def test_price_sort_wording_is_extracted_deterministically():
    ascending = (
        "cheapest car",
        "lowest priced bike",
        "low rental rate camera",
        "affordable car rental",
        "budget-friendly bike",
        "price low to high",
        "low to high rental fees",
        "sort by price ascending",
        "rental rate ascending",
    )
    descending = (
        "most expensive car",
        "highest price bike",
        "rental fee high to low",
        "high to low price",
        "order by rate desc",
        "price descending",
    )

    assert all(extract_sort_order(query) == "price_asc" for query in ascending)
    assert all(extract_sort_order(query) == "price_desc" for query in descending)
    assert extract_sort_order("car under 1000") is None


def test_browse_orders_the_complete_filtered_window_by_rental_fee(tmp_path):
    index = PersistentBM25Index(tmp_path / "price-browse.sqlite3")
    index.upsert(
        [
            product_row("car-5000", city_name="Coimbatore", rental_fee=5000),
            product_row("car-250", city_name="Coimbatore", rental_fee=250),
            product_row("car-null", city_name="Coimbatore", rental_fee=None),
            product_row("car-900", city_name="Coimbatore", rental_fee=900),
            product_row("car-zero", city_name="Coimbatore", rental_fee=0),
            product_row("car-one", city_name="Coimbatore", rental_fee=1),
        ]
    )
    filters = {"categorical": {"city_name": "Coimbatore"}}

    ascending = index.browse(filters, 10, sort_order="price_asc")
    descending = index.browse(filters, 10, sort_order="price_desc")

    assert [row["doc_id"] for row in ascending] == [
        "car-250",
        "car-900",
        "car-5000",
        "car-zero",
        "car-one",
        "car-null",
    ]
    assert [row["doc_id"] for row in descending] == [
        "car-5000",
        "car-900",
        "car-250",
        "car-one",
        "car-zero",
        "car-null",
    ]
    under_1000 = index.browse(
        {
            "categorical": {"city_name": "Coimbatore"},
            "max_rental_fee": 1000,
        },
        10,
        sort_order="price_asc",
    )
    assert [row["doc_id"] for row in under_1000] == [
        "car-250",
        "car-900",
    ]
    index.close()


def test_bare_number_is_not_budget_when_it_looks_like_quantity_or_model_year(
    tmp_path,
):
    index = build_index(tmp_path / "guarded-budget-plan.sqlite3")
    index.upsert(
        [
            product_row(
                "car-chennai",
                main_category_name="Automobiles",
                subcategory_name="Car",
                state_name="Tamil Nadu",
                city_name="Chennai",
            )
        ]
    )
    value_index = query_filter_value_index(index)

    assert deterministic_filter_query_plan("2 rent car", value_index) is None
    assert deterministic_filter_query_plan("2020 rent car", value_index) is None
    assert deterministic_filter_query_plan("1000 cc car", value_index) is None
    index.close()


def test_ambiguous_category_typo_is_not_forced(tmp_path):
    index = build_index(tmp_path / "ambiguous-fuzzy-plan.sqlite3")
    index.upsert(
        [
            product_row(
                "bake-chennai",
                main_category_name="Services",
                subcategory_name="Bake",
                state_name="Tamil Nadu",
                city_name="Chennai",
            )
        ]
    )
    value_index = query_filter_value_index(index)

    assert deterministic_filter_query_plan("bke", value_index) is None
    index.close()


def test_deterministic_filter_plan_rejects_descriptive_queries(tmp_path):
    index = build_index(tmp_path / "semantic-plan.sqlite3")
    value_index = query_filter_value_index(index)

    assert deterministic_filter_query_plan(
        "red bike with ABS in Chennai",
        value_index,
    ) is None
    assert deterministic_filter_query_plan(
        "red bke with ABS in chni",
        value_index,
    ) is None
    assert deterministic_filter_query_plan(
        "vehicle for recreational driving on rough terrain",
        value_index,
    ) is None
    assert deterministic_filter_query_plan(
        "wanted bike",
        value_index,
    ) is None
    index.close()


def test_normalized_query_plan_cache_skips_repeated_provider_call(tmp_path):
    index = build_index(tmp_path / "plan-cache.sqlite3")
    provider = CountingQueryProvider()
    engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        query_provider=provider,
    )

    first = engine.plan("red bike")
    second = engine.plan("  RED   BIKE ")

    assert provider.calls == 1
    assert first["plan_cache_hit"] is False
    assert second["plan_cache_hit"] is True
    assert second["query_model_metrics"] == {}
    index.close()


def test_shared_plan_cache_survives_engine_restart(tmp_path):
    index = build_index(tmp_path / "shared-plan-cache.sqlite3")
    cache = DictSharedCache()
    first_provider = CountingQueryProvider()
    first_engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        query_provider=first_provider,
        shared_plan_cache=cache,
    )

    first = first_engine.plan("red bike")

    second_provider = CountingQueryProvider()
    second_engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        query_provider=second_provider,
        shared_plan_cache=cache,
    )
    second = second_engine.plan(" RED   BIKE ")

    assert first["plan_cache_hit"] is False
    assert second["plan_cache_hit"] is True
    assert first_provider.calls == 1
    assert second_provider.calls == 0
    assert all(
        "red" not in key
        for _namespace, key in cache.values
    )
    assert second_engine.plan_cache_health() == {
        "redis_enabled": True,
        "redis_connected": True,
        "query_plan_cache_backend": "redis+memory",
        "result_cache_enabled": True,
        "result_cache_ttl_seconds": 300,
    }
    index.close()


def test_result_id_cache_skips_repeated_search_and_invalidates_on_index_change(
    tmp_path,
    monkeypatch,
):
    index = build_index(tmp_path / "result-cache.sqlite3")
    cache = DictSharedCache()
    engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        shared_plan_cache=cache,
    )
    browse_calls = []

    def browse(*_args, **_kwargs):
        browse_calls.append(True)
        return [101, 102]

    monkeypatch.setattr(search_engine, "related_tail_product_ids", browse)
    monkeypatch.setattr(
        search_engine,
        "fetch_products_by_ids",
        lambda ids: [{"id": product_id} for product_id in ids],
    )

    first = engine.search("bike", limit=20)
    restarted_engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        shared_plan_cache=cache,
    )
    second = restarted_engine.search("  BIKE ", limit=20)

    assert first["result_cache_hit"] is False
    assert second["result_cache_hit"] is True
    assert len(browse_calls) == 1
    assert [str(product["id"]) for product in second["products"]] == [
        "101",
        "102",
    ]
    assert all(
        product["result_tier"] == "filtered"
        for product in second["products"]
    )
    assert any(
        namespace == "search_result"
        for namespace, _key in cache.values
    )

    index.upsert(
        [
            product_row(
                "second-bike",
                main_category_name="Automobiles",
                subcategory_name="Bike",
            )
        ]
    )
    third = restarted_engine.search("bike", limit=20)

    assert third["result_cache_hit"] is False
    assert len(browse_calls) == 2
    index.close()


def test_simple_query_skips_model_retrieval_and_reranking(tmp_path, monkeypatch):
    index = build_index(tmp_path / "fast-search.sqlite3")

    class FailingProvider:
        def structured_chat(self, *_args):
            raise AssertionError("The hosted query model must not be called.")

    engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
        query_provider=FailingProvider(),
    )
    monkeypatch.setattr(
        search_engine,
        "related_tail_product_ids",
        lambda *_args, **_kwargs: [101, 102],
    )
    monkeypatch.setattr(
        search_engine,
        "fetch_products_by_ids",
        lambda ids: [{"id": product_id} for product_id in ids],
    )
    monkeypatch.setattr(
        engine,
        "retrieve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Semantic retrieval must not run.")
        ),
    )
    monkeypatch.setattr(
        engine,
        "rank",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("The reranker must not run.")
        ),
    )

    result = engine.search("bike", limit=20)

    assert result["query_plan"]["execution_path"] == "deterministic_filter"
    assert result["vector_results"] == []
    assert result["reranked"] == []
    assert [product["result_tier"] for product in result["products"]] == [
        "filtered",
        "filtered",
    ]
    index.close()


def test_semantic_vector_and_bm25_retrieval_start_in_parallel(
    tmp_path,
    monkeypatch,
):
    index = build_index(tmp_path / "parallel-retrieval.sqlite3")
    engine = ProductSearchEngine(
        collection=FakeCollection(),
        bm25_index=index,
    )
    barrier = threading.Barrier(2, timeout=2)

    def vector(*_args, **_kwargs):
        barrier.wait()
        return []

    def bm25(*_args, **_kwargs):
        barrier.wait()
        return []

    monkeypatch.setattr(search_engine, "vector_search", vector)
    monkeypatch.setattr(search_engine, "bm25_search", bm25)
    monkeypatch.setattr(
        search_engine,
        "filter_candidates_by_ad_type",
        lambda candidates, *_args, **_kwargs: candidates,
    )

    result = engine.retrieve(
        {
            "semantic_query": "red bike",
            "keyword_query": "red bike",
            "target_ad_type": "offer",
            "inferred_categories": {},
        },
        {"categorical": {}},
    )

    assert result["vector_results"] == []
    assert result["bm25_results"] == []
    index.close()
