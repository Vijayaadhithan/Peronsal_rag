import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bm25_index import PersistentBM25Index
from chat import (
    MYSQL_RESULT_ID_COLUMN,
    MYSQL_RESULT_TABLE,
    MYSQL_SEARCH_ID_COLUMN,
    MYSQL_TABLE,
    enrich_query_plan,
    extract_duration_filter,
    extract_price_constraints,
    extract_product_ids,
    fetch_products_by_ids,
    filter_candidates_by_ad_type,
    infer_target_ad_type,
    merge_results,
    metadata_matches_filters,
    parse_query_plan,
    query_filter_value_index,
    rerank,
    resolve_query_filters,
)
from retrieval import related_tail_product_ids


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = None
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.fake_cursor = FakeCursor(rows)

    def cursor(self):
        return self.fake_cursor


class FakeReranker:
    def compute_score(self, pairs, **_kwargs):
        return [0.1 if "first" in passage else 0.9 for _, passage in pairs]


def product_index_row(doc_id, content, **metadata):
    return {
        "doc_id": doc_id,
        "product_id": metadata.pop("product_id", doc_id),
        "content": content,
        **metadata,
    }


def test_parse_query_plan_normalizes_fields_and_price_range():
    plan = parse_query_plan(
        """
        {
          "semantic_query": " road bike ",
          "keyword_query": " Shimano road bike ",
          "filters": {
            "city": " Chennai ",
            "min_rental_fee": 5000,
            "max_rental_fee": 1000
          }
        }
        """,
        "original query",
    )

    assert plan["semantic_query"] == "road bike"
    assert plan["keyword_query"] == "Shimano road bike"
    assert plan["filters"]["city"] == "Chennai"
    assert plan["filters"]["min_rental_fee"] == 1000
    assert plan["filters"]["max_rental_fee"] == 5000


def test_parse_query_plan_drops_inferred_parent_filters():
    plan = parse_query_plan(
        """
        {
          "semantic_query": "bachelor mansion",
          "keyword_query": "bachelor mansion",
          "filters": {
            "main_category": "Accommodation & Spaces",
            "subcategory": "Mansion",
            "state": "Tamil Nadu",
            "city": "Coimbatore"
          }
        }
        """,
        "bachelor mansion in Coimbatore",
    )

    assert plan["filters"]["main_category"] is None
    assert plan["filters"]["state"] is None
    assert plan["filters"]["subcategory"] == "Mansion"
    assert plan["filters"]["city"] == "Coimbatore"


def test_parse_query_plan_keeps_guessed_category_soft():
    plan = parse_query_plan(
        """
        {
          "semantic_query": "portable device that records distant subjects",
          "keyword_query": "portable recording distant subjects",
          "target_ad_type": "offer",
          "filters": {
            "main_category": "Audio & Video Equipments",
            "subcategory": "Camera"
          }
        }
        """,
        "something portable that records a wedding clearly from far away",
    )

    assert plan["filters"]["main_category"] is None
    assert plan["filters"]["subcategory"] is None
    assert plan["inferred_categories"] == {
        "main_category": "Audio & Video Equipments",
        "subcategory": "Camera",
    }


def test_functional_category_word_does_not_become_hard_filter():
    plan = parse_query_plan(
        """
        {
          "semantic_query": "vehicle for recreational driving on rough terrain",
          "keyword_query": "off-road vehicle ATV",
          "target_ad_type": "offer",
          "filters": {
            "main_category": "Personal & Home Services",
            "subcategory": "Driving"
          }
        }
        """,
        "A vehicle for recreational driving on rough terrain.",
    )
    value_index = {
        "main_category": {
            "personal & home services": "Personal & Home Services"
        },
        "subcategory": {"driving": "Driving"},
        "state": {},
        "city": {},
        "locality": {},
        "rental_duration": {},
        "_subcategory_main_category": {
            "driving": "Personal & Home Services"
        },
        "_city_state": {},
        "_locality_location": {},
    }

    enriched = enrich_query_plan(
        "A vehicle for recreational driving on rough terrain.",
        plan,
        value_index,
    )

    assert enriched["filters"]["main_category"] is None
    assert enriched["filters"]["subcategory"] is None
    assert enriched["inferred_categories"]["main_category"] is None
    assert enriched["inferred_categories"]["subcategory"] is None


def test_unique_keyword_concept_becomes_soft_subcategory_hint():
    plan = {
        "semantic_query": "vehicle for rough terrain",
        "keyword_query": "off-road vehicle, 4x4, SUV, ATV",
        "target_ad_type": "offer",
        "filters": {
            "main_category": None,
            "subcategory": None,
            "state": None,
            "city": None,
            "locality": None,
            "rental_duration": None,
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "inferred_categories": {
            "main_category": None,
            "subcategory": None,
        },
        "fallback_reason": None,
    }
    value_index = {
        "main_category": {"sports & toys": "Sports & Toys"},
        "subcategory": {
            "atv bike": "ATV Bike",
            "quad bike": "Quad Bike",
            "mountain bike": "Mountain Bike",
        },
        "state": {},
        "city": {},
        "locality": {},
        "rental_duration": {},
        "_subcategory_main_category": {"atv bike": "Sports & Toys"},
        "_city_state": {},
        "_locality_location": {},
    }

    enriched = enrich_query_plan(
        "A vehicle for recreational driving on rough terrain.",
        plan,
        value_index,
    )

    assert enriched["filters"]["subcategory"] is None
    assert enriched["inferred_categories"]["subcategory"] == "ATV Bike"
    assert enriched["inferred_categories"]["main_category"] == "Sports & Toys"


def test_rough_terrain_query_gets_deterministic_atv_expansion():
    plan = {
        "semantic_query": "vehicle for recreational driving on rough terrain",
        "keyword_query": "vehicle rough terrain recreational driving",
        "target_ad_type": "offer",
        "filters": {
            "main_category": None,
            "subcategory": None,
            "state": None,
            "city": None,
            "locality": None,
            "rental_duration": None,
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "inferred_categories": {
            "main_category": None,
            "subcategory": None,
        },
        "fallback_reason": None,
    }
    value_index = {
        "main_category": {"sports & toys": "Sports & Toys"},
        "subcategory": {
            "atv bike": "ATV Bike",
            "quad bike": "Quad Bike",
        },
        "state": {},
        "city": {},
        "locality": {},
        "rental_duration": {},
        "_subcategory_main_category": {"atv bike": "Sports & Toys"},
        "_city_state": {},
        "_locality_location": {},
    }

    enriched = enrich_query_plan(
        "A vehicle for recreational driving on rough terrain.",
        plan,
        value_index,
    )

    assert "ATV" in enriched["keyword_query"]
    assert enriched["filters"]["subcategory"] is None
    assert enriched["inferred_categories"]["subcategory"] == "ATV Bike"


def test_enrich_query_plan_restores_exact_filters_and_keyword_intent():
    plan = {
        "semantic_query": "bachelor mansion",
        "keyword_query": "under 1500 per day",
        "filters": {
            "main_category": None,
            "subcategory": "Mansion",
            "state": None,
            "city": None,
            "locality": None,
            "rental_duration": "Per Day",
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "fallback_reason": None,
    }
    value_index = {
        "main_category": {},
        "subcategory": {"mansion": "Mansion"},
        "state": {},
        "city": {"coimbatore": "Coimbatore"},
        "locality": {"coimbatore": "Coimbatore"},
        "rental_duration": {"per day": "Per Day"},
    }

    enriched = enrich_query_plan(
        "bachelor mansion in Coimbatore under 1500 per day",
        plan,
        value_index,
    )

    assert enriched["keyword_query"] == "bachelor mansion"
    assert enriched["filters"]["city"] == "Coimbatore"
    assert enriched["filters"]["locality"] is None
    assert enriched["filters"]["max_rental_fee"] == 1500


def test_enrich_query_plan_overrides_wrong_duration_category_and_intent():
    plan = {
        "semantic_query": "rent car",
        "keyword_query": "car rental",
        "target_ad_type": "wanted",
        "filters": {
            "main_category": None,
            "subcategory": "Caravan",
            "state": None,
            "city": None,
            "locality": None,
            "rental_duration": "Per Week",
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "fallback_reason": None,
    }
    value_index = {
        "main_category": {"automobiles": "Automobiles"},
        "subcategory": {"car": "Car", "caravan": "Caravan"},
        "state": {},
        "city": {"coimbatore": "Coimbatore"},
        "locality": {},
        "rental_duration": {
            "per hour": "Per Hour",
            "per week": "Per Week",
        },
    }

    enriched = enrich_query_plan(
        "hourly rental car within 800 in Coimbatore",
        plan,
        value_index,
    )

    assert enriched["target_ad_type"] == "offer"
    assert enriched["filters"]["subcategory"] == "Car"
    assert enriched["filters"]["city"] == "Coimbatore"
    assert enriched["filters"]["rental_duration"] == "Per Hour"
    assert enriched["filters"]["max_rental_fee"] == 800


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("hourly car rental", "Per Hour"),
        ("bike for a day", "Per Day"),
        ("weekly bike rental", "Per Week"),
        ("car for one month", "Per Month"),
        ("taxi per ride", "Per Ride"),
    ],
)
def test_extract_duration_filter_maps_natural_language(query, expected):
    values = {
        normalize.lower(): normalize
        for normalize in ("Per Hour", "Per Day", "Per Week", "Per Month", "Per Ride")
    }
    assert extract_duration_filter(query, values) == expected


def test_infer_target_ad_type_uses_searcher_perspective():
    assert infer_target_ad_type("I need a bike for a week") == "offer"
    assert infer_target_ad_type("looking for a rental car") == "offer"
    assert infer_target_ad_type("show people who need a rental car") == "wanted"
    assert infer_target_ad_type("someone looking for bikes") == "wanted"
    assert infer_target_ad_type("a person who needs a bike") == "wanted"
    assert infer_target_ad_type("show wanted ads for bikes") == "wanted"


def test_extract_price_constraints_handles_range_and_minimum():
    assert extract_price_constraints("bike between Rs 1000 and Rs 2500") == (
        1000,
        2500,
    )
    assert extract_price_constraints("room above ₹750") == (750, None)
    assert extract_price_constraints("hourly car within 800") == (None, 800)
    assert extract_price_constraints("bike in 1000 range per hour") == (None, 1000)


def test_enrich_query_plan_handles_people_seeking_hourly_bikes():
    plan = {
        "semantic_query": "bike",
        "keyword_query": "bike",
        "target_ad_type": "offer",
        "filters": {
            "main_category": None,
            "subcategory": None,
            "state": None,
            "city": None,
            "locality": None,
            "rental_duration": None,
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "fallback_reason": None,
    }
    value_index = {
        "main_category": {},
        "subcategory": {"bike": "Bike"},
        "state": {},
        "city": {},
        "locality": {},
        "rental_duration": {"per hour": "Per Hour"},
    }

    enriched = enrich_query_plan(
        "someone looking for bikes in 1000 range per hour",
        plan,
        value_index,
    )

    assert enriched["target_ad_type"] == "wanted"
    assert enriched["filters"]["subcategory"] == "Bike"
    assert enriched["filters"]["rental_duration"] == "Per Hour"
    assert enriched["filters"]["max_rental_fee"] == 1000


def test_persistent_bm25_index_searches_without_loading_corpus(tmp_path):
    index = PersistentBM25Index(tmp_path / "bm25.sqlite3")
    index.upsert(
        [
            product_index_row("1", "ZXMODEL road bicycle"),
            product_index_row("2", "unrelated terms"),
            product_index_row("3", "different keywords"),
        ]
    )

    results = index.search("ZXMODEL", {"categorical": {}}, top_k=2)

    assert [result["doc_id"] for result in results] == ["1"]
    assert index.count() == 3
    index.close()


def test_filters_resolve_case_insensitively_and_apply_to_both_searches(tmp_path):
    index = PersistentBM25Index(tmp_path / "filters.sqlite3")
    index.upsert(
        [
            product_index_row(
                "1",
                "bike",
                city_name="Coimbatore",
                rental_fee=900.0,
            ),
            product_index_row(
                "2",
                "bike",
                city_name="Chennai",
                rental_fee=1200.0,
            ),
            product_index_row(
                "3",
                "bike",
                city_name="Coimbatore",
                rental_fee=1.0,
            ),
        ]
    )
    filters = {
        "city": "coimbatore",
        "min_rental_fee": 500,
        "max_rental_fee": 1000,
    }

    resolved, unresolved = resolve_query_filters(
        filters,
        query_filter_value_index(index),
    )

    assert unresolved == {}
    assert resolved["categorical"] == {"city_name": "Coimbatore"}
    assert [
        row["doc_id"] for row in index.search("bike", resolved, top_k=10)
    ] == ["1"]
    assert metadata_matches_filters(
        {
            "source_file": "mysql:test.ads_search_ready",
            "city_name": "Coimbatore",
            "rental_fee": 900,
        },
        "mysql:test.ads_search_ready",
        resolved,
    )
    assert not metadata_matches_filters(
        {
            "source_file": "mysql:test.ads_search_ready",
            "city_name": "Chennai",
            "rental_fee": 900,
        },
        "mysql:test.ads_search_ready",
        resolved,
    )
    assert not metadata_matches_filters(
        {
            "source_file": "mysql:test.ads_search_ready",
            "city_name": "Coimbatore",
            "rental_fee": 1,
        },
        "mysql:test.ads_search_ready",
        resolved,
    )
    index.close()


def test_city_alias_and_subcategory_parent_are_resolved_from_index(tmp_path):
    index = PersistentBM25Index(tmp_path / "taxonomy.sqlite3")
    index.upsert(
        [
            product_index_row(
                "1",
                "quad bike",
                main_category_name="Sports & Toys",
                subcategory_name="Quad Bike",
                city_name="Bengaluru",
                rental_duration="Per Hour",
            )
        ]
    )
    value_index = query_filter_value_index(index)
    plan = {
        "semantic_query": "Quad Bike",
        "keyword_query": "Quad Bike",
        "target_ad_type": "offer",
        "filters": {
            "main_category": None,
            "subcategory": "Quad Bike",
            "state": None,
            "city": "bangalore",
            "locality": "Bangalore",
            "rental_duration": "Per Hour",
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "fallback_reason": None,
    }

    enriched = enrich_query_plan(
        "Quad Bike for Hourly Rent in bangalore",
        plan,
        value_index,
    )
    resolved, unresolved = resolve_query_filters(
        enriched["filters"],
        value_index,
    )

    assert enriched["filters"]["main_category"] == "Sports & Toys"
    assert enriched["filters"]["subcategory"] == "Quad Bike"
    assert enriched["filters"]["city"] == "Bengaluru"
    assert enriched["filters"]["locality"] is None
    assert unresolved == {}
    assert resolved["categorical"] == {
        "main_category_name": "Sports & Toys",
        "subcategory_name": "Quad Bike",
        "city_name": "Bengaluru",
        "rental_duration": "Per Hour",
    }
    index.close()


def test_ambiguous_subcategory_does_not_force_parent_category(tmp_path):
    index = PersistentBM25Index(tmp_path / "ambiguous-taxonomy.sqlite3")
    index.upsert(
        [
            product_index_row(
                "1",
                "computer technician",
                main_category_name="IT & ITES Services",
                subcategory_name="Technician",
            ),
            product_index_row(
                "2",
                "home technician",
                main_category_name="Personal & Home Services",
                subcategory_name="Technician",
            ),
        ]
    )

    value_index = query_filter_value_index(index)

    assert "technician" not in value_index["_subcategory_main_category"]
    index.close()


def test_unique_locality_derives_city_and_state(tmp_path):
    index = PersistentBM25Index(tmp_path / "location-hierarchy.sqlite3")
    index.upsert(
        [
            product_index_row(
                "1",
                "camera",
                state_name="Tamil Nadu",
                city_name="Chennai",
                locality_name="Tambaram",
            )
        ]
    )
    value_index = query_filter_value_index(index)
    plan = {
        "semantic_query": "camera",
        "keyword_query": "camera",
        "target_ad_type": "offer",
        "filters": {
            "main_category": None,
            "subcategory": None,
            "state": None,
            "city": None,
            "locality": "Tambaram",
            "rental_duration": None,
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "fallback_reason": None,
    }

    enriched = enrich_query_plan("camera in Tambaram", plan, value_index)

    assert enriched["filters"]["locality"] == "Tambaram"
    assert enriched["filters"]["city"] == "Chennai"
    assert enriched["filters"]["state"] == "Tamil Nadu"
    index.close()


def test_query_fallback_recovers_misspelled_city(tmp_path):
    index = PersistentBM25Index(tmp_path / "fuzzy-location.sqlite3")
    index.upsert(
        [
            product_index_row(
                "1",
                "wedding photographer",
                state_name="Tamil Nadu",
                city_name="Coimbatore",
                locality_name="RS Puram",
            ),
            product_index_row(
                "2",
                "camera",
                state_name="Tamil Nadu",
                city_name="Chennai",
                locality_name="Tambaram",
            ),
        ]
    )
    value_index = query_filter_value_index(index)
    plan = {
        "semantic_query": "wedding photographer",
        "keyword_query": "wedding photographer",
        "target_ad_type": "offer",
        "filters": {
            "main_category": None,
            "subcategory": None,
            "state": None,
            "city": None,
            "locality": None,
            "rental_duration": None,
            "min_rental_fee": None,
            "max_rental_fee": None,
        },
        "inferred_categories": {
            "main_category": None,
            "subcategory": None,
        },
        "fallback_reason": "query provider unavailable",
    }

    enriched = enrich_query_plan(
        "need a wedding photographer in Coimbtore for a day",
        plan,
        value_index,
    )

    assert enriched["filters"]["city"] == "Coimbatore"
    assert enriched["filters"]["state"] == "Tamil Nadu"
    assert enriched["filters"]["rental_duration"] == "Per Day"
    index.close()


def test_bge_rerank_adapter_orders_by_cross_encoder_score():
    candidates = [
        {"id": "1", "text": "first passage", "metadata": {"id": 1}},
        {"id": "2", "text": "best passage", "metadata": {"id": 2}},
    ]

    results = rerank("query", candidates, FakeReranker(), top_k=2)

    assert [result["id"] for result in results] == ["2", "1"]
    assert [result["score"] for result in results] == [0.9, 0.1]


def test_hybrid_merge_uses_rrf_and_soft_category_boost():
    vector_results = [
        {
            "id": "vector-only",
            "text": "portable recorder",
            "metadata": {"subcategory_name": "Camera"},
            "score": 0.1,
            "source": "vector",
        },
        {
            "id": "both",
            "text": "camera",
            "metadata": {"subcategory_name": "Camera"},
            "score": 0.2,
            "source": "vector",
        },
    ]
    bm25_results = [
        {
            "id": "both",
            "text": "camera",
            "metadata": {"subcategory_name": "Camera"},
            "score": 2.0,
            "source": "bm25",
        },
        {
            "id": "bm25-only",
            "text": "recording equipment",
            "metadata": {"subcategory_name": "Other"},
            "score": 1.0,
            "source": "bm25",
        },
    ]

    merged = merge_results(
        vector_results,
        bm25_results,
        {"subcategory": "Camera"},
        rrf_constant=60,
        soft_category_boost=0.005,
    )

    assert [item["id"] for item in merged] == [
        "both",
        "vector-only",
        "bm25-only",
    ]
    assert merged[0]["source"] == "vector+bm25"


def test_rerank_diversifies_first_results_then_keeps_lower_ranked_duplicates():
    class OrderedReranker:
        def compute_score(self, _pairs, **_kwargs):
            return [0.9, 0.8, 0.7]

    candidates = [
        {
            "id": "1",
            "text": "first duplicate",
            "metadata": {"title": "Polaris Quad Bike"},
        },
        {
            "id": "2",
            "text": "second duplicate",
            "metadata": {"title": " polaris   quad bike "},
        },
        {
            "id": "3",
            "text": "different listing",
            "metadata": {"title": "RZR Quad Bike"},
        },
    ]

    results = rerank("quad bike", candidates, OrderedReranker(), top_k=3)

    assert [result["id"] for result in results] == ["1", "3", "2"]


def test_related_tail_uses_any_available_filter_and_offer_type(tmp_path):
    index = PersistentBM25Index(tmp_path / "related-tail.sqlite3")
    index.upsert(
        [
            product_index_row(
                "primary-bike",
                "primary bike",
                city_name="Chennai",
                rental_fee=900,
            ),
            product_index_row(
                "related-offer",
                "related offer",
                city_name="Chennai",
                rental_fee=800,
            ),
            product_index_row(
                "related-wanted",
                "related wanted",
                city_name="Chennai",
                rental_fee=700,
            ),
            product_index_row(
                "other-city",
                "other city",
                city_name="Madurai",
                rental_fee=600,
            ),
        ]
    )
    product_types = {
        "primary-bike": "1",
        "related-offer": "1",
        "related-wanted": "2",
        "other-city": "1",
    }

    product_ids = related_tail_product_ids(
        index,
        {"categorical": {"city_name": "Chennai"}},
        {"main_category": None, "subcategory": None},
        "offer",
        limit=10,
        exclude_product_ids={"primary-bike"},
        type_fetcher=lambda ids: {
            str(product_id): product_types[str(product_id)]
            for product_id in ids
        },
    )

    assert product_ids == ["related-offer"]
    index.close()


def test_related_tail_combines_inferred_category_with_partial_filters(tmp_path):
    index = PersistentBM25Index(tmp_path / "inferred-related-tail.sqlite3")
    index.upsert(
        [
            product_index_row(
                "camera-chennai",
                "camera",
                subcategory_name="Camera",
                city_name="Chennai",
            ),
            product_index_row(
                "bike-chennai",
                "bike",
                subcategory_name="Bike",
                city_name="Chennai",
            ),
        ]
    )

    product_ids = related_tail_product_ids(
        index,
        {"categorical": {"city_name": "Chennai"}},
        {"main_category": None, "subcategory": "Camera"},
        "offer",
        limit=10,
        type_fetcher=lambda ids: {
            str(product_id): "1"
            for product_id in ids
        },
    )

    assert product_ids == ["camera-chennai"]
    index.close()


def test_related_tail_requires_at_least_one_resolved_or_inferred_filter(tmp_path):
    index = PersistentBM25Index(tmp_path / "unfiltered-tail.sqlite3")
    index.upsert([product_index_row("bike", "bike")])

    product_ids = related_tail_product_ids(
        index,
        {"categorical": {}},
        {"main_category": None, "subcategory": None},
        "offer",
        limit=10,
        type_fetcher=lambda _ids: {"bike": "1"},
    )

    assert product_ids == []
    index.close()


def test_extract_product_ids_uses_rank_order_and_deduplicates():
    reranked = [
        {
            "metadata": {
                "source_type": "mysql",
                "source_table": MYSQL_TABLE,
                MYSQL_SEARCH_ID_COLUMN: 20,
            }
        },
        {
            "metadata": {
                "source_type": "mysql",
                "source_table": MYSQL_TABLE,
                MYSQL_SEARCH_ID_COLUMN: 10,
            }
        },
        {
            "metadata": {
                "source_type": "mysql",
                "source_table": MYSQL_TABLE,
                MYSQL_SEARCH_ID_COLUMN: 20,
            }
        },
    ]

    assert extract_product_ids(reranked) == [20, 10]


def test_extract_product_ids_skips_non_product_sources():
    reranked = [
        {"metadata": {"source_type": "csv", MYSQL_SEARCH_ID_COLUMN: 1}},
        {
            "metadata": {
                "source_type": "mysql",
                "source_table": "another_table",
                MYSQL_SEARCH_ID_COLUMN: 2,
            }
        },
        {
            "metadata": {
                "source_type": "mysql",
                "source_table": MYSQL_TABLE,
                "primary_key_column": MYSQL_SEARCH_ID_COLUMN,
                "primary_key_value": 3,
            }
        },
    ]

    assert extract_product_ids(reranked) == [3]


def test_filter_candidates_by_ad_type_excludes_wanted_ads():
    candidates = [
        {
            "id": "doc-1",
            "metadata": {
                "source_type": "mysql",
                "source_table": MYSQL_TABLE,
                MYSQL_SEARCH_ID_COLUMN: 10,
            },
        },
        {
            "id": "doc-2",
            "metadata": {
                "source_type": "mysql",
                "source_table": MYSQL_TABLE,
                MYSQL_SEARCH_ID_COLUMN: 20,
            },
        },
    ]
    connection = FakeConnection(
        [
            {MYSQL_RESULT_ID_COLUMN: "10", "type": "1"},
            {MYSQL_RESULT_ID_COLUMN: "20", "type": "2"},
        ]
    )

    offers = filter_candidates_by_ad_type(
        candidates,
        "offer",
        connection=connection,
    )

    assert [candidate["id"] for candidate in offers] == ["doc-1"]


def test_fetch_products_by_ids_preserves_reranker_order():
    connection = FakeConnection(
        [
            {MYSQL_RESULT_ID_COLUMN: "10", "title": "First"},
            {MYSQL_RESULT_ID_COLUMN: "20", "title": "Second"},
        ]
    )

    products = fetch_products_by_ids([20, 10, 20], connection=connection)

    assert [row[MYSQL_RESULT_ID_COLUMN] for row in products] == ["20", "10"]
    assert (
        connection.fake_cursor.query
        == f"SELECT * FROM `{MYSQL_RESULT_TABLE}` "
        f"WHERE `{MYSQL_RESULT_ID_COLUMN}` IN (%s, %s)"
    )
    assert connection.fake_cursor.params == [20, 10]


def test_fetch_products_by_ids_returns_empty_without_query():
    assert fetch_products_by_ids([], connection=FakeConnection([])) == []
