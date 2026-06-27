import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bm25_index import PersistentBM25Index
from chat import (
    MYSQL_RESULT_ID_COLUMN,
    MYSQL_RESULT_TABLE,
    MYSQL_SEARCH_ID_COLUMN,
    MYSQL_TABLE,
    enrich_query_plan,
    extract_price_constraints,
    extract_product_ids,
    fetch_products_by_ids,
    metadata_matches_filters,
    parse_query_plan,
    query_filter_value_index,
    rerank,
    resolve_query_filters,
)


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


def test_extract_price_constraints_handles_range_and_minimum():
    assert extract_price_constraints("bike between Rs 1000 and Rs 2500") == (
        1000,
        2500,
    )
    assert extract_price_constraints("room above ₹750") == (750, None)


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
    index.close()


def test_bge_rerank_adapter_orders_by_cross_encoder_score():
    candidates = [
        {"id": "1", "text": "first passage", "metadata": {"id": 1}},
        {"id": "2", "text": "best passage", "metadata": {"id": 2}},
    ]

    results = rerank("query", candidates, FakeReranker(), top_k=2)

    assert [result["id"] for result in results] == ["2", "1"]
    assert [result["score"] for result in results] == [0.9, 0.1]


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
