import argparse
import json

from api import (
    PUBLIC_PRODUCT_FIELDS,
    product_is_visible,
    public_product,
)
from bm25_index import PersistentBM25Index
from mysql_store import (
    fetch_product_types_by_ids,
    fetch_products_by_ids,
    mysql_connection,
    mysql_source_name,
    quote_mysql_identifier,
    require_pymysql,
)
from query_planner import (
    build_query_filter_catalog,
    enrich_query_plan,
    extract_duration_filter,
    extract_price_constraints,
    extract_query_plan,
    infer_target_ad_type,
    normalize_filter_value,
    parse_query_plan,
    query_filter_value_index,
    resolve_query_filters,
)
from reranker import load_reranker, rerank
from retrieval import (
    bm25_search,
    extract_product_ids,
    filter_candidates_by_ad_type,
    load_collection,
    merge_results,
    metadata_matches_filters,
    vector_search,
)
from search_engine import ProductSearchEngine
from settings import (
    APP_NAME,
    MYSQL_DATABASE,
    MYSQL_RESULT_ID_COLUMN,
    MYSQL_RESULT_TABLE,
    MYSQL_SEARCH_ID_COLUMN,
    MYSQL_TABLE,
    QUERY_EXTRACT_MODEL,
    RERANK_MODEL,
)
from tenant_config import discover_tenant_profiles
from vector_store import get_tenant_vector_collection


def build_engine(company: str | None = None) -> ProductSearchEngine:
    if company is None:
        engine = ProductSearchEngine()
        engine.chat_public_fields = PUBLIC_PRODUCT_FIELDS
        engine.chat_field_mapping = {}
        return engine
    profiles = discover_tenant_profiles()
    try:
        profile = profiles[company]
    except KeyError as exc:
        available = ", ".join(sorted(profiles)) or "none"
        raise RuntimeError(
            f"Unknown company {company!r}; available: {available}"
        ) from exc
    engine = ProductSearchEngine(
        collection=get_tenant_vector_collection(profile),
        bm25_index=PersistentBM25Index(profile.storage.bm25_path),
        company_id=profile.company_id,
        mysql_config=profile.database,
        close_bm25_index=True,
    )
    engine.chat_public_fields = profile.payload.public_fields
    engine.chat_field_mapping = profile.payload.field_mapping
    return engine


def print_search_result(result: dict, engine: ProductSearchEngine) -> None:
    query_plan = result["query_plan"]
    visible_plan = {
        key: value
        for key, value in query_plan.items()
        if key != "fallback_reason"
    }
    print(
        "Query plan: "
        f"{json.dumps(visible_plan, ensure_ascii=False)}"
    )
    if result["unresolved_filters"]:
        print(
            "Unresolved filters: "
            f"{json.dumps(result['unresolved_filters'], ensure_ascii=False)}"
        )
    print(
        "Timings: "
        f"plan={result.get('seconds', 0):.3f}s "
        f"vector={result.get('vector_seconds', 0):.3f}s "
        f"bm25={result.get('bm25_seconds', 0):.3f}s "
        f"rerank={result.get('reranker_seconds', 0):.3f}s "
        f"provider={result.get('reranker_provider', 'none')}"
    )
    products = [
        public_product(
            product,
            fields=engine.chat_public_fields,
            field_mapping=engine.chat_field_mapping,
        )
        for product in result["products"]
        if product_is_visible(product)
    ]
    print(
        json.dumps(
            products,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run one search or open the interactive search shell."
    )
    parser.add_argument(
        "--company",
        help="tenant profile slug; omit to use the legacy local indexes",
    )
    parser.add_argument(
        "--query",
        help="run one query and exit instead of opening the interactive shell",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="maximum results for one-shot or interactive searches",
    )
    args = parser.parse_args()
    if args.limit <= 0:
        raise RuntimeError("--limit must be greater than zero.")

    backend_label = f"company {args.company}" if args.company else "local"
    print(f"Opening {backend_label} search indexes...", flush=True)
    engine = build_engine(args.company)
    bm25_count = engine.bm25_index.count()
    if not bm25_count:
        command = (
            f"python src/ingest.py --company {args.company} --mysql-bm25-only"
            if args.company
            else "python src/ingest.py --mysql-bm25-only"
        )
        print(
            "No persistent BM25 product index found. Run: "
            + command
        )
        engine.close()
        return

    print(f"Opened BM25 index with {bm25_count} products.", flush=True)
    print(f"\n{APP_NAME} semantic product search ready.", flush=True)

    try:
        if args.query:
            result = engine.search(args.query, limit=args.limit)
            print_search_result(result, engine)
            return
        print("Type 'exit' to quit.\n")
        while True:
            question = input("Ask: ").strip()
            if question.lower() in ["exit", "quit"]:
                break
            if not question:
                continue

            result = engine.search(question, limit=args.limit)
            print_search_result(result, engine)
            print("\n" + "-" * 80 + "\n")
    finally:
        engine.close()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
