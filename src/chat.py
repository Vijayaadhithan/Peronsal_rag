import json
import time

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
from reranker import BGEReranker, load_reranker, rerank
from retrieval import (
    bm25_search,
    extract_product_ids,
    filter_candidates_by_ad_type,
    load_collection,
    merge_results,
    metadata_matches_filters,
    vector_search,
)
from settings import (
    APP_NAME,
    BM25_TOP_K,
    MYSQL_DATABASE,
    MYSQL_RESULT_ID_COLUMN,
    MYSQL_RESULT_TABLE,
    MYSQL_SEARCH_ID_COLUMN,
    MYSQL_TABLE,
    QUERY_EXTRACT_MODEL,
    RERANK_MODEL,
    RERANK_TOP_K,
    VECTOR_CANDIDATE_K,
    VECTOR_TOP_K,
)


def main():
    print("Opening Chroma collection...", flush=True)
    collection = load_collection()
    source_name = mysql_source_name()
    bm25_index = PersistentBM25Index()
    bm25_count = bm25_index.count()
    if not bm25_count:
        print(
            "No persistent BM25 product index found. Run: "
            "python src/ingest.py --mysql-bm25-only"
        )
        bm25_index.close()
        return

    print(f"Opened BM25 index with {bm25_count} products.", flush=True)
    filter_value_index = query_filter_value_index(bm25_index)
    query_filter_catalog = build_query_filter_catalog(filter_value_index)
    ranker = None
    print(f"\n{APP_NAME} semantic product search ready. Type 'exit' to quit.\n")

    try:
        while True:
            question = input("Ask: ").strip()
            if question.lower() in ["exit", "quit"]:
                break
            if not question:
                continue

            extraction_started = time.perf_counter()
            print(
                f"Extracting search intent with {QUERY_EXTRACT_MODEL}...",
                end="",
                flush=True,
            )
            query_plan = extract_query_plan(question, query_filter_catalog)
            query_plan = enrich_query_plan(
                question,
                query_plan,
                filter_value_index,
            )
            extraction_seconds = time.perf_counter() - extraction_started
            print(f" done ({extraction_seconds:.2f}s).", flush=True)
            if query_plan["fallback_reason"]:
                print(
                    "Query extraction failed; using the original query for vector and "
                    f"BM25 search. Reason: {query_plan['fallback_reason']}"
                )

            resolved_filters, unresolved_filters = resolve_query_filters(
                query_plan["filters"],
                filter_value_index,
            )
            if unresolved_filters:
                print(
                    "Ignoring filters that do not exactly match indexed values: "
                    f"{json.dumps(unresolved_filters, ensure_ascii=False)}"
                )

            visible_plan = {
                key: value
                for key, value in query_plan.items()
                if key != "fallback_reason"
            }
            print(
                "Query plan: "
                f"{json.dumps(visible_plan, ensure_ascii=False)}"
            )
            print("Searching...", end="", flush=True)
            vector_started = time.perf_counter()
            vector_results = vector_search(
                query_plan["semantic_query"],
                collection,
                VECTOR_TOP_K,
                candidate_k=VECTOR_CANDIDATE_K,
                source_name=source_name,
                resolved_filters=resolved_filters,
            )
            vector_seconds = time.perf_counter() - vector_started

            bm25_started = time.perf_counter()
            bm25_results = bm25_search(
                query_plan["keyword_query"],
                bm25_index,
                collection,
                resolved_filters,
                BM25_TOP_K,
            )
            bm25_seconds = time.perf_counter() - bm25_started

            merged = merge_results(vector_results, bm25_results)
            merged = filter_candidates_by_ad_type(
                merged,
                query_plan["target_ad_type"],
            )
            if not merged:
                target_label = (
                    "wanted ads"
                    if query_plan["target_ad_type"] == "wanted"
                    else "offer ads"
                )
                print(
                    f" done.\n\nNo matching {target_label} found after applying "
                    "the requested filters.\n"
                )
                print("-" * 80 + "\n")
                continue

            if ranker is None:
                print(f" loading {RERANK_MODEL}...", end="", flush=True)
                load_started = time.perf_counter()
                ranker = load_reranker()
                print(
                    f" loaded ({time.perf_counter() - load_started:.2f}s)...",
                    end="",
                    flush=True,
                )
            rerank_started = time.perf_counter()
            reranked = rerank(
                query_plan["semantic_query"],
                merged,
                ranker,
                RERANK_TOP_K,
            )
            rerank_seconds = time.perf_counter() - rerank_started
            print(
                " done "
                f"(vector {vector_seconds:.2f}s, BM25 {bm25_seconds:.3f}s, "
                f"rerank {rerank_seconds:.2f}s).",
                flush=True,
            )

            product_ids = extract_product_ids(reranked)
            products = fetch_products_by_ids(product_ids)
            print(
                f"\nProducts from {MYSQL_DATABASE}.{MYSQL_RESULT_TABLE} "
                f"({len(products)} rows):\n"
            )
            print(json.dumps(products, ensure_ascii=False, indent=2, default=str))
            print("\n" + "-" * 80 + "\n")
    finally:
        bm25_index.close()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
