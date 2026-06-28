import hashlib
import logging
import time
import uuid
from collections import OrderedDict
from copy import deepcopy

from bm25_index import PersistentBM25Index
from gemini_client import last_gemini_metrics
from mysql_store import fetch_products_by_ids, mysql_source_name
from ollama_client import last_ollama_embedding_metrics
from query_planner import (
    build_query_filter_catalog,
    deterministic_filter_query_plan,
    enrich_query_plan,
    extract_query_plan,
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
    related_tail_product_ids,
    vector_search,
)
from settings import (
    BM25_TOP_K,
    EMBED_MODEL,
    HYBRID_CANDIDATE_K,
    MYSQL_RESULT_ID_COLUMN,
    PRIMARY_RANKED_K,
    QUERY_DETERMINISTIC_FAST_PATH,
    QUERY_EXTRACT_MODELS,
    QUERY_PLAN_CACHE_SIZE,
    QUERY_PLAN_CACHE_TTL_SECONDS,
    RELATED_TAIL_ENABLED,
    RESULT_CACHE_ENABLED,
    RESULT_CACHE_TTL_SECONDS,
    RERANK_CANDIDATE_K,
    RETRIEVAL_OVERFETCH_FACTOR,
    RERANK_MODEL,
    RERANK_TOP_K,
    VECTOR_CANDIDATE_K,
    VECTOR_TOP_K,
)

LOGGER = logging.getLogger("uvicorn.error")
RESULT_CACHE_SCHEMA_VERSION = "v1"


def active_filter_names(filters: dict) -> list[str]:
    names = []
    for key, value in filters.items():
        if isinstance(value, dict):
            names.extend(
                f"{key}.{child_key}"
                for child_key, child_value in value.items()
                if child_value not in (None, "", [], {})
            )
        elif value not in (None, "", [], {}):
            names.append(key)
    return names


class ProductSearchEngine:
    def __init__(
        self,
        collection=None,
        bm25_index=None,
        query_provider=None,
        embedding_provider=None,
        ranker=None,
        shared_plan_cache=None,
    ):
        self.collection = collection or load_collection()
        self._owns_bm25_index = bm25_index is None
        self.bm25_index = bm25_index or PersistentBM25Index()
        self.query_provider = query_provider
        self.embedding_provider = embedding_provider
        self.ranker = ranker
        self.shared_plan_cache = shared_plan_cache
        self.source_name = mysql_source_name()
        self.filter_value_index = query_filter_value_index(self.bm25_index)
        self.filter_catalog = build_query_filter_catalog(self.filter_value_index)
        self._query_plan_cache: OrderedDict[str, tuple[float, dict]] = (
            OrderedDict()
        )

    def close(self) -> None:
        if self._owns_bm25_index:
            self.bm25_index.close()

    @staticmethod
    def _query_cache_key(query: str) -> str:
        return " ".join(query.casefold().split())

    def set_shared_plan_cache(self, cache) -> None:
        self.shared_plan_cache = cache

    def plan_cache_health(self) -> dict:
        if self.shared_plan_cache is None:
            return {
                "redis_enabled": False,
                "redis_connected": False,
                "query_plan_cache_backend": "memory",
                "result_cache_enabled": False,
                "result_cache_ttl_seconds": RESULT_CACHE_TTL_SECONDS,
            }
        return {
            "redis_enabled": True,
            "redis_connected": bool(
                getattr(self.shared_plan_cache, "connected", False)
            ),
            "query_plan_cache_backend": (
                "redis+memory"
                if getattr(self.shared_plan_cache, "connected", False)
                else "memory_fallback"
            ),
            "result_cache_enabled": RESULT_CACHE_ENABLED,
            "result_cache_ttl_seconds": RESULT_CACHE_TTL_SECONDS,
        }

    def _result_cache_key(self, query: str, limit: int | None) -> str:
        version_parts = (
            RESULT_CACHE_SCHEMA_VERSION,
            str(self.bm25_index.revision()),
            str(self.bm25_index.count()),
            str(limit),
            RERANK_MODEL,
            str(RERANK_CANDIDATE_K),
            str(PRIMARY_RANKED_K),
            self._query_cache_key(query),
        )
        return hashlib.sha256("\0".join(version_parts).encode()).hexdigest()

    def _cached_search_result(
        self,
        query: str,
        limit: int | None,
        trace_id: str,
    ) -> dict | None:
        if not RESULT_CACHE_ENABLED or self.shared_plan_cache is None:
            return None
        started = time.perf_counter()
        cache_key = self._result_cache_key(query, limit)
        cached = self.shared_plan_cache.get_json(
            "search_result",
            cache_key,
        )
        if cached is None:
            LOGGER.info(
                "[search:%s] step=result_cache status=miss duration_ms=%.0f",
                trace_id,
                (time.perf_counter() - started) * 1000,
            )
            return None
        required = {
            "query_plan",
            "resolved_filters",
            "unresolved_filters",
            "product_ids",
            "primary_product_ids",
            "related_product_ids",
        }
        if not required.issubset(cached) or not isinstance(
            cached["product_ids"],
            list,
        ):
            LOGGER.warning(
                "[search:%s] step=result_cache status=invalid",
                trace_id,
            )
            return None

        mysql_started = time.perf_counter()
        products = fetch_products_by_ids(cached["product_ids"])
        primary_identities = {
            str(product_id)
            for product_id in cached["primary_product_ids"]
        }
        deterministic = (
            cached["query_plan"].get("execution_path")
            == "deterministic_filter"
        )
        products = [
            {
                **product,
                "result_tier": (
                    "filtered"
                    if deterministic
                    else (
                        "ranked"
                        if str(product.get(MYSQL_RESULT_ID_COLUMN))
                        in primary_identities
                        else "related"
                    )
                ),
            }
            for product in products
        ]
        elapsed = time.perf_counter() - started
        LOGGER.info(
            "[search:%s] step=result_cache status=hit ids=%d rows=%d "
            "mysql_ms=%.0f duration_ms=%.0f",
            trace_id,
            len(cached["product_ids"]),
            len(products),
            (time.perf_counter() - mysql_started) * 1000,
            elapsed * 1000,
        )
        return {
            "query_plan": cached["query_plan"],
            "resolved_filters": cached["resolved_filters"],
            "unresolved_filters": cached["unresolved_filters"],
            "query_model_metrics": {},
            "seconds": 0.0,
            "plan_cache_hit": True,
            "vector_results": [],
            "bm25_results": [],
            "candidates": [],
            "vector_seconds": 0.0,
            "bm25_seconds": 0.0,
            "embedding_model_metrics": {},
            "reranked": [],
            "reranker_load_seconds": 0.0,
            "reranker_seconds": 0.0,
            "related_tail_seconds": 0.0,
            "primary_product_ids": cached["primary_product_ids"],
            "related_product_ids": cached["related_product_ids"],
            "product_ids": cached["product_ids"],
            "products": products,
            "result_cache_hit": True,
            "result_cache_seconds": elapsed,
        }

    def _cache_search_result(
        self,
        query: str,
        limit: int | None,
        result: dict,
    ) -> None:
        if (
            not RESULT_CACHE_ENABLED
            or self.shared_plan_cache is None
            or result["query_plan"].get("fallback_reason")
        ):
            return
        payload = {
            "query_plan": result["query_plan"],
            "resolved_filters": result["resolved_filters"],
            "unresolved_filters": result["unresolved_filters"],
            "product_ids": [
                str(product_id)
                for product_id in result.get("product_ids", [])
            ],
            "primary_product_ids": [
                str(product_id)
                for product_id in result.get("primary_product_ids", [])
            ],
            "related_product_ids": [
                str(product_id)
                for product_id in result.get("related_product_ids", [])
            ],
        }
        self.shared_plan_cache.set_json(
            "search_result",
            self._result_cache_key(query, limit),
            payload,
            RESULT_CACHE_TTL_SECONDS,
        )

    def _cached_plan(self, query: str) -> dict | None:
        if QUERY_PLAN_CACHE_SIZE <= 0 or QUERY_PLAN_CACHE_TTL_SECONDS <= 0:
            return None
        key = self._query_cache_key(query)
        cached = self._query_plan_cache.get(key)
        if cached is not None:
            expires_at, result = cached
            if expires_at > time.monotonic():
                self._query_plan_cache.move_to_end(key)
                return deepcopy(result)
            del self._query_plan_cache[key]
        if self.shared_plan_cache is None:
            return None
        shared_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
        result = self.shared_plan_cache.get_json("query_plan", shared_key)
        if result is None:
            return None
        self._cache_memory_plan(key, result)
        return deepcopy(result)

    def _cache_memory_plan(self, key: str, result: dict) -> None:
        self._query_plan_cache[key] = (
            time.monotonic() + QUERY_PLAN_CACHE_TTL_SECONDS,
            deepcopy(result),
        )
        self._query_plan_cache.move_to_end(key)
        while len(self._query_plan_cache) > QUERY_PLAN_CACHE_SIZE:
            self._query_plan_cache.popitem(last=False)

    def _cache_plan(self, query: str, result: dict) -> None:
        if QUERY_PLAN_CACHE_SIZE <= 0 or QUERY_PLAN_CACHE_TTL_SECONDS <= 0:
            return
        key = self._query_cache_key(query)
        self._cache_memory_plan(key, result)
        if self.shared_plan_cache is not None:
            shared_key = hashlib.sha256(key.encode("utf-8")).hexdigest()
            self.shared_plan_cache.set_json(
                "query_plan",
                shared_key,
                result,
                QUERY_PLAN_CACHE_TTL_SECONDS,
            )

    def plan(self, query: str, trace_id: str = "-") -> dict:
        started = time.perf_counter()
        LOGGER.info(
            "[search:%s] step=plan status=start query_chars=%d models=%s",
            trace_id,
            len(query),
            " -> ".join(QUERY_EXTRACT_MODELS),
        )
        query_plan = (
            deterministic_filter_query_plan(
                query,
                self.filter_value_index,
            )
            if QUERY_DETERMINISTIC_FAST_PATH
            else None
        )
        if query_plan is None:
            cached = self._cached_plan(query)
            if cached is not None:
                elapsed = time.perf_counter() - started
                cached.update(
                    {
                        "query_model_metrics": {},
                        "seconds": elapsed,
                        "plan_cache_hit": True,
                    }
                )
                LOGGER.info(
                    "[search:%s] step=plan status=cache_hit path=%s "
                    "duration_ms=%.0f",
                    trace_id,
                    cached["query_plan"].get("execution_path", "semantic"),
                    elapsed * 1000,
                )
                return cached
            query_plan = extract_query_plan(
                query,
                self.filter_catalog,
                query_provider=self.query_provider,
            )
            query_plan = enrich_query_plan(
                query,
                query_plan,
                self.filter_value_index,
            )
            query_plan["execution_path"] = "semantic"
        resolved, unresolved = resolve_query_filters(
            query_plan["filters"],
            self.filter_value_index,
        )
        query_metrics = (
            last_gemini_metrics()
            if self.query_provider is None
            and query_plan["execution_path"] == "semantic"
            else {}
        )
        elapsed = time.perf_counter() - started
        log_method = (
            LOGGER.warning
            if query_plan.get("fallback_reason")
            else LOGGER.info
        )
        model_label = query_metrics.get("model")
        attempted_label = ",".join(
            query_metrics.get("attempted_models", [])
        )
        if query_plan["execution_path"] == "deterministic_filter":
            model_label = "none"
            attempted_label = "none"
        elif not model_label:
            model_label = type(self.query_provider).__name__
            attempted_label = attempted_label or "custom"
        log_method(
            "[search:%s] step=plan status=%s path=%s model=%s attempted=%s "
            "filters=%s unresolved=%d reason=%s duration_ms=%.0f",
            trace_id,
            (
                "provider_fallback"
                if query_plan.get("fallback_reason")
                else "complete"
            ),
            query_plan["execution_path"],
            model_label,
            attempted_label,
            ",".join(active_filter_names(query_plan["filters"])) or "none",
            len(unresolved),
            query_plan.get("fallback_reason") or "none",
            elapsed * 1000,
        )
        result = {
            "query_plan": query_plan,
            "resolved_filters": resolved,
            "unresolved_filters": unresolved,
            "query_model_metrics": query_metrics,
            "seconds": elapsed,
            "plan_cache_hit": False,
        }
        if not query_plan.get("fallback_reason"):
            self._cache_plan(
                query,
                {
                    "query_plan": query_plan,
                    "resolved_filters": resolved,
                    "unresolved_filters": unresolved,
                },
            )
        return result

    def retrieve(
        self,
        query_plan: dict,
        resolved_filters: dict,
        candidate_limit: int | None = None,
        trace_id: str = "-",
    ) -> dict:
        extended_window = (
            candidate_limit is not None and candidate_limit > RERANK_TOP_K
        )
        requested = candidate_limit or RERANK_TOP_K
        retrieval_depth = (
            max(
                VECTOR_TOP_K,
                BM25_TOP_K,
                requested * RETRIEVAL_OVERFETCH_FACTOR,
            )
            if extended_window
            else None
        )
        vector_top_k = retrieval_depth or VECTOR_TOP_K
        bm25_top_k = retrieval_depth or BM25_TOP_K
        hybrid_top_k = max(HYBRID_CANDIDATE_K, requested)
        LOGGER.info(
            "[search:%s] step=retrieve status=start embedding_model=%s "
            "vector_k=%d bm25_k=%d hybrid_k=%d filters=%s",
            trace_id,
            EMBED_MODEL,
            vector_top_k,
            bm25_top_k,
            hybrid_top_k,
            ",".join(active_filter_names(resolved_filters)) or "none",
        )

        vector_started = time.perf_counter()
        vector_results = vector_search(
            query_plan["semantic_query"],
            self.collection,
            vector_top_k,
            candidate_k=max(VECTOR_CANDIDATE_K, vector_top_k),
            source_name=self.source_name,
            resolved_filters=resolved_filters,
            embedding_provider=self.embedding_provider,
        )
        vector_seconds = time.perf_counter() - vector_started

        bm25_started = time.perf_counter()
        bm25_results = bm25_search(
            query_plan["keyword_query"],
            self.bm25_index,
            self.collection,
            resolved_filters,
            bm25_top_k,
        )
        bm25_seconds = time.perf_counter() - bm25_started

        merged = merge_results(
            vector_results,
            bm25_results,
            query_plan.get("inferred_categories"),
        )
        # Apply ad intent before truncating the candidate window. Otherwise a
        # page can be short merely because unwanted ad types occupied the K
        # slots ahead of valid products.
        candidates = filter_candidates_by_ad_type(
            merged,
            query_plan["target_ad_type"],
        )[:hybrid_top_k]
        LOGGER.info(
            "[search:%s] step=retrieve status=complete vector=%d bm25=%d "
            "merged=%d candidates=%d vector_ms=%.0f bm25_ms=%.0f",
            trace_id,
            len(vector_results),
            len(bm25_results),
            len(merged),
            len(candidates),
            vector_seconds * 1000,
            bm25_seconds * 1000,
        )
        return {
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "candidates": candidates,
            "vector_seconds": vector_seconds,
            "bm25_seconds": bm25_seconds,
            "embedding_model_metrics": (
                last_ollama_embedding_metrics()
                if self.embedding_provider is None
                else {}
            ),
        }

    def ensure_reranker(self) -> float:
        if self.ranker is not None:
            return 0.0
        started = time.perf_counter()
        self.ranker = load_reranker()
        return time.perf_counter() - started

    def rank(
        self,
        query: str,
        candidates: list[dict],
        query_plan: dict | None = None,
        top_k: int | None = None,
        trace_id: str = "-",
    ) -> dict:
        load_seconds = self.ensure_reranker()
        ranking_query = query
        if query_plan is not None:
            context = []
            keyword_query = query_plan.get("keyword_query")
            if keyword_query and keyword_query.casefold() != query.casefold():
                context.append(f"Search concepts: {keyword_query}")
            inferred = query_plan.get("inferred_categories") or {}
            category_hints = [
                value
                for value in (
                    inferred.get("main_category"),
                    inferred.get("subcategory"),
                )
                if value
            ]
            if category_hints:
                context.append(
                    "Possible catalog categories: "
                    + ", ".join(dict.fromkeys(category_hints))
                )
            if context:
                ranking_query = query + "\n" + "\n".join(context)
        started = time.perf_counter()
        LOGGER.info(
            "[search:%s] step=rerank status=start model=%s candidates=%d top_k=%d",
            trace_id,
            RERANK_MODEL,
            len(candidates),
            RERANK_TOP_K if top_k is None else top_k,
        )
        results = rerank(
            ranking_query,
            candidates,
            self.ranker,
            RERANK_TOP_K if top_k is None else top_k,
            diversity_top_k=RERANK_TOP_K,
        )
        elapsed = time.perf_counter() - started
        LOGGER.info(
            "[search:%s] step=rerank status=complete results=%d "
            "load_ms=%.0f duration_ms=%.0f",
            trace_id,
            len(results),
            load_seconds * 1000,
            elapsed * 1000,
        )
        return {
            "results": results,
            "load_seconds": load_seconds,
            "seconds": elapsed,
        }

    def _filtered_search(
        self,
        planned: dict,
        limit: int | None,
        trace_id: str,
        search_started: float,
    ) -> dict:
        result_limit = limit if limit is not None else RERANK_TOP_K
        browse_started = time.perf_counter()
        product_ids = related_tail_product_ids(
            self.bm25_index,
            planned["resolved_filters"],
            planned["query_plan"].get("inferred_categories"),
            planned["query_plan"]["target_ad_type"],
            result_limit,
        )
        browse_seconds = time.perf_counter() - browse_started
        LOGGER.info(
            "[search:%s] step=fast_filter status=complete filters=%s "
            "products=%d duration_ms=%.0f",
            trace_id,
            ",".join(
                active_filter_names(planned["resolved_filters"])
            ) or "none",
            len(product_ids),
            browse_seconds * 1000,
        )
        products = [
            {**product, "result_tier": "filtered"}
            for product in fetch_products_by_ids(product_ids)
        ]
        LOGGER.info(
            "[search:%s] step=search status=complete path=deterministic_filter "
            "products=%d duration_ms=%.0f",
            trace_id,
            len(products),
            (time.perf_counter() - search_started) * 1000,
        )
        return {
            **planned,
            "vector_results": [],
            "bm25_results": [],
            "candidates": [],
            "vector_seconds": 0.0,
            "bm25_seconds": 0.0,
            "embedding_model_metrics": {},
            "reranked": [],
            "reranker_load_seconds": 0.0,
            "reranker_seconds": 0.0,
            "related_tail_seconds": browse_seconds,
            "primary_product_ids": [],
            "related_product_ids": product_ids,
            "product_ids": product_ids,
            "products": products,
        }

    def search(self, query: str, limit: int | None = None) -> dict:
        if limit is not None and limit <= 0:
            raise ValueError("Search limit must be greater than zero.")
        primary_limit = (
            min(PRIMARY_RANKED_K, limit)
            if limit is not None
            else RERANK_TOP_K
        )
        rerank_candidate_limit = (
            max(primary_limit, RERANK_CANDIDATE_K)
            if limit is not None
            else None
        )
        trace_id = uuid.uuid4().hex[:8]
        started = time.perf_counter()
        LOGGER.info(
            "[search:%s] step=search status=start query_chars=%d limit=%s",
            trace_id,
            len(query),
            limit if limit is not None else "default",
        )
        cached_result = self._cached_search_result(
            query,
            limit,
            trace_id,
        )
        if cached_result is not None:
            LOGGER.info(
                "[search:%s] step=search status=complete "
                "path=result_cache products=%d duration_ms=%.0f",
                trace_id,
                len(cached_result["products"]),
                (time.perf_counter() - started) * 1000,
            )
            return cached_result
        planned = self.plan(query, trace_id=trace_id)
        if (
            planned["query_plan"].get("execution_path")
            == "deterministic_filter"
        ):
            result = self._filtered_search(
                planned,
                limit,
                trace_id,
                started,
            )
            result["result_cache_hit"] = False
            result["result_cache_seconds"] = 0.0
            self._cache_search_result(query, limit, result)
            return result
        retrieved = self.retrieve(
            planned["query_plan"],
            planned["resolved_filters"],
            candidate_limit=rerank_candidate_limit,
            trace_id=trace_id,
        )
        candidates = retrieved["candidates"]
        if candidates:
            ranked = self.rank(
                query,
                candidates,
                planned["query_plan"],
                top_k=primary_limit,
                trace_id=trace_id,
            )
        else:
            LOGGER.info(
                "[search:%s] step=rerank status=skipped reason=no_candidates",
                trace_id,
            )
            ranked = {"results": [], "load_seconds": 0.0, "seconds": 0.0}
        primary_product_ids = extract_product_ids(ranked["results"])
        tail_limit = (
            max(limit - len(primary_product_ids), 0)
            if limit is not None
            else 0
        )
        tail_started = time.perf_counter()
        related_product_ids = []
        if RELATED_TAIL_ENABLED and tail_limit:
            related_product_ids = related_tail_product_ids(
                self.bm25_index,
                planned["resolved_filters"],
                planned["query_plan"].get("inferred_categories"),
                planned["query_plan"]["target_ad_type"],
                tail_limit,
                exclude_doc_ids={
                    result["id"]
                    for result in ranked["results"]
                },
                exclude_product_ids=set(primary_product_ids),
            )
        related_tail_seconds = time.perf_counter() - tail_started
        product_ids = list(
            dict.fromkeys((*primary_product_ids, *related_product_ids))
        )
        LOGGER.info(
            "[search:%s] step=related_tail status=complete filters=%s "
            "primary=%d related=%d duration_ms=%.0f",
            trace_id,
            ",".join(
                active_filter_names(planned["resolved_filters"])
            ) or "none",
            len(primary_product_ids),
            len(related_product_ids),
            related_tail_seconds * 1000,
        )
        LOGGER.info(
            "[search:%s] step=mysql_map status=start product_ids=%d",
            trace_id,
            len(product_ids),
        )
        products = fetch_products_by_ids(product_ids)
        primary_identities = {
            str(product_id)
            for product_id in primary_product_ids
        }
        products = [
            {
                **product,
                "result_tier": (
                    "ranked"
                    if str(product.get(MYSQL_RESULT_ID_COLUMN))
                    in primary_identities
                    else "related"
                ),
            }
            for product in products
        ]
        LOGGER.info(
            "[search:%s] step=mysql_map status=complete rows=%d",
            trace_id,
            len(products),
        )
        LOGGER.info(
            "[search:%s] step=search status=complete products=%d duration_ms=%.0f",
            trace_id,
            len(products),
            (time.perf_counter() - started) * 1000,
        )
        result = {
            **planned,
            **retrieved,
            "reranked": ranked["results"],
            "reranker_load_seconds": ranked["load_seconds"],
            "reranker_seconds": ranked["seconds"],
            "related_tail_seconds": related_tail_seconds,
            "primary_product_ids": primary_product_ids,
            "related_product_ids": related_product_ids,
            "product_ids": product_ids,
            "products": products,
            "result_cache_hit": False,
            "result_cache_seconds": 0.0,
        }
        self._cache_search_result(query, limit, result)
        return result
