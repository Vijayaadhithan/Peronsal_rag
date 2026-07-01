import base64
import binascii
import hashlib
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from bm25_index import PersistentBM25Index
from ollama_client import preload_ollama_embedding
from rate_limit import TenantRateLimiter
from redis_cache import create_redis_cache
from reranker import SharedReranker
from search_engine import ProductSearchEngine
from settings import (
    API_AUTH_ENABLED,
    API_CORS_ORIGINS,
    API_DEFAULT_PAGE_SIZE,
    API_MAX_PAGE_SIZE,
    API_MAX_RESULTS,
    API_MAX_SESSIONS,
    API_PRELOAD_EMBEDDING,
    API_PRELOAD_RERANKER,
    API_SESSION_TTL_SECONDS,
    API_RATE_LIMIT_ENABLED,
    API_TENANT_CONFIG_DIR,
    API_TENANT_ENGINE_CACHE_SIZE,
    API_TENANT_MAX_CONCURRENT_SEARCHES,
    APP_NAME,
    RERANK_MODEL,
    REDIS_ENABLED,
    REDIS_KEY_PREFIX,
    REDIS_URL,
    USAGE_DB_PATH,
    USAGE_TRACKING_ENABLED,
)
from tenant_config import (
    TenantProfile,
    TenantRegistry,
    load_tenant_registry,
)
from vector_store import get_tenant_vector_collection
from usage_store import MonthlyUsageStore

LOGGER = logging.getLogger("uvicorn.error")

PUBLIC_PRODUCT_FIELDS = (
    "result_tier",
    "id",
    "type",
    "category_type",
    "parent_id",
    "category_id",
    "title",
    "slug",
    "description",
    "rental_duration",
    "rental_fee",
    "is_rent_negotiable",
    "city_id",
    "locality_id",
    "custom_locality",
    "photos",
    "total_favorite",
    "total_like",
    "users_rating_count",
    "rating_avg",
    "created_at",
    "updated_at",
)


class InvalidCursorError(ValueError):
    pass


class ExpiredCursorError(ValueError):
    pass


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str | None = Field(default=None, max_length=1000)
    cursor: str | None = Field(default=None, max_length=512)
    page_size: int = Field(
        default=API_DEFAULT_PAGE_SIZE,
        ge=1,
        le=API_MAX_PAGE_SIZE,
    )

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        if not value:
            raise ValueError("query must not be blank")
        return value

    @model_validator(mode="after")
    def require_query_or_cursor(self):
        if (self.query is None) == (self.cursor is None):
            raise ValueError("provide exactly one of query or cursor")
        return self


class PaginationResponse(BaseModel):
    page_size: int
    returned: int
    offset: int
    total_results: int
    has_more: bool
    next_cursor: str | None


class SearchResponse(BaseModel):
    company_id: str | None = None
    search_id: str
    query: str
    cached: bool
    items: list[dict[str, Any]]
    interpreted_query: dict[str, Any]
    applied_filters: dict[str, Any]
    unresolved_filters: dict[str, Any]
    timings_ms: dict[str, float]
    usage: dict[str, Any] = Field(default_factory=dict)
    pagination: PaginationResponse


class HealthResponse(BaseModel):
    status: str
    app: str
    indexed_products: int
    max_result_window: int
    session_ttl_seconds: int
    reranker_model: str
    reranker_loaded: bool
    reranker_load_ms: float
    embedding_warmup: dict[str, Any]
    redis_enabled: bool
    redis_connected: bool
    query_plan_cache_backend: str
    result_cache_enabled: bool
    result_cache_ttl_seconds: int
    company_id: str | None = None


@dataclass
class SearchSession:
    search_id: str
    query: str
    items: list[dict[str, Any]]
    interpreted_query: dict[str, Any]
    applied_filters: dict[str, Any]
    unresolved_filters: dict[str, Any]
    timings_ms: dict[str, float]
    usage: dict[str, Any]
    expires_at: float
    company_id: str | None = None


class SearchSessionStore:
    def __init__(
        self,
        ttl_seconds: int = API_SESSION_TTL_SECONDS,
        max_sessions: int = API_MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ):
        if ttl_seconds <= 0:
            raise ValueError("Session TTL must be greater than zero.")
        if max_sessions <= 0:
            raise ValueError("Maximum sessions must be greater than zero.")
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.clock = clock
        self._sessions: OrderedDict[str, SearchSession] = OrderedDict()
        self._lock = threading.Lock()

    def create(self, **values) -> SearchSession:
        with self._lock:
            now = self.clock()
            self._remove_expired(now)
            search_id = str(uuid.uuid4())
            session = SearchSession(
                search_id=search_id,
                expires_at=now + self.ttl_seconds,
                **values,
            )
            self._sessions[search_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return session

    def get(self, search_id: str) -> SearchSession:
        with self._lock:
            session = self._sessions.get(search_id)
            if session is None:
                raise InvalidCursorError("The cursor is invalid.")
            if session.expires_at <= self.clock():
                del self._sessions[search_id]
                raise ExpiredCursorError(
                    "The cursor has expired. Start a new search with query."
                )
            self._sessions.move_to_end(search_id)
            return session

    def _remove_expired(self, now: float) -> None:
        expired = [
            search_id
            for search_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for search_id in expired:
            del self._sessions[search_id]


def encode_cursor(search_id: str, offset: int) -> str:
    raw = f"v1:{search_id}:{offset}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[str, int]:
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.urlsafe_b64decode(cursor + padding).decode()
        version, search_id, raw_offset = decoded.split(":")
        parsed_id = str(uuid.UUID(search_id))
        offset = int(raw_offset)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise InvalidCursorError("The cursor is invalid.") from exc
    if version != "v1" or offset < 0:
        raise InvalidCursorError("The cursor is invalid.")
    return parsed_id, offset


class ProductSearchService:
    def __init__(
        self,
        engine: ProductSearchEngine,
        sessions: SearchSessionStore | None = None,
        max_results: int = API_MAX_RESULTS,
        company_id: str | None = None,
        public_fields: tuple[str, ...] = PUBLIC_PRODUCT_FIELDS,
        field_mapping: dict[str, str] | None = None,
        usage_store: MonthlyUsageStore | None = None,
        max_concurrent_searches: int = API_TENANT_MAX_CONCURRENT_SEARCHES,
    ):
        if max_results <= 0:
            raise ValueError("Maximum results must be greater than zero.")
        if max_concurrent_searches <= 0:
            raise ValueError("Maximum concurrent searches must be greater than zero.")
        self.engine = engine
        self.sessions = sessions or SearchSessionStore()
        self.max_results = max_results
        self.company_id = company_id
        self.public_fields = tuple(public_fields)
        self.field_mapping = dict(field_mapping or {})
        self.usage_store = usage_store
        self._engine_lock = threading.Lock()
        self._search_slots = threading.BoundedSemaphore(
            max_concurrent_searches
        )
        self.reranker_load_ms = 0.0
        self.embedding_warmup: dict[str, Any] = {}

    def warmup(self) -> float:
        with self._engine_lock:
            load_seconds = self.engine.ensure_reranker()
        self.reranker_load_ms = load_seconds * 1000
        return self.reranker_load_ms

    def health(self) -> HealthResponse:
        with self._engine_lock:
            indexed_products = self.engine.bm25_index.count()
            cache_health = (
                self.engine.plan_cache_health()
                if hasattr(self.engine, "plan_cache_health")
                else {
                    "redis_enabled": False,
                    "redis_connected": False,
                    "query_plan_cache_backend": "memory",
                    "result_cache_enabled": False,
                    "result_cache_ttl_seconds": 0,
                }
            )
        return HealthResponse(
            status="ok",
            app=APP_NAME,
            indexed_products=indexed_products,
            max_result_window=self.max_results,
            session_ttl_seconds=self.sessions.ttl_seconds,
            reranker_model=getattr(
                self.engine.ranker,
                "model_label",
                RERANK_MODEL,
            ),
            reranker_loaded=self.engine.ranker is not None,
            reranker_load_ms=self.reranker_load_ms,
            embedding_warmup=self.embedding_warmup,
            company_id=self.company_id,
            **cache_health,
        )

    def search(self, request: SearchRequest) -> SearchResponse:
        if request.query is not None:
            session = self._start_search(request.query)
            offset = 0
            cached = bool(
                session.interpreted_query.get("result_cache_hit")
            )
        else:
            search_id, offset = decode_cursor(request.cursor or "")
            session = self.sessions.get(search_id)
            cached = True
        return self._page(session, offset, request.page_size, cached)

    def _start_search(self, query: str) -> SearchSession:
        started = time.perf_counter()
        with self._search_slots:
            result = self.engine.search(query, limit=self.max_results)
        total_ms = (time.perf_counter() - started) * 1000
        query_plan = result["query_plan"]
        usage = self._record_usage(result)
        interpreted_query = {
            key: query_plan.get(key)
            for key in (
                "semantic_query",
                "keyword_query",
                "target_ad_type",
                "sort_order",
            )
        }
        interpreted_query.update(
            {
                "execution_path": query_plan.get(
                    "execution_path",
                    "semantic",
                ),
                "plan_cache_hit": bool(result.get("plan_cache_hit")),
                "result_cache_hit": bool(result.get("result_cache_hit")),
                "query_corrections": query_plan.get(
                    "query_corrections",
                    [],
                ),
                "reranker_provider": result.get(
                    "reranker_provider",
                    "none",
                ),
            }
        )
        timings_ms = {
            "planning": result.get("seconds", 0.0) * 1000,
            "vector_search": result.get("vector_seconds", 0.0) * 1000,
            "bm25_search": result.get("bm25_seconds", 0.0) * 1000,
            "related_tail": result.get("related_tail_seconds", 0.0) * 1000,
            "reranker_load": result.get("reranker_load_seconds", 0.0) * 1000,
            "reranking": result.get("reranker_seconds", 0.0) * 1000,
            "result_cache": result.get(
                "result_cache_seconds",
                0.0,
            )
            * 1000,
            "total": total_ms,
        }
        query_metrics = result.get("query_model_metrics") or {}
        embedding_metrics = result.get("embedding_model_metrics") or {}
        timings_ms.update(
            {
                "query_model_total": query_metrics.get("total_ms", 0.0),
                "query_model_load": query_metrics.get("load_ms", 0.0),
                "embedding_model_total": embedding_metrics.get("total_ms", 0.0),
                "embedding_model_load": embedding_metrics.get("load_ms", 0.0),
            }
        )
        return self.sessions.create(
            query=query,
            items=[
                public_product(
                    product,
                    fields=self.public_fields,
                    field_mapping=self.field_mapping,
                )
                for product in result["products"]
                if product_is_visible(product)
            ],
            interpreted_query=interpreted_query,
            applied_filters=result["resolved_filters"],
            unresolved_filters=result["unresolved_filters"],
            timings_ms=timings_ms,
            usage=usage,
            company_id=self.company_id,
        )

    def _record_usage(self, result: dict) -> dict[str, Any]:
        company_id = self.company_id or "legacy"
        events = []
        query_metrics = result.get("query_model_metrics") or {}
        query_attempts = query_metrics.get("attempts") or (
            [query_metrics] if query_metrics.get("model") else []
        )
        for attempt in query_attempts:
            model = str(attempt.get("model") or "")
            if not model:
                continue
            events.append(
                {
                    "provider": "google",
                    "model": model,
                    "operation": "query_planning",
                    "status": str(attempt.get("status") or "success"),
                    "input_tokens": int(
                        attempt.get("input_tokens", 0) or 0
                    ),
                    "output_tokens": int(
                        attempt.get("output_tokens", 0) or 0
                    ),
                    "total_tokens": int(
                        attempt.get("total_tokens", 0) or 0
                    ),
                }
            )
        for attempt in result.get("reranker_attempts") or []:
            provider_name = str(attempt.get("provider") or "")
            provider = (
                "voyage"
                if provider_name.startswith("voyage")
                else provider_name
            )
            usage = attempt.get("usage") or {}
            events.append(
                {
                    "provider": provider,
                    "model": str(attempt.get("model") or provider_name),
                    "operation": "reranking",
                    "status": str(attempt.get("status") or "success"),
                    "input_tokens": int(
                        usage.get("input_tokens", 0) or 0
                    ),
                    "output_tokens": int(
                        usage.get("output_tokens", 0) or 0
                    ),
                    "total_tokens": int(
                        usage.get("total_tokens", 0) or 0
                    ),
                }
            )
        execution_path = str(
            result.get("query_plan", {}).get("execution_path", "unknown")
        )
        if self.usage_store is not None:
            self.usage_store.record(
                company_id=company_id,
                provider="internal",
                model=execution_path,
                operation="search",
                status=(
                    "cache_hit"
                    if result.get("result_cache_hit")
                    else "success"
                ),
            )
            for event in events:
                self.usage_store.record(company_id=company_id, **event)
        return {
            "tracked": self.usage_store is not None,
            "model_requests": len(events),
            "input_tokens": sum(
                event["input_tokens"] for event in events
            ),
            "output_tokens": sum(
                event["output_tokens"] for event in events
            ),
            "total_tokens": sum(
                event["total_tokens"] for event in events
            ),
            "breakdown": events,
        }

    def usage_summary(self, month_utc: str | None = None) -> dict[str, Any]:
        if self.usage_store is None:
            raise RuntimeError("Monthly usage tracking is disabled.")
        return self.usage_store.summary(
            self.company_id or "legacy",
            month_utc,
        )

    @staticmethod
    def _page(
        session: SearchSession,
        offset: int,
        page_size: int,
        cached: bool,
    ) -> SearchResponse:
        if offset > len(session.items):
            raise InvalidCursorError("The cursor offset is invalid.")
        end = min(offset + page_size, len(session.items))
        items = session.items[offset:end]
        has_more = end < len(session.items)
        next_cursor = (
            encode_cursor(session.search_id, end)
            if has_more
            else None
        )
        return SearchResponse(
            company_id=session.company_id,
            search_id=session.search_id,
            query=session.query,
            cached=cached,
            items=items,
            interpreted_query=session.interpreted_query,
            applied_filters=session.applied_filters,
            unresolved_filters=session.unresolved_filters,
            timings_ms=session.timings_ms,
            usage=session.usage,
            pagination=PaginationResponse(
                page_size=page_size,
                returned=len(items),
                offset=offset,
                total_results=len(session.items),
                has_more=has_more,
                next_cursor=next_cursor,
            ),
        )

    def close(self) -> None:
        close = getattr(self.engine, "close", None)
        if callable(close):
            close()


def product_is_visible(product: dict[str, Any]) -> bool:
    # ads.type is the canonical offer/wanted discriminator. Do not infer
    # visibility from ads.status: the source uses multiple status lifecycles,
    # and valid wanted rows commonly carry status=2.
    return product.get("deleted_at") is None


def public_product(
    product: dict[str, Any],
    *,
    fields: tuple[str, ...] = PUBLIC_PRODUCT_FIELDS,
    field_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    mapping = field_mapping or {}
    output = {}
    for public_field in fields:
        source_field = mapping.get(public_field, public_field)
        if source_field in product:
            output[public_field] = product[source_field]
    return output


class TenantServicePool:
    """Lazily opens isolated tenant search engines with an LRU memory bound."""

    def __init__(
        self,
        registry: TenantRegistry,
        *,
        shared_cache=None,
        max_services: int = API_TENANT_ENGINE_CACHE_SIZE,
        engine_factory=None,
        usage_store: MonthlyUsageStore | None = None,
    ):
        if max_services <= 0:
            raise ValueError("max_services must be greater than zero")
        self.registry = registry
        self.shared_cache = shared_cache
        self.max_services = max_services
        self.engine_factory = engine_factory
        self.usage_store = usage_store
        self.shared_reranker = SharedReranker()
        self.reranker_load_ms = 0.0
        self.embedding_warmup: dict[str, Any] = {}
        self._services: OrderedDict[str, ProductSearchService] = OrderedDict()
        self._lock = threading.Lock()

    def preload_reranker(self) -> float:
        _ranker, seconds = self.shared_reranker.ensure()
        self.reranker_load_ms = max(self.reranker_load_ms, seconds * 1000)
        return seconds * 1000

    def get(self, company_id: str) -> ProductSearchService:
        with self._lock:
            existing = self._services.get(company_id)
            if existing is not None:
                self._services.move_to_end(company_id)
                return existing
            profile = self.registry.get(company_id)
            service = self._build_service(profile)
            self._services[company_id] = service
            while len(self._services) > self.max_services:
                _evicted_id, evicted = self._services.popitem(last=False)
                evicted.close()
            return service

    def _build_service(self, profile: TenantProfile) -> ProductSearchService:
        if profile.planner_adapter != "gainr":
            raise RuntimeError(
                f"Unsupported planner adapter {profile.planner_adapter!r} for "
                f"tenant {profile.company_id!r}."
            )
        if self.engine_factory is None:
            collection = get_tenant_vector_collection(
                profile,
                create=False,
            )
            bm25_index = PersistentBM25Index(profile.storage.bm25_path)
            engine = ProductSearchEngine(
                collection=collection,
                bm25_index=bm25_index,
                shared_plan_cache=self.shared_cache,
                company_id=profile.company_id,
                mysql_config=profile.database,
                shared_reranker=self.shared_reranker,
                close_bm25_index=True,
                planner_enabled=profile.planner_enabled,
                planner_prompt_context=profile.planner_prompt_context,
            )
        else:
            engine = self.engine_factory(
                profile,
                self.shared_cache,
                self.shared_reranker,
            )
        service = ProductSearchService(
            engine,
            company_id=profile.company_id,
            public_fields=profile.payload.public_fields,
            field_mapping=profile.payload.field_mapping,
            usage_store=self.usage_store,
        )
        service.reranker_load_ms = self.reranker_load_ms
        service.embedding_warmup = self.embedding_warmup
        return service

    def close(self) -> None:
        with self._lock:
            services = list(self._services.values())
            self._services.clear()
        for service in services:
            service.close()


def create_app(
    engine_factory: Callable[[], ProductSearchEngine] = ProductSearchEngine,
    service: ProductSearchService | None = None,
    tenant_registry: TenantRegistry | None = None,
    tenant_engine_factory=None,
    rate_limiter: TenantRateLimiter | None = None,
    preload_models: bool | None = None,
    usage_store: MonthlyUsageStore | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        engine = None
        pool = None
        redis_cache = None
        active_usage_store = usage_store
        owns_usage_store = False
        if active_usage_store is None and USAGE_TRACKING_ENABLED:
            active_usage_store = MonthlyUsageStore(USAGE_DB_PATH)
            owns_usage_store = True
        application.state.usage_store = active_usage_store
        tenant_mode = service is None and (
            tenant_registry is not None or API_AUTH_ENABLED
        )
        application.state.tenant_mode = tenant_mode
        if tenant_mode:
            redis_cache = create_redis_cache(
                REDIS_ENABLED,
                REDIS_URL,
                REDIS_KEY_PREFIX,
            )
            registry = tenant_registry or load_tenant_registry(
                API_TENANT_CONFIG_DIR,
                require_api_keys=True,
            )
            pool = TenantServicePool(
                registry,
                shared_cache=redis_cache,
                max_services=API_TENANT_ENGINE_CACHE_SIZE,
                engine_factory=tenant_engine_factory,
                usage_store=active_usage_store,
            )
            application.state.tenant_registry = registry
            application.state.tenant_service_pool = pool
            application.state.rate_limiter = rate_limiter or TenantRateLimiter(
                redis_cache
            )
            application.state.search_service = None
        elif service is None:
            redis_cache = create_redis_cache(
                REDIS_ENABLED,
                REDIS_URL,
                REDIS_KEY_PREFIX,
            )
            engine = engine_factory()
            engine.set_shared_plan_cache(redis_cache)
            application.state.search_service = ProductSearchService(
                engine,
                usage_store=active_usage_store,
            )
        else:
            application.state.search_service = service

        preload_reranker = (
            API_PRELOAD_RERANKER
            if preload_models is None
            else preload_models
        )
        preload_embedding = (
            API_PRELOAD_EMBEDDING
            if preload_models is None
            else preload_models
        )
        if preload_reranker:
            LOGGER.info("Initializing the configured reranker chain...")
            load_ms = (
                pool.preload_reranker()
                if pool is not None
                else application.state.search_service.warmup()
            )
            ranker = (
                pool.shared_reranker.ranker
                if pool is not None
                else application.state.search_service.engine.ranker
            )
            LOGGER.info(
                "Reranker chain ready model_order=%s in %.0f ms.",
                getattr(ranker, "model_label", RERANK_MODEL),
                load_ms,
            )
        if preload_embedding and service is None:
            LOGGER.info("Preloading the Ollama embedding model...")
            embedding_warmup = preload_ollama_embedding()
            if pool is not None:
                pool.embedding_warmup = embedding_warmup
            else:
                application.state.search_service.embedding_warmup = embedding_warmup
            LOGGER.info(
                "Ollama embedding model ready in %.0f ms.",
                embedding_warmup["embedding_model"].get("total_ms", 0.0),
            )
        try:
            yield
        finally:
            if pool is not None:
                pool.close()
            if engine is not None:
                engine.close()
            if redis_cache is not None:
                redis_cache.close()
            if owns_usage_store and active_usage_store is not None:
                active_usage_store.close()

    application = FastAPI(
        title=f"{APP_NAME} API",
        version="1.0.0",
        lifespan=lifespan,
    )
    if API_CORS_ORIGINS:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=API_CORS_ORIGINS,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def resolve_company_profile(
        api_key: str | None,
        *,
        company_endpoint: str | None = None,
    ) -> TenantProfile:
        if not application.state.tenant_mode:
            raise HTTPException(
                status_code=404,
                detail="Company authentication requires tenant mode.",
            )
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        profile = application.state.tenant_registry.resolve_api_key(api_key)
        if profile is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key.",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if company_endpoint is not None:
            endpoint_profile = (
                application.state.tenant_registry.resolve_endpoint(
                    company_endpoint
                )
            )
            if endpoint_profile is None:
                raise HTTPException(
                    status_code=404,
                    detail="Unknown company endpoint.",
                )
            if endpoint_profile.company_id != profile.company_id:
                raise HTTPException(
                    status_code=403,
                    detail="API key does not match the company endpoint.",
                )
        return profile

    def resolve_service(
        api_key: str | None,
        *,
        apply_rate_limit: bool,
        company_endpoint: str | None = None,
    ) -> ProductSearchService:
        if not application.state.tenant_mode:
            if company_endpoint is not None:
                raise HTTPException(
                    status_code=404,
                    detail="Company endpoints require tenant mode.",
                )
            return application.state.search_service
        profile = resolve_company_profile(
            api_key,
            company_endpoint=company_endpoint,
        )
        if apply_rate_limit and API_RATE_LIMIT_ENABLED:
            allowed, _remaining = application.state.rate_limiter.allow(
                profile,
                hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
            )
            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Company rate limit exceeded.",
                    headers={"Retry-After": "1"},
                )
        return application.state.tenant_service_pool.get(profile.company_id)

    def company_search_request(
        company_endpoint: str,
        payload: dict[str, Any],
    ) -> SearchRequest:
        if not application.state.tenant_mode:
            raise HTTPException(
                status_code=404,
                detail="Company endpoints require tenant mode.",
            )
        profile = application.state.tenant_registry.resolve_endpoint(
            company_endpoint
        )
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Unknown company endpoint.",
            )
        mapping = profile.payload.request_mapping or {
            "query": "query",
            "cursor": "cursor",
            "page_size": "page_size",
        }
        allowed_fields = set(mapping.values())
        unexpected = sorted(set(payload) - allowed_fields)
        if unexpected:
            raise HTTPException(
                status_code=422,
                detail=f"Unexpected request fields: {unexpected}",
            )
        normalized = {
            canonical: payload[company_field]
            for canonical, company_field in mapping.items()
            if company_field in payload
        }
        try:
            return SearchRequest.model_validate(normalized)
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=exc.errors(include_context=False),
            ) from exc

    def execute_search(
        request: SearchRequest,
        x_api_key: str | None,
        *,
        company_endpoint: str | None = None,
    ) -> SearchResponse:
        try:
            search_service = resolve_service(
                x_api_key,
                apply_rate_limit=True,
                company_endpoint=company_endpoint,
            )
            return search_service.search(request)
        except InvalidCursorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ExpiredCursorError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get("/api/v1/ready", tags=["system"])
    def ready() -> dict[str, Any]:
        tenant_mode = bool(application.state.tenant_mode)
        registry = getattr(application.state, "tenant_registry", None)
        return {
            "status": "ok",
            "tenant_mode": tenant_mode,
            "configured_companies": (
                len(registry.profiles) if registry is not None else 1
            ),
        }

    @application.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health(
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> HealthResponse:
        try:
            return resolve_service(
                x_api_key,
                apply_rate_limit=False,
            ).health()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.post(
        "/api/v1/search",
        response_model=SearchResponse,
        tags=["search"],
    )
    def search(
        request: SearchRequest,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> SearchResponse:
        return execute_search(request, x_api_key)

    @application.get(
        "/api/v1/{company_endpoint}/health",
        response_model=HealthResponse,
        tags=["company"],
    )
    def company_health(
        company_endpoint: str,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> HealthResponse:
        try:
            return resolve_service(
                x_api_key,
                apply_rate_limit=False,
                company_endpoint=company_endpoint,
            ).health()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @application.get(
        "/api/v1/{company_endpoint}/auth/verify",
        tags=["company"],
    )
    def company_auth_verify(
        company_endpoint: str,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        profile = resolve_company_profile(
            x_api_key,
            company_endpoint=company_endpoint,
        )
        return {
            "authorized": True,
            "company_id": profile.company_id,
            "endpoint_slug": profile.endpoint_slug,
        }

    @application.post(
        "/api/v1/{company_endpoint}/search",
        response_model=SearchResponse,
        tags=["company"],
    )
    def company_search(
        company_endpoint: str,
        payload: dict[str, Any],
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> SearchResponse:
        request = company_search_request(company_endpoint, payload)
        return execute_search(
            request,
            x_api_key,
            company_endpoint=company_endpoint,
        )

    @application.get(
        "/api/v1/{company_endpoint}/usage",
        tags=["company"],
    )
    def company_usage(
        company_endpoint: str,
        month: str | None = Query(default=None, pattern=r"^\d{4}-\d{2}$"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> dict[str, Any]:
        try:
            return resolve_service(
                x_api_key,
                apply_rate_limit=False,
                company_endpoint=company_endpoint,
            ).usage_summary(month)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
