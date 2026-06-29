import base64
import binascii
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ollama_client import preload_ollama_embedding
from redis_cache import create_redis_cache
from search_engine import ProductSearchEngine
from settings import (
    API_CORS_ORIGINS,
    API_DEFAULT_PAGE_SIZE,
    API_MAX_PAGE_SIZE,
    API_MAX_RESULTS,
    API_MAX_SESSIONS,
    API_PRELOAD_EMBEDDING,
    API_PRELOAD_RERANKER,
    API_SESSION_TTL_SECONDS,
    APP_NAME,
    RERANK_MODEL,
    REDIS_ENABLED,
    REDIS_KEY_PREFIX,
    REDIS_URL,
)

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
    search_id: str
    query: str
    cached: bool
    items: list[dict[str, Any]]
    interpreted_query: dict[str, Any]
    applied_filters: dict[str, Any]
    unresolved_filters: dict[str, Any]
    timings_ms: dict[str, float]
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


@dataclass
class SearchSession:
    search_id: str
    query: str
    items: list[dict[str, Any]]
    interpreted_query: dict[str, Any]
    applied_filters: dict[str, Any]
    unresolved_filters: dict[str, Any]
    timings_ms: dict[str, float]
    expires_at: float


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
    ):
        if max_results <= 0:
            raise ValueError("Maximum results must be greater than zero.")
        self.engine = engine
        self.sessions = sessions or SearchSessionStore()
        self.max_results = max_results
        self._engine_lock = threading.Lock()
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
            reranker_model=RERANK_MODEL,
            reranker_loaded=self.engine.ranker is not None,
            reranker_load_ms=self.reranker_load_ms,
            embedding_warmup=self.embedding_warmup,
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
        with self._engine_lock:
            result = self.engine.search(query, limit=self.max_results)
        total_ms = (time.perf_counter() - started) * 1000
        query_plan = result["query_plan"]
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
                public_product(product)
                for product in result["products"]
                if product_is_visible(product)
            ],
            interpreted_query=interpreted_query,
            applied_filters=result["resolved_filters"],
            unresolved_filters=result["unresolved_filters"],
            timings_ms=timings_ms,
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
            search_id=session.search_id,
            query=session.query,
            cached=cached,
            items=items,
            interpreted_query=session.interpreted_query,
            applied_filters=session.applied_filters,
            unresolved_filters=session.unresolved_filters,
            timings_ms=session.timings_ms,
            pagination=PaginationResponse(
                page_size=page_size,
                returned=len(items),
                offset=offset,
                total_results=len(session.items),
                has_more=has_more,
                next_cursor=next_cursor,
            ),
        )


def product_is_visible(product: dict[str, Any]) -> bool:
    # ads.type is the canonical offer/wanted discriminator. Do not infer
    # visibility from ads.status: the source uses multiple status lifecycles,
    # and valid wanted rows commonly carry status=2.
    return product.get("deleted_at") is None


def public_product(product: dict[str, Any]) -> dict[str, Any]:
    return {
        field: product[field]
        for field in PUBLIC_PRODUCT_FIELDS
        if field in product
    }


def create_app(
    engine_factory: Callable[[], ProductSearchEngine] = ProductSearchEngine,
    service: ProductSearchService | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if service is None:
            redis_cache = create_redis_cache(
                REDIS_ENABLED,
                REDIS_URL,
                REDIS_KEY_PREFIX,
            )
            engine = engine_factory()
            engine.set_shared_plan_cache(redis_cache)
            application.state.search_service = ProductSearchService(engine)
        else:
            engine = None
            redis_cache = None
            application.state.search_service = service
        if API_PRELOAD_RERANKER:
            LOGGER.info("Preloading reranker %s once for this process...", RERANK_MODEL)
            load_ms = application.state.search_service.warmup()
            LOGGER.info("Reranker ready in %.0f ms.", load_ms)
        if API_PRELOAD_EMBEDDING and service is None:
            LOGGER.info("Preloading the Ollama embedding model...")
            embedding_warmup = preload_ollama_embedding()
            application.state.search_service.embedding_warmup = embedding_warmup
            LOGGER.info(
                "Ollama embedding model ready in %.0f ms.",
                embedding_warmup["embedding_model"].get("total_ms", 0.0),
            )
        try:
            yield
        finally:
            if engine is not None:
                engine.close()
            if redis_cache is not None:
                redis_cache.close()

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
            allow_headers=["Content-Type"],
        )

    @application.get(
        "/api/v1/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health() -> HealthResponse:
        return application.state.search_service.health()

    @application.post(
        "/api/v1/search",
        response_model=SearchResponse,
        tags=["search"],
    )
    def search(request: SearchRequest) -> SearchResponse:
        try:
            return application.state.search_service.search(request)
        except InvalidCursorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ExpiredCursorError as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return application


app = create_app()
