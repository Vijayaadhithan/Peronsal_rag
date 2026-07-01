import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0."
    )


def _load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


CONFIG = _load_config()

SOURCE_FILES_DIR = PROJECT_ROOT / "data" / "raw_docs"
RAW_DOCS_DIR = SOURCE_FILES_DIR
CHROMA_DIR = PROJECT_ROOT / "storage" / "chroma"
BM25_INDEX_PATH = PROJECT_ROOT / "storage" / "bm25.sqlite3"
APP_NAME = CONFIG.get("app_name", "Local Data Assistant")
COLLECTION_NAME = CONFIG.get("collection_name", "local_data")
USAGE_TRACKING_ENABLED = _env_bool(
    "USAGE_TRACKING_ENABLED",
    bool(CONFIG.get("api", {}).get("usage_tracking_enabled", True)),
)
_usage_db_path = Path(
    os.getenv(
        "USAGE_DB_PATH",
        str(
            CONFIG.get("api", {}).get(
                "usage_db_path",
                "storage/usage.sqlite3",
            )
        ),
    )
)
USAGE_DB_PATH = (
    _usage_db_path
    if _usage_db_path.is_absolute()
    else PROJECT_ROOT / _usage_db_path
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE")
if OLLAMA_KEEP_ALIVE is None:
    OLLAMA_KEEP_ALIVE = CONFIG.get("ollama", {}).get("keep_alive", -1)
EMBED_MODEL = CONFIG.get("embedding", {}).get("model", "embeddinggemma:latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_BASE_URL = os.getenv(
    "GEMINI_API_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
).rstrip("/")
QUERY_EXTRACT_CONFIG = CONFIG.get("query_extraction", {})
_query_extract_models = QUERY_EXTRACT_CONFIG.get("models")
if not _query_extract_models:
    _query_extract_models = [
        QUERY_EXTRACT_CONFIG.get("model", "gemma-4-26b-a4b-it")
    ]
elif isinstance(_query_extract_models, str):
    _query_extract_models = [_query_extract_models]
QUERY_EXTRACT_MODELS = tuple(
    str(model).strip()
    for model in _query_extract_models
    if str(model).strip()
)
if not QUERY_EXTRACT_MODELS:
    raise ValueError("query_extraction.models must contain at least one model.")
QUERY_EXTRACT_MODEL = QUERY_EXTRACT_MODELS[0]
QUERY_EXTRACT_TEMPERATURE = float(
    QUERY_EXTRACT_CONFIG.get("temperature", 0)
)
QUERY_EXTRACT_TIMEOUT_SECONDS = float(
    os.getenv(
        "GEMINI_TIMEOUT_SECONDS",
        str(QUERY_EXTRACT_CONFIG.get("timeout_seconds", 10)),
    )
)
if QUERY_EXTRACT_TIMEOUT_SECONDS <= 0:
    raise ValueError("GEMINI_TIMEOUT_SECONDS must be greater than zero.")
QUERY_DETERMINISTIC_FAST_PATH = bool(
    QUERY_EXTRACT_CONFIG.get("deterministic_fast_path", True)
)
QUERY_FUZZY_MATCHING = bool(
    QUERY_EXTRACT_CONFIG.get("fuzzy_matching", True)
)
QUERY_PLAN_CACHE_SIZE = int(
    QUERY_EXTRACT_CONFIG.get("cache_size", 500)
)
QUERY_PLAN_CACHE_TTL_SECONDS = int(
    QUERY_EXTRACT_CONFIG.get("cache_ttl_seconds", 900)
)
REDIS_CONFIG = CONFIG.get("redis", {})
REDIS_ENABLED = _env_bool(
    "REDIS_ENABLED",
    bool(REDIS_CONFIG.get("enabled", True)),
)
REDIS_URL = os.getenv(
    "REDIS_URL",
    str(REDIS_CONFIG.get("url", "redis://127.0.0.1:6379/0")),
)
REDIS_KEY_PREFIX = os.getenv(
    "REDIS_KEY_PREFIX",
    str(REDIS_CONFIG.get("key_prefix", "semantic_ads")),
)
RESULT_CACHE_ENABLED = _env_bool(
    "REDIS_RESULT_CACHE_ENABLED",
    bool(REDIS_CONFIG.get("result_cache_enabled", True)),
)
RESULT_CACHE_TTL_SECONDS = int(
    os.getenv(
        "REDIS_RESULT_CACHE_TTL_SECONDS",
        str(REDIS_CONFIG.get("result_cache_ttl_seconds", 300)),
    )
)
if RESULT_CACHE_TTL_SECONDS <= 0:
    raise ValueError("REDIS_RESULT_CACHE_TTL_SECONDS must be greater than zero.")

CHUNK_SIZE = int(CONFIG.get("chunking", {}).get("chunk_size", 512))
CHUNK_OVERLAP = int(CONFIG.get("chunking", {}).get("chunk_overlap", 80))

UNPRICED_RENTAL_FEE_CEILING = float(
    CONFIG.get("retrieval", {}).get("unpriced_rental_fee_ceiling", 1)
)
if UNPRICED_RENTAL_FEE_CEILING < 0:
    raise ValueError(
        "retrieval.unpriced_rental_fee_ceiling cannot be negative."
    )

VECTOR_CANDIDATE_K = int(
    CONFIG.get("retrieval", {}).get("vector_candidate_k", 100)
)
VECTOR_TOP_K = int(CONFIG.get("retrieval", {}).get("vector_top_k", 15))
BM25_TOP_K = int(CONFIG.get("retrieval", {}).get("bm25_top_k", 15))
HYBRID_CANDIDATE_K = int(
    CONFIG.get("retrieval", {}).get("hybrid_candidate_k", 60)
)
RERANK_CANDIDATE_K = int(
    CONFIG.get("retrieval", {}).get("rerank_candidate_k", 60)
)
PRIMARY_RANKED_K = int(
    CONFIG.get("retrieval", {}).get("primary_ranked_k", 60)
)
RELATED_TAIL_ENABLED = bool(
    CONFIG.get("retrieval", {}).get("related_tail_enabled", True)
)
RETRIEVAL_OVERFETCH_FACTOR = int(
    CONFIG.get("retrieval", {}).get("overfetch_factor", 2)
)
RRF_CONSTANT = int(CONFIG.get("retrieval", {}).get("rrf_constant", 60))
VECTOR_WEIGHT = float(CONFIG.get("retrieval", {}).get("vector_weight", 1.0))
BM25_WEIGHT = float(CONFIG.get("retrieval", {}).get("bm25_weight", 1.0))
SOFT_CATEGORY_BOOST = float(
    CONFIG.get("retrieval", {}).get("soft_category_boost", 0.005)
)
RERANK_TOP_K = int(CONFIG.get("retrieval", {}).get("final_top_k", 6))
RERANK_MODEL = CONFIG.get("retrieval", {}).get(
    "reranker_model", "Alibaba-NLP/gte-reranker-modernbert-base"
)
RERANK_BATCH_SIZE = int(
    CONFIG.get("retrieval", {}).get("reranker_batch_size", 4)
)
RERANK_MAX_LENGTH = int(
    CONFIG.get("retrieval", {}).get("reranker_max_length", 512)
)
RERANK_USE_FP16 = bool(
    CONFIG.get("retrieval", {}).get("reranker_use_fp16", False)
)
RERANK_PROVIDER_CONFIG = CONFIG.get("retrieval", {}).get(
    "reranker_providers",
    {},
)
_rerank_provider_order = os.getenv(
    "RERANK_PROVIDER_ORDER",
    ",".join(
        RERANK_PROVIDER_CONFIG.get(
            "order",
            ["jina", "voyage-2.5", "voyage-2.5-lite"],
        )
    ),
)
RERANK_PROVIDER_ORDER = tuple(
    provider.strip().casefold()
    for provider in _rerank_provider_order.split(",")
    if provider.strip()
)
_invalid_rerank_providers = sorted(
    set(RERANK_PROVIDER_ORDER)
    - {
        "jina",
        "voyage",
        "voyage-2.5",
        "voyage-2.5-lite",
        "local",
    }
)
if _invalid_rerank_providers:
    raise ValueError(
        f"Unsupported reranker providers: {_invalid_rerank_providers}"
    )
if not RERANK_PROVIDER_ORDER:
    raise ValueError("RERANK_PROVIDER_ORDER must contain at least one provider.")
RERANK_API_TIMEOUT_SECONDS = float(
    os.getenv(
        "RERANK_API_TIMEOUT_SECONDS",
        str(RERANK_PROVIDER_CONFIG.get("timeout_seconds", 5)),
    )
)
RERANK_MAX_DOCUMENT_CHARS = int(
    os.getenv(
        "RERANK_MAX_DOCUMENT_CHARS",
        str(RERANK_PROVIDER_CONFIG.get("max_document_chars", 4000)),
    )
)
if RERANK_API_TIMEOUT_SECONDS <= 0 or RERANK_MAX_DOCUMENT_CHARS <= 0:
    raise ValueError("Reranker timeout and document character limit must be positive.")
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_RERANK_URL = os.getenv(
    "VOYAGE_RERANK_URL",
    "https://api.voyageai.com/v1/rerank",
)
VOYAGE_RERANK_MODEL = os.getenv(
    "VOYAGE_RERANK_MODEL",
    str(RERANK_PROVIDER_CONFIG.get("voyage_model", "rerank-2.5")),
)
VOYAGE_RERANK_LITE_MODEL = os.getenv(
    "VOYAGE_RERANK_LITE_MODEL",
    str(
        RERANK_PROVIDER_CONFIG.get(
            "voyage_lite_model",
            "rerank-2.5-lite",
        )
    ),
)
VOYAGE_RERANK_RPM_PER_MODEL = int(
    os.getenv(
        "VOYAGE_RERANK_RPM_PER_MODEL",
        str(
            RERANK_PROVIDER_CONFIG.get(
                "voyage_requests_per_minute_per_model",
                3,
            )
        ),
    )
)
if VOYAGE_RERANK_RPM_PER_MODEL <= 0:
    raise ValueError("VOYAGE_RERANK_RPM_PER_MODEL must be greater than zero.")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_URL = os.getenv(
    "JINA_RERANK_URL",
    "https://api.jina.ai/v1/rerank",
)
JINA_RERANK_MODEL = os.getenv(
    "JINA_RERANK_MODEL",
    str(
        RERANK_PROVIDER_CONFIG.get(
            "jina_model",
            "jina-reranker-v2-base-multilingual",
        )
    ),
)

API_HOST = os.getenv(
    "API_HOST",
    str(CONFIG.get("api", {}).get("host", "127.0.0.1")),
)
API_PORT = int(
    os.getenv(
        "API_PORT",
        str(CONFIG.get("api", {}).get("port", 8000)),
    )
)
API_LOG_LEVEL = os.getenv(
    "API_LOG_LEVEL",
    str(CONFIG.get("api", {}).get("log_level", "info")),
).lower()
API_DEFAULT_PAGE_SIZE = int(
    CONFIG.get("api", {}).get("default_page_size", 20)
)
API_PRELOAD_RERANKER = bool(
    CONFIG.get("api", {}).get("preload_reranker", True)
)
API_PRELOAD_EMBEDDING = bool(
    CONFIG.get("api", {}).get("preload_embedding", True)
)
API_MAX_PAGE_SIZE = int(CONFIG.get("api", {}).get("max_page_size", 20))
API_MAX_RESULTS = int(CONFIG.get("api", {}).get("max_results", 200))
API_SESSION_TTL_SECONDS = int(
    CONFIG.get("api", {}).get("session_ttl_seconds", 600)
)
API_MAX_SESSIONS = int(CONFIG.get("api", {}).get("max_sessions", 500))
API_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("API_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
API_AUTH_ENABLED = _env_bool(
    "API_AUTH_ENABLED",
    bool(CONFIG.get("api", {}).get("auth_enabled", False)),
)
API_RATE_LIMIT_ENABLED = _env_bool(
    "API_RATE_LIMIT_ENABLED",
    bool(CONFIG.get("api", {}).get("rate_limit_enabled", True)),
)
API_TENANT_CONFIG_DIR = PROJECT_ROOT / os.getenv(
    "API_TENANT_CONFIG_DIR",
    str(CONFIG.get("api", {}).get("tenant_config_dir", "configs/tenants")),
)
API_TENANT_ENGINE_CACHE_SIZE = int(
    os.getenv(
        "API_TENANT_ENGINE_CACHE_SIZE",
        str(CONFIG.get("api", {}).get("tenant_engine_cache_size", 8)),
    )
)
API_TENANT_MAX_CONCURRENT_SEARCHES = int(
    os.getenv(
        "API_TENANT_MAX_CONCURRENT_SEARCHES",
        str(
            CONFIG.get("api", {}).get(
                "tenant_max_concurrent_searches",
                4,
            )
        ),
    )
)
if API_TENANT_ENGINE_CACHE_SIZE <= 0:
    raise ValueError("API_TENANT_ENGINE_CACHE_SIZE must be greater than zero.")
if API_TENANT_MAX_CONCURRENT_SEARCHES <= 0:
    raise ValueError(
        "API_TENANT_MAX_CONCURRENT_SEARCHES must be greater than zero."
    )

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_ht_test")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_TABLE = os.getenv("MYSQL_TABLE", "ads_search_ready")
MYSQL_CONTENT_COLUMN = os.getenv("MYSQL_CONTENT_COLUMN", "embedding_content")
MYSQL_BM25_COLUMN = os.getenv("MYSQL_BM25_COLUMN", "bm25_content")
MYSQL_SEARCH_ID_COLUMN = os.getenv("MYSQL_SEARCH_ID_COLUMN", "id")
MYSQL_RESULT_TABLE = os.getenv("MYSQL_RESULT_TABLE", "ads")
MYSQL_RESULT_ID_COLUMN = os.getenv("MYSQL_RESULT_ID_COLUMN", "id")
