import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


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

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
EMBED_MODEL = CONFIG.get("embedding", {}).get("model", "embeddinggemma:latest")
LLM_MODEL = CONFIG.get("llm", {}).get("model", "gemma4:12b")
TEMPERATURE = float(CONFIG.get("llm", {}).get("temperature", 0.2))
LLM_THINK = bool(CONFIG.get("llm", {}).get("think", False))
QUERY_EXTRACT_MODEL = CONFIG.get("query_extraction", {}).get(
    "model", LLM_MODEL
)
QUERY_EXTRACT_TEMPERATURE = float(
    CONFIG.get("query_extraction", {}).get("temperature", 0)
)

CHUNK_SIZE = int(CONFIG.get("chunking", {}).get("chunk_size", 512))
CHUNK_OVERLAP = int(CONFIG.get("chunking", {}).get("chunk_overlap", 80))

VECTOR_CANDIDATE_K = int(
    CONFIG.get("retrieval", {}).get("vector_candidate_k", 100)
)
VECTOR_TOP_K = int(CONFIG.get("retrieval", {}).get("vector_top_k", 15))
BM25_TOP_K = int(CONFIG.get("retrieval", {}).get("bm25_top_k", 15))
RERANK_TOP_K = int(CONFIG.get("retrieval", {}).get("final_top_k", 6))
RERANK_MODEL = CONFIG.get("retrieval", {}).get(
    "reranker_model", "BAAI/bge-reranker-large"
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
