import json
import re
import time

import requests
import chromadb

from bm25_index import PersistentBM25Index
from settings import (
    APP_NAME,
    BM25_TOP_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_RESULT_ID_COLUMN,
    MYSQL_RESULT_TABLE,
    MYSQL_SEARCH_ID_COLUMN,
    MYSQL_TABLE,
    MYSQL_USER,
    OLLAMA_BASE_URL,
    QUERY_EXTRACT_MODEL,
    QUERY_EXTRACT_TEMPERATURE,
    RERANK_BATCH_SIZE,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RERANK_TOP_K,
    RERANK_USE_FP16,
    VECTOR_CANDIDATE_K,
    VECTOR_TOP_K,
)

QUERY_FILTER_FIELDS = {
    "main_category": "main_category_name",
    "subcategory": "subcategory_name",
    "state": "state_name",
    "city": "city_name",
    "locality": "locality_name",
    "rental_duration": "rental_duration",
}
QUERY_FILTER_KEYS = (*QUERY_FILTER_FIELDS, "min_rental_fee", "max_rental_fee")
QUERY_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "semantic_query": {"type": "string"},
        "keyword_query": {"type": "string"},
        "filters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "main_category": {
                    "type": ["string", "null"],
                    "description": (
                        "Explicit broad department, such as Accommodation & Spaces "
                        "or Vehicles; never put a specific product type here."
                    ),
                },
                "subcategory": {
                    "type": ["string", "null"],
                    "description": (
                        "Explicit specific product or listing type, such as Mansion, "
                        "Bicycle, or Laptop."
                    ),
                },
                "state": {
                    "type": ["string", "null"],
                    "description": "Explicit state location.",
                },
                "city": {
                    "type": ["string", "null"],
                    "description": "Explicit city location.",
                },
                "locality": {
                    "type": ["string", "null"],
                    "description": "Explicit neighborhood or locality.",
                },
                "rental_duration": {
                    "type": ["string", "null"],
                    "description": (
                        "Explicit rental period such as Per Hour, Per Day, Per Month, "
                        "or Per Year."
                    ),
                },
                "min_rental_fee": {"type": ["number", "null"]},
                "max_rental_fee": {"type": ["number", "null"]},
            },
            "required": list(QUERY_FILTER_KEYS),
        },
    },
    "required": ["semantic_query", "keyword_query", "filters"],
}


def embed_text(text: str):
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": text,
                "keep_alive": "30m",
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
    except (requests.RequestException, KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Cannot get embeddings from Ollama at {OLLAMA_BASE_URL}. "
            f"Start Ollama and confirm '{EMBED_MODEL}' is installed."
        ) from exc


def default_query_plan(query: str, fallback_reason: str | None = None) -> dict:
    return {
        "semantic_query": query,
        "keyword_query": query,
        "filters": {key: None for key in QUERY_FILTER_KEYS},
        "fallback_reason": fallback_reason,
    }


def optional_text(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def optional_number(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def text_mentions_filter(text: str, value: str) -> bool:
    normalized_text = normalize_filter_value(text)
    normalized_value = normalize_filter_value(value)
    if normalized_value in normalized_text:
        return True
    compact_text = re.sub(r"\W+", "", normalized_text)
    compact_value = re.sub(r"\W+", "", normalized_value)
    return bool(compact_value and compact_value in compact_text)


def parse_query_plan(content: str, original_query: str) -> dict:
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("query extraction response must be a JSON object")

    semantic_query = optional_text(parsed.get("semantic_query")) or original_query
    keyword_query = optional_text(parsed.get("keyword_query")) or original_query
    raw_filters = parsed.get("filters")
    if not isinstance(raw_filters, dict):
        raw_filters = {}

    filters = {
        key: optional_text(raw_filters.get(key))
        for key in QUERY_FILTER_FIELDS
    }
    for parent_key in ("main_category", "state"):
        value = filters[parent_key]
        if value is not None and not text_mentions_filter(original_query, value):
            filters[parent_key] = None
    filters["min_rental_fee"] = optional_number(
        raw_filters.get("min_rental_fee")
    )
    filters["max_rental_fee"] = optional_number(
        raw_filters.get("max_rental_fee")
    )
    minimum = filters["min_rental_fee"]
    maximum = filters["max_rental_fee"]
    if minimum is not None and maximum is not None and minimum > maximum:
        filters["min_rental_fee"], filters["max_rental_fee"] = maximum, minimum

    return {
        "semantic_query": semantic_query,
        "keyword_query": keyword_query,
        "filters": filters,
        "fallback_reason": None,
    }


def find_catalog_value(query: str, values: dict) -> str | None:
    normalized_query = normalize_filter_value(query)
    for normalized_value, actual_value in sorted(
        values.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        pattern = rf"(?<!\w){re.escape(normalized_value)}(?!\w)"
        if re.search(pattern, normalized_query):
            return actual_value
    return None


def extract_price_constraints(query: str) -> tuple[float | None, float | None]:
    normalized = query.casefold().replace(",", "")
    currency = r"(?:rs\.?|inr|₹)?\s*"
    number = r"(\d+(?:\.\d+)?)"

    range_match = re.search(
        rf"\bbetween\s+{currency}{number}\s+and\s+{currency}{number}",
        normalized,
    )
    if range_match:
        first, second = float(range_match.group(1)), float(range_match.group(2))
        return min(first, second), max(first, second)

    maximum_match = re.search(
        rf"\b(?:under|below|less\s+than|up\s+to|maximum|max)\s+"
        rf"{currency}{number}",
        normalized,
    )
    minimum_match = re.search(
        rf"\b(?:over|above|more\s+than|at\s+least|minimum|min)\s+"
        rf"{currency}{number}",
        normalized,
    )
    minimum = float(minimum_match.group(1)) if minimum_match else None
    maximum = float(maximum_match.group(1)) if maximum_match else None
    return minimum, maximum


def enrich_query_plan(query: str, plan: dict, value_index: dict) -> dict:
    filters = dict(plan["filters"])
    for key in QUERY_FILTER_FIELDS:
        if filters.get(key) is None:
            filters[key] = find_catalog_value(query, value_index[key])

    if (
        filters.get("city") is not None
        and filters.get("locality") is not None
        and normalize_filter_value(filters["city"])
        == normalize_filter_value(filters["locality"])
    ):
        filters["locality"] = None

    minimum, maximum = extract_price_constraints(query)
    if filters.get("min_rental_fee") is None:
        filters["min_rental_fee"] = minimum
    if filters.get("max_rental_fee") is None:
        filters["max_rental_fee"] = maximum

    semantic_tokens = set(re.findall(r"[^\W_]+", plan["semantic_query"].casefold()))
    keyword_tokens = set(re.findall(r"[^\W_]+", plan["keyword_query"].casefold()))
    if semantic_tokens and not semantic_tokens.intersection(keyword_tokens):
        plan["keyword_query"] = plan["semantic_query"]

    plan["filters"] = filters
    return plan


def extract_query_plan(query: str, filter_catalog: dict | None = None) -> dict:
    system_prompt = (
        "You convert product-search requests into a retrieval plan. "
        "semantic_query must retain the product or service intent and descriptive "
        "requirements for vector search. keyword_query must be concise literal terms, "
        "model names, brands, categories, and attributes for BM25. Extract filters only "
        "when explicitly stated by the user. Never invent a category, location, rental "
        "duration, or price. A main category is a broad department; a subcategory is a "
        "specific listing type. Convert phrases like per day into rental_duration, "
        "under/below into max_rental_fee, and above/over into min_rental_fee. Once a "
        "location, duration, or price is extracted as a filter, remove it from "
        "semantic_query and keyword_query. Do not infer parent fields: a city does not "
        "authorize a state filter, and a subcategory does not authorize a main-category "
        "filter. For example, 'mansion in Coimbatore per day' means subcategory=Mansion, "
        "city=Coimbatore, rental_duration=Per Day, main_category=null, and state=null. "
        "Use null for every absent filter."
    )
    catalog_text = ""
    if filter_catalog:
        catalog_text = (
            "\nFor catalogued fields, use only these exact indexed values:\n"
            f"{json.dumps(filter_catalog, ensure_ascii=False)}\n"
        )
    user_prompt = (
        f"User query:\n{query}\n\n"
        f"{catalog_text}"
        "Return only JSON matching this schema:\n"
        f"{json.dumps(QUERY_PLAN_SCHEMA, separators=(',', ':'))}"
    )
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": QUERY_EXTRACT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "format": QUERY_PLAN_SCHEMA,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"temperature": QUERY_EXTRACT_TEMPERATURE},
            },
            timeout=300,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        return parse_query_plan(content, query)
    except (
        requests.RequestException,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        return default_query_plan(query, str(exc))


def require_pymysql():
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError(
            "Product lookup requires PyMySQL. Install requirements.txt first."
        ) from exc
    return pymysql


def mysql_connection(cursorclass=None):
    pymysql = require_pymysql()
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=cursorclass,
        read_timeout=300,
        write_timeout=300,
    )


def quote_mysql_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValueError("MySQL identifiers must be non-empty strings")
    return f"`{identifier.replace('`', '``')}`"


def mysql_source_name() -> str:
    return f"mysql:{MYSQL_DATABASE}.{MYSQL_TABLE}"


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError("No vector collection found. Run: python src/ingest.py") from exc


def normalize_filter_value(value) -> str:
    return " ".join(str(value).casefold().split())


def query_filter_value_index(bm25_index: PersistentBM25Index) -> dict:
    stored_values = bm25_index.filter_value_index()
    return {
        query_key: stored_values[metadata_key]
        for query_key, metadata_key in QUERY_FILTER_FIELDS.items()
    }


def build_query_filter_catalog(value_index: dict, max_values: int = 100) -> dict:
    catalog = {}
    for key in ("main_category", "state", "rental_duration"):
        values = sorted(
            value_index[key].values(),
            key=lambda value: str(value).casefold(),
        )
        if values and len(values) <= max_values:
            catalog[key] = values
    return catalog


def resolve_query_filters(filters: dict, value_index: dict) -> tuple[dict, dict]:
    resolved = {"categorical": {}}
    unresolved = {}

    for query_key, metadata_key in QUERY_FILTER_FIELDS.items():
        requested = filters.get(query_key)
        if requested is None:
            continue
        actual = value_index[query_key].get(normalize_filter_value(requested))
        if actual is None:
            unresolved[query_key] = requested
            continue
        resolved["categorical"][metadata_key] = actual

    for key in ("min_rental_fee", "max_rental_fee"):
        value = filters.get(key)
        if value is not None:
            resolved[key] = value

    return resolved, unresolved


def metadata_matches_filters(metadata, source_name, resolved_filters) -> bool:
    if metadata.get("source_file") != source_name:
        return False
    for key, expected in resolved_filters["categorical"].items():
        if metadata.get(key) != expected:
            return False

    minimum = resolved_filters.get("min_rental_fee")
    maximum = resolved_filters.get("max_rental_fee")
    if minimum is None and maximum is None:
        return True
    try:
        rental_fee = float(metadata.get("rental_fee"))
    except (TypeError, ValueError):
        return False
    if minimum is not None and rental_fee < minimum:
        return False
    if maximum is not None and rental_fee > maximum:
        return False
    return True


def vector_search(
    query,
    collection,
    top_k=15,
    candidate_k=100,
    source_name=None,
    resolved_filters=None,
):
    if collection.count() <= 0:
        return []

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(candidate_k, top_k), collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []

    for doc_id, text, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if (
            source_name is not None
            and resolved_filters is not None
            and not metadata_matches_filters(
                meta,
                source_name,
                resolved_filters,
            )
        ):
            continue
        output.append({
            "id": doc_id,
            "text": text,
            "metadata": meta,
            "score": float(dist),
            "source": "vector"
        })
        if len(output) >= top_k:
            break

    return output


def bm25_search(query, index, collection, resolved_filters, top_k=15):
    ranked = index.search(query, resolved_filters, top_k)
    if not ranked:
        return []

    scores = {item["doc_id"]: item["score"] for item in ranked}
    ordered_ids = [item["doc_id"] for item in ranked]
    data = collection.get(
        ids=ordered_ids,
        include=["documents", "metadatas"],
    )
    documents = {
        doc_id: {"text": text, "metadata": metadata}
        for doc_id, text, metadata in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
        )
    }
    return [
        {
            "id": doc_id,
            "text": documents[doc_id]["text"],
            "metadata": documents[doc_id]["metadata"],
            "score": scores[doc_id],
            "source": "bm25",
        }
        for doc_id in ordered_ids
        if doc_id in documents
    ]


def merge_results(vector_results, bm25_results):
    merged = {}

    for item in vector_results + bm25_results:
        if item["id"] not in merged:
            merged[item["id"]] = item
        else:
            merged[item["id"]]["source"] += "+" + item["source"]

    return list(merged.values())


class BGEReranker:
    def __init__(
        self,
        model_name: str,
        use_fp16: bool = False,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "BAAI reranking requires torch and transformers. "
                "Install requirements.txt first."
            ) from exc

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if use_fp16 and self.device.type != "cpu":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_score(self, pairs, batch_size=None, max_length=None):
        if not pairs:
            return []
        batch_size = batch_size or self.batch_size
        max_length = max_length or self.max_length
        scores = []

        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self.tokenizer(
                    [pair[0] for pair in batch],
                    [pair[1] for pair in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                logits = self.model(**encoded).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())

        return scores


def load_reranker():
    try:
        return BGEReranker(
            RERANK_MODEL,
            use_fp16=RERANK_USE_FP16,
            batch_size=RERANK_BATCH_SIZE,
            max_length=RERANK_MAX_LENGTH,
        )
    except ImportError as exc:
        raise RuntimeError(
            "BAAI reranking dependencies are missing. Install requirements.txt first."
        ) from exc


def rerank(query, candidates, ranker, top_k=6):
    if not candidates:
        return []

    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = ranker.compute_score(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
    )
    if isinstance(scores, (int, float)):
        scores = [scores]

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [
        {
            "id": candidate["id"],
            "text": candidate["text"],
            "metadata": candidate["metadata"],
            "score": float(score),
        }
        for candidate, score in ranked[:top_k]
    ]


def extract_product_ids(reranked):
    product_ids = []
    seen = set()

    for result in reranked:
        metadata = result.get("metadata") or {}
        if metadata.get("source_type") != "mysql":
            continue
        if metadata.get("source_table") != MYSQL_TABLE:
            continue

        product_id = metadata.get(MYSQL_SEARCH_ID_COLUMN)
        if (
            product_id is None
            and metadata.get("primary_key_column") == MYSQL_SEARCH_ID_COLUMN
        ):
            product_id = metadata.get("primary_key_value")
        if product_id is None:
            continue

        identity = str(product_id)
        if identity in seen:
            continue
        seen.add(identity)
        product_ids.append(product_id)

    return product_ids


def fetch_products_by_ids(product_ids, connection=None):
    unique_ids = list(dict.fromkeys(product_ids))
    if not unique_ids:
        return []

    owns_connection = connection is None
    if owns_connection:
        pymysql = require_pymysql()
        connection = mysql_connection(cursorclass=pymysql.cursors.DictCursor)

    placeholders = ", ".join(["%s"] * len(unique_ids))
    query = (
        f"SELECT * FROM {quote_mysql_identifier(MYSQL_RESULT_TABLE)} "
        f"WHERE {quote_mysql_identifier(MYSQL_RESULT_ID_COLUMN)} "
        f"IN ({placeholders})"
    )

    try:
        with connection.cursor() as cursor:
            cursor.execute(query, unique_ids)
            rows = cursor.fetchall()
    finally:
        if owns_connection:
            connection.close()

    rows_by_id = {
        str(row[MYSQL_RESULT_ID_COLUMN]): row
        for row in rows
        if row.get(MYSQL_RESULT_ID_COLUMN) is not None
    }
    return [
        rows_by_id[str(product_id)]
        for product_id in unique_ids
        if str(product_id) in rows_by_id
    ]


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

            print(
                "Query plan: "
                f"{json.dumps({key: value for key, value in query_plan.items() if key != 'fallback_reason'}, ensure_ascii=False)}"
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
            if not merged:
                print(" done.\n\nNo matching products found.\n")
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
