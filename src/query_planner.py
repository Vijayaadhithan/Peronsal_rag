import json
import re

from bm25_index import PersistentBM25Index
from ollama_client import structured_chat
from settings import QUERY_EXTRACT_MODEL, QUERY_EXTRACT_TEMPERATURE

QUERY_FILTER_FIELDS = {
    "main_category": "main_category_name",
    "subcategory": "subcategory_name",
    "state": "state_name",
    "city": "city_name",
    "locality": "locality_name",
    "rental_duration": "rental_duration",
}
QUERY_FILTER_KEYS = (*QUERY_FILTER_FIELDS, "min_rental_fee", "max_rental_fee")
OFFER_AD_TYPE = "1"
WANTED_AD_TYPE = "2"
DURATION_PATTERNS = (
    (
        "Per Hour",
        r"\b(?:hourly|per\s+hour|by\s+the\s+hour|"
        r"for\s+(?:(?:an|one|1)\s+)?hour)\b",
    ),
    (
        "Per Day",
        r"\b(?:daily|per\s+day|by\s+the\s+day|"
        r"for\s+(?:(?:a|one|1)\s+)?day)\b",
    ),
    (
        "Per Week",
        r"\b(?:weekly|per\s+week|by\s+the\s+week|"
        r"for\s+(?:(?:a|one|1)\s+)?week)\b",
    ),
    (
        "Per Month",
        r"\b(?:monthly|per\s+month|by\s+the\s+month|"
        r"for\s+(?:(?:a|one|1)\s+)?month)\b",
    ),
    (
        "Per Ride",
        r"\b(?:per\s+ride|by\s+the\s+ride|for\s+(?:a|one|1)\s+ride)\b",
    ),
)
QUERY_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "semantic_query": {"type": "string"},
        "keyword_query": {"type": "string"},
        "target_ad_type": {
            "type": "string",
            "enum": ["offer", "wanted"],
            "description": (
                "Use offer when the searcher wants to rent, buy, or hire something. "
                "Use wanted only when they explicitly ask to find request/wanted ads "
                "posted by other people."
            ),
        },
        "filters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "main_category": {
                    "type": ["string", "null"],
                    "description": (
                        "Explicit broad department, such as Accommodation & Spaces "
                        "or Automobiles; never put a specific product type here."
                    ),
                },
                "subcategory": {
                    "type": ["string", "null"],
                    "description": (
                        "Explicit specific indexed listing type, such as Mansion, Car, "
                        "Bike, or Laptop."
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
                        "Explicit rental period. Allowed values are Per Hour, Per Day, "
                        "Per Week, Per Month, and Per Ride."
                    ),
                },
                "min_rental_fee": {"type": ["number", "null"]},
                "max_rental_fee": {"type": ["number", "null"]},
            },
            "required": list(QUERY_FILTER_KEYS),
        },
    },
    "required": ["semantic_query", "keyword_query", "target_ad_type", "filters"],
}


def normalize_filter_value(value) -> str:
    return " ".join(str(value).casefold().split())


def default_query_plan(query: str, fallback_reason: str | None = None) -> dict:
    return {
        "semantic_query": query,
        "keyword_query": query,
        "target_ad_type": "offer",
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
    target_ad_type = parsed.get("target_ad_type")
    if target_ad_type not in {"offer", "wanted"}:
        target_ad_type = "offer"
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
        "target_ad_type": target_ad_type,
        "filters": filters,
        "fallback_reason": None,
    }


def find_catalog_value(
    query: str,
    values: dict,
    allow_plural: bool = False,
) -> str | None:
    normalized_query = normalize_filter_value(query)
    for normalized_value, actual_value in sorted(
        values.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        escaped_value = re.escape(normalized_value)
        if allow_plural and re.fullmatch(r"[a-z0-9_-]+", normalized_value):
            if re.search(r"[^aeiou]y$", normalized_value):
                escaped_value = rf"{re.escape(normalized_value[:-1])}(?:y|ies)"
            elif normalized_value.endswith(("s", "x", "z", "ch", "sh")):
                escaped_value = rf"{escaped_value}(?:es)?"
            else:
                escaped_value = rf"{escaped_value}s?"
        pattern = rf"(?<!\w){escaped_value}(?!\w)"
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
    if not range_match:
        range_match = re.search(
            rf"\bfrom\s+{currency}{number}\s+to\s+{currency}{number}",
            normalized,
        )
    if range_match:
        first, second = float(range_match.group(1)), float(range_match.group(2))
        return min(first, second), max(first, second)

    maximum_match = re.search(
        rf"\b(?:under|below|less\s+than|not\s+more\s+than|up\s+to|"
        rf"within|withing|budget(?:\s+of)?|maximum|max)\s+"
        rf"{currency}{number}",
        normalized,
    )
    if not maximum_match:
        maximum_match = re.search(
            rf"\b(?:(?:in|around)\s+(?:the\s+)?)?"
            rf"{currency}{number}\s+(?:price\s+)?range\b",
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


def extract_duration_filter(query: str, values: dict) -> str | None:
    normalized_query = normalize_filter_value(query)
    for canonical_value, pattern in DURATION_PATTERNS:
        if re.search(pattern, normalized_query):
            return values.get(normalize_filter_value(canonical_value)) or canonical_value
    return None


def infer_target_ad_type(query: str) -> str:
    normalized = normalize_filter_value(query)
    wanted_patterns = (
        r"\b(?:wanted|request|requirement)\s+ads?\b",
        r"\bads?\s+(?:from|by)\s+people\s+(?:who\s+)?"
        r"(?:need|want|require)\b",
        r"\b(?:people|persons?|someone|somebody|anyone|buyers|renters|customers)"
        r"\s+(?:who\s+)?(?:(?:need|want|require)s?|"
        r"(?:is|are)\s+looking\s+for|looking\s+for)\b",
        r"\blooking\s+for\s+(?:people|buyers|renters|customers)\b",
        r"\bshow\s+me\s+(?:requests|requirements)\b",
    )
    return (
        "wanted"
        if any(re.search(pattern, normalized) for pattern in wanted_patterns)
        else "offer"
    )


def enrich_query_plan(query: str, plan: dict, value_index: dict) -> dict:
    filters = dict(plan["filters"])
    for key in QUERY_FILTER_FIELDS:
        if key == "rental_duration":
            continue
        exact_value = find_catalog_value(
            query,
            value_index[key],
            allow_plural=key in {"main_category", "subcategory"},
        )
        if exact_value is not None:
            filters[key] = exact_value

    filters["rental_duration"] = extract_duration_filter(
        query,
        value_index["rental_duration"],
    )
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
    plan["target_ad_type"] = infer_target_ad_type(query)
    return plan


def extract_query_plan(query: str, filter_catalog: dict | None = None) -> dict:
    system_prompt = (
        "You convert product-search requests into a retrieval plan. "
        "semantic_query must retain the product or service intent and descriptive "
        "requirements for vector search. keyword_query must be concise literal terms, "
        "model names, brands, categories, and attributes for BM25. Extract filters only "
        "when explicitly stated by the user. Never invent a category, location, rental "
        "duration, or price. A main category is a broad department; a subcategory is a "
        "specific listing type. Map hourly/per hour to Per Hour, daily/for a day to "
        "Per Day, weekly/for a week to Per Week, monthly/for a month to Per Month, "
        "and per ride to Per Ride. Convert under/below/within into max_rental_fee and "
        "above/over into min_rental_fee. Once a location, duration, or price is "
        "extracted as a filter, remove it from semantic_query and keyword_query. "
        "Do not infer parent fields: a city does not authorize a state filter, and a "
        "subcategory does not authorize a main-category filter. For example, "
        "'mansion in Coimbatore per day' means subcategory=Mansion, city=Coimbatore, "
        "rental_duration=Per Day, main_category=null, and state=null. Interpret the "
        "request from the searcher's perspective. 'I need a bike', 'find me a car', "
        "and 'looking for a laptop' all target offer ads because the searcher wants an "
        "available item. 'Someone looking for bikes', 'people who need a car', and "
        "'find renters looking for a laptop' target wanted ads because the user is "
        "searching for another person's request. Use target_ad_type=wanted only when "
        "the user explicitly asks for wanted/request ads or for people who need an "
        "item. Use null for every absent filter."
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
        content = structured_chat(
            QUERY_EXTRACT_MODEL,
            system_prompt,
            user_prompt,
            QUERY_PLAN_SCHEMA,
            QUERY_EXTRACT_TEMPERATURE,
        )
        return parse_query_plan(content, query)
    except (RuntimeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return default_query_plan(query, str(exc))


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
