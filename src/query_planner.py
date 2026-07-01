import json
import re
from difflib import SequenceMatcher, get_close_matches

from bm25_index import PersistentBM25Index
from gemini_client import structured_chat
from settings import (
    QUERY_EXTRACT_MODEL,
    QUERY_EXTRACT_TEMPERATURE,
    QUERY_FUZZY_MATCHING,
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
QUERY_FILTER_ALIASES = {
    "state": {
        "orissa": "odisha",
    },
    "city": {
        "bangalore": "bengaluru",
        "bangaluru": "bengaluru",
        "bombay": "mumbai",
        "calcutta": "kolkata",
        "cochin": "kochi",
        "madras": "chennai",
        "mysore": "mysuru",
        "poona": "pune",
        "trivandrum": "thiruvananthapuram",
        "baroda": "vadodara",
        "prayagraj": "allahabad",
        "mangaluru": "mangalore",
        "gurugram": "gurgaon",
    },
}
FUZZY_MATCH_THRESHOLDS = {
    "main_category": 0.90,
    "subcategory": 0.90,
    "state": 0.90,
    "city": 0.88,
    "locality": 0.92,
}
LOCATION_PREPOSITIONS = {"in", "near", "at", "around"}
LOCATION_STOP_WORDS = {
    "for",
    "per",
    "under",
    "below",
    "above",
    "over",
    "within",
    "between",
    "with",
    "by",
    "hourly",
    "daily",
    "weekly",
    "monthly",
}
GENERIC_CATEGORY_HINT_TOKENS = {
    "equipment",
    "hire",
    "item",
    "off",
    "product",
    "rent",
    "rental",
    "service",
    "thing",
    "vehicle",
}
FAST_PATH_FILLER_TOKENS = {
    "a",
    "ad",
    "ads",
    "an",
    "any",
    "available",
    "by",
    "find",
    "for",
    "from",
    "get",
    "give",
    "hire",
    "i",
    "in",
    "item",
    "items",
    "looking",
    "me",
    "my",
    "near",
    "need",
    "of",
    "product",
    "products",
    "rent",
    "rental",
    "rentals",
    "required",
    "require",
    "search",
    "searching",
    "show",
    "some",
    "the",
    "to",
    "want",
}
FAST_PATH_WANTED_TOKENS = {
    "customers",
    "people",
    "person",
    "persons",
    "renters",
    "request",
    "someone",
    "wanted",
    "who",
}
FAST_PATH_PRICE_TOKENS = {
    "above",
    "and",
    "below",
    "between",
    "budget",
    "inr",
    "less",
    "max",
    "maximum",
    "min",
    "minimum",
    "more",
    "not",
    "over",
    "price",
    "range",
    "rs",
    "than",
    "under",
    "up",
    "withing",
    "within",
}
FAST_PATH_DURATION_TOKENS = {
    "1",
    "a",
    "an",
    "daily",
    "day",
    "hour",
    "hourly",
    "month",
    "monthly",
    "one",
    "per",
    "ride",
    "week",
    "weekly",
}
FAST_PATH_SORT_TOKENS = {
    "asc",
    "ascending",
    "affordable",
    "budget",
    "cheap",
    "cheapest",
    "cost",
    "costs",
    "desc",
    "descending",
    "expensive",
    "fee",
    "fees",
    "first",
    "friendly",
    "high",
    "highest",
    "least",
    "low",
    "lowest",
    "most",
    "order",
    "ordered",
    "price",
    "prices",
    "rate",
    "rates",
    "sort",
    "sorted",
}
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
        "sort_order": None,
        "filters": {key: None for key in QUERY_FILTER_KEYS},
        "inferred_categories": {
            "main_category": None,
            "subcategory": None,
        },
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


def category_term_pattern(value: str) -> str:
    normalized = normalize_filter_value(value)
    escaped = re.escape(normalized)
    if re.fullmatch(r"[a-z0-9_-]+", normalized):
        if re.search(r"[^aeiou]y$", normalized):
            return rf"{re.escape(normalized[:-1])}(?:y|ies)"
        if normalized.endswith(("s", "x", "z", "ch", "sh")):
            return rf"{escaped}(?:es)?"
        return rf"{escaped}s?"
    return escaped


def is_explicit_category_request(query: str, value: str) -> bool:
    normalized_query = normalize_filter_value(query)
    term = category_term_pattern(value)
    article = r"(?:a|an|the|some)?\s*"
    request = (
        r"(?:need|want|require|rent|hire|find|show\s+me|"
        r"looking\s+for|searching\s+for)"
    )
    patterns = (
        rf"^{article}{term}(?!\w)",
        rf"\b{request}\s+{article}{term}(?!\w)",
        rf"\b(?:wanted|request)\s+ads?\s+for\s+{article}{term}(?!\w)",
        rf"\b(?:rental|hire)\s+{term}(?!\w)",
        rf"(?<!\w){term}\s+(?:rent|hire|for\s+(?:rent|hire)|rental|"
        rf"in|near|at|under|below|within|between|per\s+hour|"
        rf"per\s+day|per\s+week|per\s+month)\b",
        rf"\b\d+(?:\.\d+)?\s+{term}(?!\w)",
        rf"(?<!\w){term}\s+\d+(?:\.\d+)?\b",
    )
    return any(re.search(pattern, normalized_query) for pattern in patterns)


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
    inferred_categories = {
        "main_category": None,
        "subcategory": None,
    }
    for category_key in inferred_categories:
        value = filters[category_key]
        if value is not None and not is_explicit_category_request(
            original_query,
            value,
        ):
            inferred_categories[category_key] = value
            filters[category_key] = None
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
        "inferred_categories": inferred_categories,
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


def canonical_catalog_value(
    query_key: str,
    requested_value: str,
    values: dict,
    allow_fuzzy: bool = True,
) -> str | None:
    normalized = normalize_filter_value(requested_value)
    normalized = QUERY_FILTER_ALIASES.get(query_key, {}).get(
        normalized,
        normalized,
    )
    exact = values.get(normalized)
    if exact is not None or not allow_fuzzy:
        return exact

    threshold = FUZZY_MATCH_THRESHOLDS.get(query_key)
    if threshold is None or len(normalized) < 4:
        return None
    match = fuzzy_catalog_match(normalized, values, threshold)
    return match[0] if match is not None else None


def fuzzy_catalog_match(
    normalized: str,
    values: dict,
    threshold: float,
) -> tuple[str, float] | None:
    matches = get_close_matches(
        normalized,
        values.keys(),
        n=2,
        cutoff=threshold,
    )
    if not matches:
        return None
    first_score = SequenceMatcher(None, normalized, matches[0]).ratio()
    if len(matches) > 1:
        second_score = SequenceMatcher(None, normalized, matches[1]).ratio()
        if first_score - second_score < 0.04:
            return None
    return values[matches[0]], first_score


def edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance for short catalog terms."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def ordered_subsequence(shorter: str, longer: str) -> bool:
    iterator = iter(longer)
    return all(character in iterator for character in shorter)


def typo_catalog_match(
    normalized: str,
    values: dict,
    single_token_only: bool = False,
) -> tuple[str, float] | None:
    """Match a likely typo while rejecting close or ambiguous catalog values."""
    normalized = normalize_filter_value(normalized)
    if len(normalized) < 3:
        return None

    candidates: dict[str, tuple[str, float]] = {}
    variants = [normalized]
    if normalized.endswith("s") and len(normalized) > 3:
        variants.append(normalized[:-1])

    for candidate, actual in values.items():
        if single_token_only and " " in candidate:
            continue
        best_score = 0.0
        for variant in variants:
            length = max(len(variant), len(candidate))
            if abs(len(variant) - len(candidate)) > 3:
                continue
            distance = edit_distance(variant, candidate)
            max_edits = 1 if length <= 5 else 2
            if distance <= max_edits:
                best_score = max(
                    best_score,
                    0.98 - (0.02 * distance),
                )
            if (
                len(variant) >= 3
                and len(candidate) - len(variant) in range(1, 4)
                and variant[:2] == candidate[:2]
                and variant[-1] == candidate[-1]
                and ordered_subsequence(variant, candidate)
            ):
                best_score = max(best_score, 0.92)
        if best_score:
            identity = normalize_filter_value(actual)
            current = candidates.get(identity)
            if current is None or best_score > current[1]:
                candidates[identity] = (actual, best_score)

    ranked = sorted(
        candidates.values(),
        key=lambda item: (item[1], -len(str(item[0]))),
        reverse=True,
    )
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < 0.04:
        return None
    return ranked[0]


def location_phrases(query: str) -> list[str]:
    tokens = re.findall(r"[^\W_]+", normalize_filter_value(query))
    phrases = []
    for index, token in enumerate(tokens):
        if token not in LOCATION_PREPOSITIONS:
            continue
        location_tokens = []
        for candidate in tokens[index + 1 : index + 5]:
            if candidate in LOCATION_STOP_WORDS:
                break
            location_tokens.append(candidate)
        if not location_tokens or location_tokens[0].isdigit():
            continue
        phrases.extend(
            " ".join(location_tokens[:length])
            for length in range(len(location_tokens), 0, -1)
        )
    return list(dict.fromkeys(phrases))


def find_fuzzy_location(query: str, value_index: dict) -> tuple[str, str] | None:
    candidates = []
    key_priority = {"city": 3, "state": 2, "locality": 1}
    for phrase in location_phrases(query):
        for key in ("city", "state", "locality"):
            threshold = FUZZY_MATCH_THRESHOLDS[key]
            match = fuzzy_catalog_match(
                phrase,
                value_index[key],
                threshold,
            )
            if match is None and QUERY_FUZZY_MATCHING:
                match = typo_catalog_match(phrase, value_index[key])
            if match is not None:
                actual, score = match
                candidates.append((score, key_priority[key], key, actual))
    if not candidates:
        return None

    unique_candidates = {}
    for candidate in candidates:
        identity = normalize_filter_value(candidate[3])
        if candidate > unique_candidates.get(identity, (-1, -1, "", "")):
            unique_candidates[identity] = candidate
    candidates = list(unique_candidates.values())
    candidates.sort(reverse=True)
    if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.03:
        return None
    _, _, key, actual = candidates[0]
    return key, actual


def correct_explicit_query_typos(
    query: str,
    value_index: dict,
) -> tuple[str, list[dict[str, str]]]:
    """Correct high-confidence catalog typos before deterministic planning."""
    if not QUERY_FUZZY_MATCHING:
        return query, []

    corrected = normalize_filter_value(query)
    corrections: list[dict[str, str]] = []

    location_candidates = []
    key_priority = {"city": 3, "state": 2, "locality": 1}
    for phrase in location_phrases(corrected):
        for key in ("city", "state", "locality"):
            if canonical_catalog_value(
                key,
                phrase,
                value_index[key],
                allow_fuzzy=False,
            ) is not None:
                continue
            match = typo_catalog_match(phrase, value_index[key])
            if match is not None:
                actual, score = match
                location_candidates.append(
                    (score, key_priority[key], key, phrase, actual)
                )
    location_candidates.sort(reverse=True)
    if location_candidates and (
        len(location_candidates) == 1
        or location_candidates[0][0] - location_candidates[1][0] >= 0.04
        or normalize_filter_value(location_candidates[0][4])
        == normalize_filter_value(location_candidates[1][4])
    ):
        _, _, key, source, actual = location_candidates[0]
        corrected = re.sub(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            normalize_filter_value(actual),
            corrected,
            count=1,
        )
        corrections.append(
            {"field": key, "input": source, "value": actual}
        )

    has_exact_category = any(
        find_catalog_value(
            corrected,
            value_index[key],
            allow_plural=True,
        )
        for key in ("main_category", "subcategory")
    )
    if not has_exact_category:
        ignored_tokens = (
            FAST_PATH_FILLER_TOKENS
            | FAST_PATH_WANTED_TOKENS
            | FAST_PATH_PRICE_TOKENS
            | FAST_PATH_DURATION_TOKENS
            | LOCATION_PREPOSITIONS
            | LOCATION_STOP_WORDS
        )
        location_tokens = {
            part
            for location_key in ("city", "state", "locality")
            for value in value_index[location_key]
            for part in value.split()
        }
        category_candidates = []
        for token in re.findall(r"[^\W_]+", corrected):
            if (
                token in ignored_tokens
                or token.isdigit()
                or token in location_tokens
            ):
                continue
            for key, priority in (("subcategory", 2), ("main_category", 1)):
                match = typo_catalog_match(
                    token,
                    value_index[key],
                    single_token_only=True,
                )
                if match is not None:
                    actual, score = match
                    category_candidates.append(
                        (score, priority, key, token, actual)
                    )
        category_candidates.sort(reverse=True)
        if category_candidates and (
            len(category_candidates) == 1
            or category_candidates[0][0] - category_candidates[1][0] >= 0.04
            or normalize_filter_value(category_candidates[0][4])
            == normalize_filter_value(category_candidates[1][4])
        ):
            _, _, key, source, actual = category_candidates[0]
            corrected = re.sub(
                rf"(?<!\w){re.escape(source)}(?!\w)",
                normalize_filter_value(actual),
                corrected,
                count=1,
            )
            corrections.append(
                {"field": key, "input": source, "value": actual}
            )

    return corrected, corrections


def infer_keyword_subcategory(keyword_query: str, values: dict) -> str | None:
    query_tokens = {
        token
        for token in re.findall(r"[^\W_]+", normalize_filter_value(keyword_query))
        if len(token) >= 3 and token not in GENERIC_CATEGORY_HINT_TOKENS
    }
    if not query_tokens:
        return None

    token_categories: dict[str, set[str]] = {}
    for actual_value in values.values():
        category_tokens = set(
            re.findall(r"[^\W_]+", normalize_filter_value(actual_value))
        )
        for token in query_tokens.intersection(category_tokens):
            token_categories.setdefault(token, set()).add(actual_value)

    scores: dict[str, int] = {}
    for categories in token_categories.values():
        if len(categories) != 1:
            continue
        category = next(iter(categories))
        scores[category] = scores.get(category, 0) + 1
    if not scores:
        return None

    ranked = sorted(
        scores.items(),
        key=lambda item: (item[1], len(item[0])),
        reverse=True,
    )
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def find_catalog_alias(query: str, query_key: str, values: dict) -> str | None:
    normalized_query = normalize_filter_value(query)
    for alias, canonical in QUERY_FILTER_ALIASES.get(query_key, {}).items():
        pattern = rf"(?<!\w){re.escape(alias)}(?!\w)"
        if re.search(pattern, normalized_query):
            return values.get(canonical)
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


def extract_sort_order(query: str) -> str | None:
    """Extract explicit price ordering without treating it as relevance text."""
    normalized = normalize_filter_value(query)
    price_term = r"(?:rental\s+)?(?:price|prices|rate|rates|rent|fee|fees)"
    low_to_high = (
        rf"(?:{price_term}\s+(?:from\s+)?low(?:est)?\s+to\s+high(?:est)?|"
        rf"low(?:est)?\s+to\s+high(?:est)?\s+{price_term})"
    )
    high_to_low = (
        rf"(?:{price_term}\s+(?:from\s+)?high(?:est)?\s+to\s+low(?:est)?|"
        rf"high(?:est)?\s+to\s+low(?:est)?\s+{price_term})"
    )
    if re.search(rf"\b{low_to_high}\b", normalized):
        return "price_asc"
    if re.search(rf"\b{high_to_low}\b", normalized):
        return "price_desc"
    ascending_patterns = (
        r"\b(?:cheapest|lowest(?:[\s-]+priced)?|least[\s-]+expensive)\b",
        r"\b(?:cheap|affordable|budget[\s-]+friendly)\b",
        rf"\b(?:low|lowest|minimum|min)\s+{price_term}\b",
        r"\blow(?:est)?\s+(?:cost|costs)\b",
        rf"\b{price_term}\s+(?:from\s+)?low(?:est)?\s+to\s+high(?:est)?\b",
        rf"\blow(?:est)?\s+to\s+high(?:est)?\s+{price_term}\b",
        rf"\b(?:sort|order|sorted|ordered)(?:\s+by)?\s+{price_term}"
        rf"\s+(?:asc|ascending|low(?:est)?\s+to\s+high(?:est)?)\b",
        rf"\b(?:asc|ascending)\s+{price_term}\b",
        rf"\b{price_term}\s+(?:asc|ascending)\b",
    )
    descending_patterns = (
        r"\b(?:most[\s-]+expensive|highest(?:[\s-]+priced)?)\b",
        rf"\b(?:high|highest|maximum|max)\s+{price_term}\b",
        rf"\b{price_term}\s+(?:from\s+)?high(?:est)?\s+to\s+low(?:est)?\b",
        rf"\bhigh(?:est)?\s+to\s+low(?:est)?\s+{price_term}\b",
        rf"\b(?:sort|order|sorted|ordered)(?:\s+by)?\s+{price_term}"
        rf"\s+(?:desc|descending|high(?:est)?\s+to\s+low(?:est)?)\b",
        rf"\b(?:desc|descending)\s+{price_term}\b",
        rf"\b{price_term}\s+(?:desc|descending)\b",
    )
    if any(re.search(pattern, normalized) for pattern in ascending_patterns):
        return "price_asc"
    if any(re.search(pattern, normalized) for pattern in descending_patterns):
        return "price_desc"
    return None


def extract_standalone_budget(query: str) -> float | None:
    """Infer one bare amount only for the conservative direct-filter path."""
    normalized = query.casefold().replace(",", "")
    if re.search(
        r"\b(?:cc|bhp|hp|km|kms|kilometers?|model|seater|year)\b",
        normalized,
    ):
        return None
    amounts = re.findall(
        r"(?<![\w.])(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)(?![\w.])",
        normalized,
    )
    if len(amounts) != 1:
        return None
    amount = float(amounts[0])
    if amount < 10 or 1900 <= amount <= 2100:
        return None
    return amount


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


def expand_functional_keyword_query(query: str, keyword_query: str) -> str:
    normalized = normalize_filter_value(query)
    rough_terrain = (
        re.search(r"\brough\s+terrain\b", normalized)
        or re.search(r"\boff[\s-]?road\b", normalized)
    )
    vehicle_context = re.search(
        r"\b(?:vehicle|driv(?:e|ing)|recreational)\b",
        normalized,
    )
    if rough_terrain and vehicle_context:
        concepts = "off-road vehicle ATV 4x4"
        if "atv" not in normalize_filter_value(keyword_query):
            return f"{keyword_query} {concepts}".strip()
    return keyword_query


def enrich_query_plan(query: str, plan: dict, value_index: dict) -> dict:
    plan["keyword_query"] = expand_functional_keyword_query(
        query,
        plan["keyword_query"],
    )
    filters = dict(plan["filters"])
    inferred_categories = dict(
        plan.get(
            "inferred_categories",
            {"main_category": None, "subcategory": None},
        )
    )
    for key in QUERY_FILTER_FIELDS:
        if key == "rental_duration":
            continue
        exact_value = find_catalog_value(
            query,
            value_index[key],
            allow_plural=key in {"main_category", "subcategory"},
        )
        if exact_value is None:
            exact_value = find_catalog_alias(query, key, value_index[key])
        category_is_explicit = (
            key not in inferred_categories
            or exact_value is None
            or is_explicit_category_request(query, exact_value)
        )
        if exact_value is not None and not category_is_explicit:
            filters[key] = None
            if len(normalize_filter_value(exact_value).split()) == 1:
                inferred_categories["main_category"] = None
                inferred_categories["subcategory"] = None
            else:
                inferred_categories[key] = exact_value
            continue
        if exact_value is not None:
            filters[key] = exact_value
            if key in inferred_categories:
                inferred_categories[key] = None
        elif filters.get(key) is not None:
            canonical_value = canonical_catalog_value(
                key,
                filters[key],
                value_index[key],
            )
            value_was_stated = text_mentions_filter(query, filters[key])
            if key in inferred_categories:
                inferred_categories[key] = canonical_value or filters[key]
                filters[key] = None
            elif canonical_value is not None and value_was_stated:
                filters[key] = canonical_value
            else:
                filters[key] = None

    if not any(filters.get(key) for key in ("state", "city", "locality")):
        fuzzy_location = find_fuzzy_location(query, value_index)
        if fuzzy_location is not None:
            key, actual = fuzzy_location
            filters[key] = actual

    for key, requested in tuple(inferred_categories.items()):
        if requested is None:
            continue
        inferred_categories[key] = (
            canonical_catalog_value(key, requested, value_index[key])
            or requested
        )

    if (
        filters.get("subcategory") is None
        and inferred_categories.get("subcategory") is None
    ):
        inferred_categories["subcategory"] = infer_keyword_subcategory(
            plan["keyword_query"],
            value_index["subcategory"],
        )

    filters["rental_duration"] = extract_duration_filter(
        query,
        value_index["rental_duration"],
    )
    if filters.get("subcategory") is not None:
        parent = value_index.get("_subcategory_main_category", {}).get(
            normalize_filter_value(filters["subcategory"])
        )
        if parent is not None:
            filters["main_category"] = parent
            inferred_categories["main_category"] = None
    elif inferred_categories.get("subcategory") is not None:
        parent = value_index.get("_subcategory_main_category", {}).get(
            normalize_filter_value(inferred_categories["subcategory"])
        )
        if parent is not None:
            inferred_categories["main_category"] = parent
    if (
        filters.get("city") is not None
        and filters.get("locality") is not None
    ):
        normalized_city = normalize_filter_value(filters["city"])
        normalized_locality = normalize_filter_value(filters["locality"])
        locality_as_city = QUERY_FILTER_ALIASES.get("city", {}).get(
            normalized_locality,
            normalized_locality,
        )
        if normalized_city == locality_as_city:
            filters["locality"] = None

    locality = filters.get("locality")
    if locality is not None:
        location = value_index.get("_locality_location", {}).get(
            normalize_filter_value(locality)
        )
        if location is not None:
            filters["city"] = filters.get("city") or location["city"]
            filters["state"] = filters.get("state") or location["state"]

    city = filters.get("city")
    if city is not None and filters.get("state") is None:
        state = value_index.get("_city_state", {}).get(
            normalize_filter_value(city)
        )
        if state is not None:
            filters["state"] = state

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
    plan["inferred_categories"] = inferred_categories
    plan["target_ad_type"] = infer_target_ad_type(query)
    plan["sort_order"] = extract_sort_order(query)
    return plan


def deterministic_filter_query_plan(
    query: str,
    value_index: dict,
) -> dict | None:
    """Return a direct-filter plan for simple explicit catalog queries."""
    sort_order = extract_sort_order(query)
    corrected_query, corrections = correct_explicit_query_typos(
        query,
        value_index,
    )
    # In a rental catalog, "car retail" alongside explicit price ordering is
    # overwhelmingly a typo for "car rental". Keep the correction narrow so a
    # genuine retail query without ordering still goes through semantic search.
    if sort_order and re.search(r"\bretail\b", corrected_query.casefold()):
        corrected_query = re.sub(
            r"\bretail\b",
            "rental",
            corrected_query,
            flags=re.IGNORECASE,
        )
        corrections.append(
            {"field": "intent", "input": "retail", "value": "rental"}
        )
    plan = enrich_query_plan(
        corrected_query,
        default_query_plan(corrected_query),
        value_index,
    )
    filters = plan["filters"]
    if not any(
        filters.get(key)
        for key in ("main_category", "subcategory")
    ):
        return None
    if (
        filters.get("min_rental_fee") is None
        and filters.get("max_rental_fee") is None
    ):
        filters["max_rental_fee"] = extract_standalone_budget(
            corrected_query
        )

    residual = normalize_filter_value(corrected_query)
    for key in QUERY_FILTER_FIELDS:
        value = filters.get(key)
        if not value:
            continue
        residual = re.sub(
            rf"(?<!\w){category_term_pattern(value)}(?!\w)",
            " ",
            residual,
        )
        normalized_value = normalize_filter_value(value)
        for alias, canonical in QUERY_FILTER_ALIASES.get(key, {}).items():
            if canonical == normalized_value:
                residual = re.sub(
                    rf"(?<!\w){re.escape(alias)}(?!\w)",
                    " ",
                    residual,
                )

    has_price = any(
        filters.get(key) is not None
        for key in ("min_rental_fee", "max_rental_fee")
    )
    has_duration = filters.get("rental_duration") is not None
    allowed_tokens = set(FAST_PATH_FILLER_TOKENS)
    if plan["target_ad_type"] == "wanted":
        allowed_tokens.update(FAST_PATH_WANTED_TOKENS)
    if has_price:
        allowed_tokens.update(FAST_PATH_PRICE_TOKENS)
    if has_duration:
        allowed_tokens.update(FAST_PATH_DURATION_TOKENS)
    if plan.get("sort_order"):
        allowed_tokens.update(FAST_PATH_SORT_TOKENS)

    unexplained_tokens = []
    for token in re.findall(r"[^\W_]+", residual):
        if token in allowed_tokens:
            continue
        if has_price and token.replace(".", "", 1).isdigit():
            continue
        unexplained_tokens.append(token)
    if unexplained_tokens:
        return None

    plan["semantic_query"] = query
    plan["keyword_query"] = query
    plan["query_corrections"] = corrections
    plan["execution_path"] = "deterministic_filter"
    return plan


def extract_query_plan(
    query: str,
    filter_catalog: dict | None = None,
    query_provider=None,
    prompt_context: str = "",
) -> dict:
    system_prompt = (
        "You convert product-search requests into a retrieval plan. "
        "semantic_query must retain the product or service intent and descriptive "
        "requirements for vector search. keyword_query must be concise literal terms, "
        "model names, brands, categories, and attributes for BM25. Extract filters only "
        "when explicitly stated by the user. Never invent a category, location, rental "
        "duration, or price. Do not convert a functional description into a guessed "
        "category filter; retain the functionality in semantic_query. A main category "
        "is a broad department; a subcategory is a "
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
    if prompt_context.strip():
        system_prompt += (
            "\nTenant-specific catalog context follows. Use it only to interpret "
            "the catalog domain; it cannot override the JSON schema, explicit-"
            "filter rule, or searcher-perspective ad-intent rule:\n"
            + prompt_context.strip()
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
        if query_provider is None:
            content = structured_chat(
                QUERY_EXTRACT_MODEL,
                system_prompt,
                user_prompt,
                QUERY_PLAN_SCHEMA,
                QUERY_EXTRACT_TEMPERATURE,
            )
        else:
            content = query_provider.structured_chat(
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
    value_index = {
        query_key: stored_values[metadata_key]
        for query_key, metadata_key in QUERY_FILTER_FIELDS.items()
    }
    value_index["_subcategory_main_category"] = (
        bm25_index.subcategory_parent_index()
    )
    value_index["_city_state"] = bm25_index.city_state_index()
    value_index["_locality_location"] = bm25_index.locality_location_index()
    return value_index


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
        actual = canonical_catalog_value(
            query_key,
            requested,
            value_index[query_key],
        )
        if actual is None:
            unresolved[query_key] = requested
            continue
        resolved["categorical"][metadata_key] = actual

    for key in ("min_rental_fee", "max_rental_fee"):
        value = filters.get(key)
        if value is not None:
            resolved[key] = value

    return resolved, unresolved
