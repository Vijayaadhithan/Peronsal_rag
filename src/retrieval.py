import chromadb

from ollama_client import embed_text
from mysql_store import fetch_product_types_by_ids
from query_planner import OFFER_AD_TYPE, WANTED_AD_TYPE
from settings import (
    CHROMA_DIR,
    COLLECTION_NAME,
    MYSQL_SEARCH_ID_COLUMN,
    MYSQL_TABLE,
)


def load_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "No vector collection found. Run: python src/ingest.py"
        ) from exc


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
    for doc_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if (
            source_name is not None
            and resolved_filters is not None
            and not metadata_matches_filters(
                metadata,
                source_name,
                resolved_filters,
            )
        ):
            continue
        output.append(
            {
                "id": doc_id,
                "text": text,
                "metadata": metadata,
                "score": float(distance),
                "source": "vector",
            }
        )
        if len(output) >= top_k:
            break
    return output


def bm25_search(query, index, collection, resolved_filters, top_k=15):
    ranked = index.search(query, resolved_filters, top_k)
    if not ranked:
        return []

    scores = {item["doc_id"]: item["score"] for item in ranked}
    ordered_ids = [item["doc_id"] for item in ranked]
    data = collection.get(ids=ordered_ids, include=["documents", "metadatas"])
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


def extract_product_ids(candidates):
    product_ids = []
    seen = set()

    for result in candidates:
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


def filter_candidates_by_ad_type(
    candidates,
    target_ad_type: str,
    connection=None,
):
    expected_type = WANTED_AD_TYPE if target_ad_type == "wanted" else OFFER_AD_TYPE
    product_ids = extract_product_ids(candidates)
    product_types = fetch_product_types_by_ids(product_ids, connection=connection)

    filtered = []
    for candidate in candidates:
        candidate_ids = extract_product_ids([candidate])
        if not candidate_ids:
            continue
        if product_types.get(str(candidate_ids[0])) == expected_type:
            filtered.append(candidate)
    return filtered
