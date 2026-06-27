from collections import Counter

import chromadb

from settings import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL


def get_collection(create: bool = False):
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    if create:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return client, collection
    try:
        return client, client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "No vector collection found. Run: python src/ingest.py"
        ) from exc


def list_indexed_documents() -> None:
    _, collection = get_collection()
    data = collection.get(include=["metadatas"])
    counts = Counter(metadata["source_file"] for metadata in data["metadatas"])

    print(f"Collection: {COLLECTION_NAME} ({collection.count()} chunks)\n")
    for filename, count in sorted(counts.items(), key=lambda item: item[0].lower()):
        print(f"{count:>5} chunks  {filename}")


def confirm(message: str, assume_yes: bool = False) -> bool:
    return assume_yes or input(f"{message} [y/N]: ").strip().lower() in {"y", "yes"}


def delete_indexed_document(filename: str, assume_yes: bool = False) -> None:
    _, collection = get_collection()
    existing = collection.get(where={"source_file": filename}, include=[])
    count = len(existing["ids"])
    if not count:
        raise RuntimeError(f"'{filename}' is not present in the collection.")
    if not confirm(
        f"Delete {count} indexed chunks for '{filename}'? The source file will be kept.",
        assume_yes,
    ):
        print("Cancelled.")
        return
    collection.delete(where={"source_file": filename})
    print(f"Deleted {count} chunks for '{filename}'.")


def clear_collection(assume_yes: bool = False) -> None:
    client, collection = get_collection()
    count = collection.count()
    if not confirm(
        f"Delete the entire '{COLLECTION_NAME}' collection ({count} chunks)?",
        assume_yes,
    ):
        print("Cancelled.")
        return
    client.delete_collection(COLLECTION_NAME)
    print(f"Deleted collection '{COLLECTION_NAME}'. Source files were kept.")


def source_is_current(
    collection,
    filename: str,
    ids: list[str],
    documents: list[str],
) -> bool:
    existing = collection.get(
        where={"source_file": filename},
        include=["documents", "metadatas"],
    )
    if len(existing["ids"]) != len(ids):
        return False

    stored_documents = dict(zip(existing["ids"], existing["documents"]))
    expected_documents = dict(zip(ids, documents))
    models_match = all(
        metadata.get("embedding_model") == EMBED_MODEL
        for metadata in existing["metadatas"]
    )
    return models_match and stored_documents == expected_documents


def mysql_current_ids(
    collection,
    ids: list[str],
    documents: list[str],
    metadatas: list[dict],
) -> set[str]:
    if not ids:
        return set()
    existing = collection.get(ids=ids, include=["documents", "metadatas"])
    expected = {
        doc_id: {
            "document": document,
            "hash": metadata.get("source_content_hash"),
        }
        for doc_id, document, metadata in zip(ids, documents, metadatas)
    }
    current = set()
    for doc_id, document, metadata in zip(
        existing["ids"],
        existing["documents"],
        existing["metadatas"],
    ):
        if metadata.get("embedding_model") != EMBED_MODEL:
            continue
        expected_row = expected.get(doc_id, {})
        hash_matches = metadata.get("source_content_hash") == expected_row.get("hash")
        document_matches = document == expected_row.get("document")
        if hash_matches or document_matches:
            current.add(doc_id)
    return current
