import time
from pathlib import Path

from bm25_index import PersistentBM25Index
from chroma_store import get_collection, mysql_current_ids, source_is_current
from document_processing import (
    prepare_bm25_index_row,
    prepare_mysql_row,
    prepare_source,
)
from mysql_store import (
    count_mysql_rows,
    detect_mysql_primary_key,
    fetch_mysql_columns,
    iter_mysql_rows,
    mysql_source_name,
)
from ollama_client import embed_texts
from settings import (
    CHROMA_DIR,
    MYSQL_BM25_COLUMN,
    MYSQL_CONTENT_COLUMN,
    MYSQL_DATABASE,
    MYSQL_TABLE,
)

EMBED_BATCH_SIZE = 32
MYSQL_BATCH_SIZE = 500


def check_sources(source_files: list[Path]) -> bool:
    valid = True
    total_units = 0
    total_chunks = 0
    for path in source_files:
        try:
            ids, _, _, skipped_units, unit_count = prepare_source(path)
            total_units += unit_count
            total_chunks += len(ids)
            print(
                f"OK: {path.name} | {unit_count} source units | {len(ids)} chunks | "
                f"{skipped_units} empty units"
            )
        except Exception as exc:
            valid = False
            print(f"ERROR: {path.name} | {type(exc).__name__}: {exc}")

    print(
        f"\nChecked {len(source_files)} source files, "
        f"{total_units} source units, {total_chunks} chunks."
    )
    return valid


def check_mysql_source(
    limit: int | None = None,
    primary_key_column: str | None = None,
) -> bool:
    columns = fetch_mysql_columns()
    if MYSQL_CONTENT_COLUMN not in columns:
        print(
            f"ERROR: column '{MYSQL_CONTENT_COLUMN}' was not found in "
            f"{MYSQL_DATABASE}.{MYSQL_TABLE}."
        )
        print(f"Available columns: {', '.join(columns)}")
        return False

    detected_primary_key = detect_mysql_primary_key(columns, primary_key_column)
    row_count = count_mysql_rows(MYSQL_CONTENT_COLUMN)
    planned_rows = min(row_count, limit) if limit is not None else row_count

    print(f"OK: MySQL table {MYSQL_DATABASE}.{MYSQL_TABLE}")
    print(f"Content column: {MYSQL_CONTENT_COLUMN}")
    print(f"Primary key column: {detected_primary_key or 'none detected'}")
    print(f"Rows with embedding text: {row_count}")
    print(f"Rows planned for ingestion: {planned_rows}")
    print("No embeddings were generated during this check.")
    return True


def embed_for_upsert(
    documents: list[str],
    embed_batch_size: int = EMBED_BATCH_SIZE,
    progress_prefix: str = "",
) -> list[list[float]]:
    embeddings = []
    for start in range(0, len(documents), embed_batch_size):
        batch = documents[start : start + embed_batch_size]
        if progress_prefix:
            completed = min(start + len(batch), len(documents))
            print(
                f"{progress_prefix} embedding {completed}/{len(documents)} texts",
                flush=True,
            )
        embeddings.extend(embed_texts(batch))
    return embeddings


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def ingest_sources(source_files: list[Path]) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _, collection = get_collection(create=True)

    for path in source_files:
        print(f"Processing: {path.name}")
        try:
            ids, documents, metadatas, skipped_units, _ = prepare_source(path)
            if not documents:
                print("  Skipped: no extractable text (OCR may be required).")
                continue
            if source_is_current(collection, path.name, ids, documents):
                print(f"  Unchanged: keeping {len(documents)} existing chunks.")
                continue

            embeddings = []
            for start in range(0, len(documents), EMBED_BATCH_SIZE):
                batch = documents[start : start + EMBED_BATCH_SIZE]
                embeddings.extend(embed_texts(batch))
                completed = min(start + len(batch), len(documents))
                print(
                    f"  Embedded {completed}/{len(documents)} chunks",
                    end="\r",
                    flush=True,
                )
            print()

            # Delete stale chunks only after extraction and embedding succeed.
            collection.delete(where={"source_file": path.name})
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            print(
                f"  Added {len(documents)} chunks; "
                f"skipped {skipped_units} empty units."
            )
        except RuntimeError:
            raise
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")

    print(f"\nIngestion complete. Collection contains {collection.count()} chunks.")


def ingest_mysql_source(
    limit: int | None = None,
    batch_size: int = MYSQL_BATCH_SIZE,
    embed_batch_size: int = EMBED_BATCH_SIZE,
    primary_key_column: str | None = None,
    replace_source: bool = False,
    force_reembed: bool = False,
) -> None:
    if batch_size <= 0:
        raise RuntimeError("--mysql-batch-size must be greater than zero.")
    if embed_batch_size <= 0:
        raise RuntimeError("--embed-batch-size must be greater than zero.")
    if limit is not None and limit <= 0:
        raise RuntimeError("--limit must be greater than zero.")

    columns = fetch_mysql_columns()
    if MYSQL_CONTENT_COLUMN not in columns:
        raise RuntimeError(
            f"Column '{MYSQL_CONTENT_COLUMN}' was not found in "
            f"{MYSQL_DATABASE}.{MYSQL_TABLE}."
        )
    detected_primary_key = detect_mysql_primary_key(columns, primary_key_column)
    bm25_column = (
        MYSQL_BM25_COLUMN if MYSQL_BM25_COLUMN in columns else MYSQL_CONTENT_COLUMN
    )
    row_count = count_mysql_rows(MYSQL_CONTENT_COLUMN)
    planned_rows = min(row_count, limit) if limit is not None else row_count

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    _, collection = get_collection(create=True)
    bm25_index = PersistentBM25Index()
    source_name = mysql_source_name()

    print(f"Processing MySQL table: {MYSQL_DATABASE}.{MYSQL_TABLE}")
    print(f"Content column: {MYSQL_CONTENT_COLUMN}")
    print(f"BM25 column: {bm25_column}")
    print(f"Primary key column: {detected_primary_key or 'none detected'}")
    print(f"Rows planned for ingestion: {planned_rows}")

    if replace_source:
        existing = collection.get(where={"source_file": source_name}, include=[])
        if existing["ids"]:
            collection.delete(where={"source_file": source_name})
            print(f"Deleted {len(existing['ids'])} existing chunks for {source_name}.")
        bm25_index.clear()
        print("Cleared the persistent BM25 product index.")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    bm25_rows: list[dict] = []
    indexed = 0
    processed = 0
    skipped_empty = 0
    skipped_current = 0
    started_at = time.monotonic()

    def flush_batch() -> None:
        nonlocal ids, documents, metadatas, bm25_rows, indexed, skipped_current
        if not documents:
            return
        batch_start = processed - len(documents) + 1
        batch_end = processed
        total_label = planned_rows if planned_rows else "unknown"
        bm25_index.upsert(bm25_rows)

        if force_reembed:
            upsert_ids = ids
            upsert_documents = documents
            upsert_metadatas = metadatas
        else:
            current_ids = mysql_current_ids(collection, ids, documents, metadatas)
            skipped_current += len(current_ids)
            upsert_ids = []
            upsert_documents = []
            upsert_metadatas = []
            for doc_id, document, metadata in zip(ids, documents, metadatas):
                if doc_id in current_ids:
                    continue
                upsert_ids.append(doc_id)
                upsert_documents.append(document)
                upsert_metadatas.append(metadata)

        if not upsert_documents:
            print(
                f"  Rows {batch_start}-{batch_end}/{total_label} unchanged; skipped.",
                flush=True,
            )
            ids = []
            documents = []
            metadatas = []
            bm25_rows = []
            return

        print(
            f"  Preparing rows {batch_start}-{batch_end}/{total_label} for Chroma "
            f"({len(upsert_documents)} changed/new)",
            flush=True,
        )
        embeddings = embed_for_upsert(
            upsert_documents,
            embed_batch_size,
            progress_prefix=f"    rows {batch_start}-{batch_end}",
        )
        collection.upsert(
            ids=upsert_ids,
            documents=upsert_documents,
            embeddings=embeddings,
            metadatas=upsert_metadatas,
        )
        indexed += len(upsert_documents)
        elapsed = time.monotonic() - started_at
        rate = processed / elapsed if elapsed else 0
        remaining = max(planned_rows - processed, 0)
        eta = remaining / rate if rate else 0
        print(
            f"  Indexed/updated {indexed} rows; skipped unchanged {skipped_current}; "
            f"processed {processed}/{total_label}; ETA {format_duration(eta)}",
            flush=True,
        )
        ids = []
        documents = []
        metadatas = []
        bm25_rows = []

    for row in iter_mysql_rows(MYSQL_CONTENT_COLUMN, detected_primary_key, limit):
        prepared = prepare_mysql_row(row, MYSQL_CONTENT_COLUMN, detected_primary_key)
        if prepared is None:
            skipped_empty += 1
            continue

        doc_id, document, metadata = prepared
        ids.append(doc_id)
        documents.append(document)
        metadatas.append(metadata)
        bm25_row = prepare_bm25_index_row(
            row,
            bm25_column,
            detected_primary_key,
        )
        if bm25_row is not None:
            bm25_rows.append(bm25_row)
        processed += 1
        if len(documents) >= batch_size:
            flush_batch()

    flush_batch()
    bm25_count = bm25_index.count()
    bm25_index.close()
    print(
        f"\nMySQL ingestion complete. Indexed/updated {indexed} rows; "
        f"skipped unchanged {skipped_current} rows; skipped empty {skipped_empty} rows. "
        f"Collection contains {collection.count()} chunks. "
        f"BM25 index contains {bm25_count} products."
    )


def rebuild_mysql_bm25_index(
    limit: int | None = None,
    batch_size: int = MYSQL_BATCH_SIZE,
    primary_key_column: str | None = None,
) -> None:
    if batch_size <= 0:
        raise RuntimeError("--mysql-batch-size must be greater than zero.")
    if limit is not None and limit <= 0:
        raise RuntimeError("--limit must be greater than zero.")

    columns = fetch_mysql_columns()
    if MYSQL_CONTENT_COLUMN not in columns:
        raise RuntimeError(
            f"Column '{MYSQL_CONTENT_COLUMN}' was not found in "
            f"{MYSQL_DATABASE}.{MYSQL_TABLE}."
        )
    detected_primary_key = detect_mysql_primary_key(columns, primary_key_column)
    bm25_column = (
        MYSQL_BM25_COLUMN if MYSQL_BM25_COLUMN in columns else MYSQL_CONTENT_COLUMN
    )
    row_count = count_mysql_rows(MYSQL_CONTENT_COLUMN)
    planned_rows = min(row_count, limit) if limit is not None else row_count

    index = PersistentBM25Index()
    index.clear()
    batch = []
    processed = 0

    print(f"Rebuilding BM25 index from {MYSQL_DATABASE}.{MYSQL_TABLE}")
    print(f"BM25 column: {bm25_column}")
    print(f"Rows planned: {planned_rows}")

    for row in iter_mysql_rows(MYSQL_CONTENT_COLUMN, detected_primary_key, limit):
        entry = prepare_bm25_index_row(
            row,
            bm25_column,
            detected_primary_key,
        )
        if entry is None:
            continue
        batch.append(entry)
        processed += 1
        if len(batch) >= batch_size:
            index.upsert(batch)
            batch = []
            print(f"  Indexed {processed}/{planned_rows}", end="\r", flush=True)

    index.upsert(batch)
    count = index.count()
    index.close()
    print(f"\nBM25 rebuild complete. Indexed {count} products.")
