import argparse

from chroma_store import (
    clear_collection,
    confirm,
    delete_indexed_document,
    get_collection,
    list_indexed_documents,
    mysql_current_ids,
    source_is_current,
)
from document_processing import (
    SUPPORTED_EXTENSIONS,
    cell_to_text,
    chunk_id,
    chunk_text,
    content_hash,
    find_source_files,
    metadata_value,
    mysql_document_id,
    mysql_row_identity,
    prepare_bm25_index_row,
    prepare_content_document,
    prepare_mysql_row,
    prepare_pdf,
    prepare_source,
    prepare_table,
)
from ingestion_service import (
    EMBED_BATCH_SIZE,
    MYSQL_BATCH_SIZE,
    check_mysql_source,
    check_sources,
    embed_for_upsert,
    format_duration,
    ingest_mysql_source,
    ingest_sources,
    rebuild_mysql_bm25_index,
)
from mysql_store import (
    count_mysql_rows,
    detect_mysql_primary_key,
    fetch_mysql_columns,
    iter_mysql_rows,
    mysql_connection,
    mysql_source_name,
    quote_mysql_identifier,
    require_pymysql,
)
from ollama_client import embed_texts
from settings import EMBED_MODEL, SOURCE_FILES_DIR
from tenant_config import discover_tenant_profiles
from vector_store import (
    clear_tenant_vectors,
    delete_tenant_source,
    list_tenant_vectors,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest local files or a tenant's MySQL/PostgreSQL search-ready "
            "table into its configured Chroma/pgvector and BM25 indexes."
        )
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--check",
        action="store_true",
        help="validate source files without calling Ollama",
    )
    actions.add_argument(
        "--list",
        action="store_true",
        help="list indexed source files and chunk counts",
    )
    actions.add_argument(
        "--delete",
        metavar="FILENAME",
        help="delete one source file from the index",
    )
    actions.add_argument(
        "--clear",
        action="store_true",
        help="delete the entire vector collection",
    )
    parser.add_argument(
        "--company",
        help=(
            "tenant profile slug under configs/tenants; required for isolated "
            "production database ingestion"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip confirmation for delete operations",
    )
    parser.add_argument(
        "--mysql",
        action="store_true",
        help=(
            "ingest the configured MySQL/PostgreSQL table; the flag name is "
            "retained for compatibility"
        ),
    )
    parser.add_argument(
        "--mysql-bm25-only",
        action="store_true",
        help="rebuild only BM25 from the configured database; no embeddings",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="limit database rows for smoke-test ingestion",
    )
    parser.add_argument(
        "--mysql-batch-size",
        type=int,
        default=MYSQL_BATCH_SIZE,
        help=(
            "database rows per vector/BM25 batch "
            f"(default: {MYSQL_BATCH_SIZE})"
        ),
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=EMBED_BATCH_SIZE,
        help=f"texts to send per Ollama embedding request (default: {EMBED_BATCH_SIZE})",
    )
    parser.add_argument(
        "--mysql-primary-key",
        help="override the detected database primary-key column",
    )
    parser.add_argument(
        "--mysql-replace-source",
        action="store_true",
        help=(
            "clear this company's vector/BM25 source before authoritative rebuild"
        ),
    )
    parser.add_argument(
        "--mysql-force-reembed",
        action="store_true",
        help="re-embed rows even when the existing content hash matches",
    )
    args = parser.parse_args()
    tenant = None
    if args.company:
        profiles = discover_tenant_profiles()
        try:
            tenant = profiles[args.company]
        except KeyError as exc:
            available = ", ".join(sorted(profiles)) or "none"
            raise SystemExit(
                f"Unknown company {args.company!r}; available: {available}"
            ) from exc

    collection_options = {}
    if tenant is not None:
        collection_options = {
            "chroma_dir": tenant.storage.chroma_dir,
            "collection_name": tenant.storage.collection_name,
        }

    if args.list:
        if tenant:
            list_tenant_vectors(tenant)
        else:
            list_indexed_documents(**collection_options)
        return
    if args.delete:
        if tenant:
            if not confirm(
                f"Delete tenant vectors for source {args.delete!r}?",
                args.yes,
            ):
                print("Cancelled.")
                return
            deleted = delete_tenant_source(tenant, args.delete)
            print(f"Deleted {deleted} vectors.")
        else:
            delete_indexed_document(
                args.delete,
                args.yes,
                **collection_options,
            )
        return
    if args.clear:
        if tenant:
            if not confirm(
                f"Delete all vectors for company {tenant.company_id!r}?",
                args.yes,
            ):
                print("Cancelled.")
                return
            deleted = clear_tenant_vectors(tenant)
            print(f"Deleted {deleted} vectors.")
        else:
            clear_collection(args.yes, **collection_options)
        return
    if args.mysql and args.check:
        raise SystemExit(
            0
            if check_mysql_source(
                args.limit,
                args.mysql_primary_key,
                tenant=tenant,
            )
            else 1
        )
    if args.mysql_bm25_only:
        rebuild_mysql_bm25_index(
            limit=args.limit,
            batch_size=args.mysql_batch_size,
            primary_key_column=args.mysql_primary_key,
            tenant=tenant,
        )
        return
    if args.mysql:
        ingest_mysql_source(
            limit=args.limit,
            batch_size=args.mysql_batch_size,
            embed_batch_size=args.embed_batch_size,
            primary_key_column=args.mysql_primary_key,
            replace_source=args.mysql_replace_source,
            force_reembed=args.mysql_force_reembed,
            tenant=tenant,
        )
        return

    if tenant is not None:
        raise SystemExit(
            "--company currently applies to configured database ingestion only; "
            "add --mysql (legacy flag name for MySQL or PostgreSQL)."
        )

    source_files = find_source_files()
    if not source_files:
        extensions = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise SystemExit(
            f"No supported files ({extensions}) found in {SOURCE_FILES_DIR}"
        )

    if args.check:
        raise SystemExit(0 if check_sources(source_files) else 1)
    ingest_sources(source_files)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"Error: {exc}") from exc
