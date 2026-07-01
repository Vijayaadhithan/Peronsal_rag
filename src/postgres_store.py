from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class PostgresRuntimeConfig:
    host: str
    port: int
    database: str
    user: str
    password: str = field(repr=False)
    search_table: str = "ads_search_ready"
    content_column: str = "embedding_content"
    bm25_column: str = "bm25_content"
    search_id_column: str = "id"
    result_table: str = "ads"
    result_id_column: str = "id"
    result_type_column: str = "type"
    schema: str = "public"


def require_psycopg():
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL support requires psycopg. Install requirements.txt first."
        ) from exc
    return psycopg, dict_row


def postgres_connection(
    config: PostgresRuntimeConfig,
    *,
    dict_rows: bool = False,
    autocommit: bool = True,
):
    psycopg, dict_row = require_psycopg()
    options = {
        "host": config.host,
        "port": config.port,
        "dbname": config.database,
        "user": config.user,
        "password": config.password,
        "autocommit": autocommit,
    }
    if dict_rows:
        options["row_factory"] = dict_row
    return psycopg.connect(**options)


def quote_postgres_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValueError("PostgreSQL identifiers must be non-empty strings")
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def qualified_table(config: PostgresRuntimeConfig, table: str) -> str:
    return (
        f"{quote_postgres_identifier(config.schema)}."
        f"{quote_postgres_identifier(table)}"
    )


def postgres_source_name(config: PostgresRuntimeConfig) -> str:
    return (
        f"postgres:{config.database}.{config.schema}."
        f"{config.search_table}"
    )


def fetch_postgres_columns(
    config: PostgresRuntimeConfig,
    table: str | None = None,
) -> list[str]:
    table = table or config.search_table
    with postgres_connection(config, dict_rows=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (config.schema, table),
            )
            return [row["column_name"] for row in cursor.fetchall()]


def detect_postgres_primary_key(
    config: PostgresRuntimeConfig,
    columns: list[str],
    override: str | None = None,
    table: str | None = None,
) -> str | None:
    table = table or config.search_table
    if override:
        if override not in columns:
            raise RuntimeError(
                f"PostgreSQL primary key column {override!r} was not found."
            )
        return override
    with postgres_connection(config, dict_rows=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = %s
                  AND tc.table_name = %s
                ORDER BY kcu.ordinal_position
                LIMIT 1
                """,
                (config.schema, table),
            )
            row = cursor.fetchone()
    if row:
        return row["column_name"]
    candidates = ("id", "product_id", "ad_id", "ads_id")
    return next((name for name in candidates if name in columns), None)


def count_postgres_rows(
    config: PostgresRuntimeConfig,
    content_column: str | None = None,
    table: str | None = None,
) -> int:
    content_column = content_column or config.content_column
    table = table or config.search_table
    quoted_content = quote_postgres_identifier(content_column)
    with postgres_connection(config, dict_rows=True) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) AS row_count FROM "
                f"{qualified_table(config, table)} "
                f"WHERE {quoted_content} IS NOT NULL "
                f"AND BTRIM({quoted_content}::text) <> ''"
            )
            return int(cursor.fetchone()["row_count"])


def iter_postgres_rows(
    config: PostgresRuntimeConfig,
    content_column: str | None,
    primary_key_column: str | None,
    limit: int | None = None,
    table: str | None = None,
):
    content_column = content_column or config.content_column
    table = table or config.search_table
    quoted_content = quote_postgres_identifier(content_column)
    query = (
        f"SELECT * FROM {qualified_table(config, table)} "
        f"WHERE {quoted_content} IS NOT NULL "
        f"AND BTRIM({quoted_content}::text) <> ''"
    )
    params: list[Any] = []
    if primary_key_column:
        query += f" ORDER BY {quote_postgres_identifier(primary_key_column)}"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    with postgres_connection(
        config,
        dict_rows=True,
        autocommit=False,
    ) as connection:
        cursor_name = f"rag_ht_{uuid4().hex[:12]}"
        with connection.cursor(name=cursor_name) as cursor:
            cursor.execute(query, params)
            for row in cursor:
                yield dict(row)


def fetch_postgres_product_types_by_ids(
    config: PostgresRuntimeConfig,
    product_ids,
    connection=None,
) -> dict[str, str]:
    unique_ids = list(dict.fromkeys(product_ids))
    if not unique_ids:
        return {}
    owns_connection = connection is None
    if owns_connection:
        connection = postgres_connection(config, dict_rows=True)
    id_column = quote_postgres_identifier(config.result_id_column)
    placeholders = ", ".join(["%s"] * len(unique_ids))
    query = (
        f"SELECT {id_column}, "
        f"{quote_postgres_identifier(config.result_type_column)} "
        f"FROM {qualified_table(config, config.result_table)} "
        f"WHERE {id_column} IN ({placeholders})"
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, unique_ids)
            rows = cursor.fetchall()
    finally:
        if owns_connection:
            connection.close()
    return {
        str(row[config.result_id_column]): str(row[config.result_type_column])
        for row in rows
        if row.get(config.result_id_column) is not None
        and row.get(config.result_type_column) is not None
    }


def fetch_postgres_products_by_ids(
    config: PostgresRuntimeConfig,
    product_ids,
    connection=None,
) -> list[dict]:
    unique_ids = list(dict.fromkeys(product_ids))
    if not unique_ids:
        return []
    owns_connection = connection is None
    if owns_connection:
        connection = postgres_connection(config, dict_rows=True)
    id_column = quote_postgres_identifier(config.result_id_column)
    placeholders = ", ".join(["%s"] * len(unique_ids))
    query = (
        f"SELECT * FROM {qualified_table(config, config.result_table)} "
        f"WHERE {id_column} IN ({placeholders})"
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, unique_ids)
            rows = [dict(row) for row in cursor.fetchall()]
    finally:
        if owns_connection:
            connection.close()
    rows_by_id = {
        str(row[config.result_id_column]): row
        for row in rows
        if row.get(config.result_id_column) is not None
    }
    return [
        rows_by_id[str(product_id)]
        for product_id in unique_ids
        if str(product_id) in rows_by_id
    ]
