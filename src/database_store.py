from __future__ import annotations

from typing import TypeAlias

from mysql_store import (
    DEFAULT_MYSQL_CONFIG,
    MySQLRuntimeConfig,
    count_mysql_rows,
    detect_mysql_primary_key,
    fetch_mysql_columns,
    fetch_product_types_by_ids as fetch_mysql_product_types_by_ids,
    fetch_products_by_ids as fetch_mysql_products_by_ids,
    iter_mysql_rows,
    mysql_source_name,
)
from postgres_store import (
    PostgresRuntimeConfig,
    count_postgres_rows,
    detect_postgres_primary_key,
    fetch_postgres_columns,
    fetch_postgres_product_types_by_ids,
    fetch_postgres_products_by_ids,
    iter_postgres_rows,
    postgres_source_name,
)


DatabaseRuntimeConfig: TypeAlias = MySQLRuntimeConfig | PostgresRuntimeConfig


def resolved_database_config(
    config: DatabaseRuntimeConfig | None = None,
) -> DatabaseRuntimeConfig:
    return config or DEFAULT_MYSQL_CONFIG


def database_backend(config: DatabaseRuntimeConfig | None = None) -> str:
    return (
        "postgres"
        if isinstance(resolved_database_config(config), PostgresRuntimeConfig)
        else "mysql"
    )


def database_source_name(
    config: DatabaseRuntimeConfig | None = None,
) -> str:
    resolved = resolved_database_config(config)
    if isinstance(resolved, PostgresRuntimeConfig):
        return postgres_source_name(resolved)
    return mysql_source_name(resolved)


def fetch_database_columns(
    config: DatabaseRuntimeConfig | None = None,
) -> list[str]:
    resolved = resolved_database_config(config)
    if isinstance(resolved, PostgresRuntimeConfig):
        return fetch_postgres_columns(resolved)
    return fetch_mysql_columns(config=resolved)


def detect_database_primary_key(
    columns: list[str],
    override: str | None = None,
    config: DatabaseRuntimeConfig | None = None,
) -> str | None:
    resolved = resolved_database_config(config)
    if isinstance(resolved, PostgresRuntimeConfig):
        return detect_postgres_primary_key(resolved, columns, override)
    return detect_mysql_primary_key(columns, override, config=resolved)


def count_database_rows(
    content_column: str | None = None,
    config: DatabaseRuntimeConfig | None = None,
) -> int:
    resolved = resolved_database_config(config)
    if isinstance(resolved, PostgresRuntimeConfig):
        return count_postgres_rows(resolved, content_column)
    return count_mysql_rows(content_column, config=resolved)


def iter_database_rows(
    content_column: str | None,
    primary_key_column: str | None,
    limit: int | None = None,
    config: DatabaseRuntimeConfig | None = None,
):
    resolved = resolved_database_config(config)
    if isinstance(resolved, PostgresRuntimeConfig):
        yield from iter_postgres_rows(
            resolved,
            content_column,
            primary_key_column,
            limit,
        )
        return
    yield from iter_mysql_rows(
        content_column,
        primary_key_column,
        limit,
        config=resolved,
    )


def fetch_product_types_by_ids(
    product_ids,
    connection=None,
    config: DatabaseRuntimeConfig | None = None,
) -> dict[str, str]:
    resolved = resolved_database_config(config)
    if isinstance(resolved, PostgresRuntimeConfig):
        return fetch_postgres_product_types_by_ids(
            resolved,
            product_ids,
            connection,
        )
    return fetch_mysql_product_types_by_ids(
        product_ids,
        connection=connection,
        config=resolved,
    )


def fetch_products_by_ids(
    product_ids,
    connection=None,
    config: DatabaseRuntimeConfig | None = None,
) -> list[dict]:
    resolved = resolved_database_config(config)
    if isinstance(resolved, PostgresRuntimeConfig):
        return fetch_postgres_products_by_ids(
            resolved,
            product_ids,
            connection,
        )
    return fetch_mysql_products_by_ids(
        product_ids,
        connection=connection,
        config=resolved,
    )
