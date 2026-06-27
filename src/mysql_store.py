from typing import Any

from settings import (
    MYSQL_CONTENT_COLUMN,
    MYSQL_DATABASE,
    MYSQL_HOST,
    MYSQL_PASSWORD,
    MYSQL_PORT,
    MYSQL_RESULT_ID_COLUMN,
    MYSQL_RESULT_TABLE,
    MYSQL_TABLE,
    MYSQL_USER,
)

MYSQL_ID_CANDIDATES = ("id", "ad_id", "ads_id", "adId", "adsId")


def require_pymysql():
    try:
        import pymysql
    except ImportError as exc:
        raise RuntimeError(
            "MySQL support requires PyMySQL. Install requirements.txt first."
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


def fetch_mysql_columns(table: str = MYSQL_TABLE) -> list[str]:
    pymysql = require_pymysql()
    with mysql_connection(cursorclass=pymysql.cursors.DictCursor) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SHOW COLUMNS FROM {quote_mysql_identifier(table)}")
            return [row["Field"] for row in cursor.fetchall()]


def detect_mysql_primary_key(
    columns: list[str],
    override: str | None = None,
    table: str = MYSQL_TABLE,
) -> str | None:
    if override:
        if override not in columns:
            raise RuntimeError(f"MySQL primary key column '{override}' was not found.")
        return override

    pymysql = require_pymysql()
    with mysql_connection(cursorclass=pymysql.cursors.DictCursor) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SHOW KEYS FROM {quote_mysql_identifier(table)} "
                "WHERE Key_name = 'PRIMARY'"
            )
            keys = cursor.fetchall()
            if keys:
                return keys[0]["Column_name"]

    return next((name for name in MYSQL_ID_CANDIDATES if name in columns), None)


def count_mysql_rows(
    content_column: str = MYSQL_CONTENT_COLUMN,
    table: str = MYSQL_TABLE,
) -> int:
    pymysql = require_pymysql()
    quoted_content = quote_mysql_identifier(content_column)
    where_clause = f"{quoted_content} IS NOT NULL AND TRIM({quoted_content}) <> ''"
    with mysql_connection(cursorclass=pymysql.cursors.DictCursor) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT COUNT(*) AS row_count FROM {quote_mysql_identifier(table)} "
                f"WHERE {where_clause}"
            )
            return int(cursor.fetchone()["row_count"])


def iter_mysql_rows(
    content_column: str,
    primary_key_column: str | None,
    limit: int | None = None,
    table: str = MYSQL_TABLE,
):
    pymysql = require_pymysql()
    quoted_content = quote_mysql_identifier(content_column)
    query = (
        f"SELECT * FROM {quote_mysql_identifier(table)} "
        f"WHERE {quoted_content} IS NOT NULL AND TRIM({quoted_content}) <> ''"
    )
    params: list[Any] = []
    if primary_key_column:
        query += f" ORDER BY {quote_mysql_identifier(primary_key_column)}"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    with mysql_connection(cursorclass=pymysql.cursors.SSDictCursor) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, params)
            yield from cursor


def fetch_product_types_by_ids(
    product_ids,
    connection=None,
) -> dict[str, str]:
    unique_ids = list(dict.fromkeys(product_ids))
    if not unique_ids:
        return {}

    owns_connection = connection is None
    if owns_connection:
        pymysql = require_pymysql()
        connection = mysql_connection(cursorclass=pymysql.cursors.DictCursor)

    placeholders = ", ".join(["%s"] * len(unique_ids))
    query = (
        f"SELECT {quote_mysql_identifier(MYSQL_RESULT_ID_COLUMN)}, `type` "
        f"FROM {quote_mysql_identifier(MYSQL_RESULT_TABLE)} "
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

    return {
        str(row[MYSQL_RESULT_ID_COLUMN]): str(row["type"])
        for row in rows
        if row.get(MYSQL_RESULT_ID_COLUMN) is not None
        and row.get("type") is not None
    }


def fetch_products_by_ids(product_ids, connection=None) -> list[dict]:
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
