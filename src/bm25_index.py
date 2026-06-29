import re
import sqlite3
from pathlib import Path

from settings import BM25_INDEX_PATH, UNPRICED_RENTAL_FEE_CEILING

FILTER_COLUMNS = (
    "main_category_name",
    "subcategory_name",
    "state_name",
    "city_name",
    "locality_name",
    "rental_duration",
)


def tokenize_query(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
        )
    )


class PersistentBM25Index:
    def __init__(self, path: Path | str = BM25_INDEX_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The API creates the index during application startup and serves
        # synchronous requests from worker threads. Access is serialized by
        # ProductSearchService, so the connection can safely cross threads.
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS products (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL UNIQUE,
                product_id TEXT NOT NULL,
                content TEXT NOT NULL,
                main_category_name TEXT,
                subcategory_name TEXT,
                state_name TEXT,
                city_name TEXT,
                locality_name TEXT,
                rental_duration TEXT,
                rental_fee REAL
            );

            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value INTEGER NOT NULL
            );

            INSERT OR IGNORE INTO index_metadata(key, value)
            VALUES ('revision', 0);

            CREATE VIRTUAL TABLE IF NOT EXISTS products_fts USING fts5(
                content,
                content='products',
                content_rowid='rowid',
                tokenize='unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS products_ai AFTER INSERT ON products BEGIN
                INSERT INTO products_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS products_ad AFTER DELETE ON products BEGIN
                INSERT INTO products_fts(products_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS products_au AFTER UPDATE ON products BEGIN
                INSERT INTO products_fts(products_fts, rowid, content)
                VALUES ('delete', old.rowid, old.content);
                INSERT INTO products_fts(rowid, content)
                VALUES (new.rowid, new.content);
            END;

            CREATE INDEX IF NOT EXISTS products_main_category
                ON products(main_category_name);
            CREATE INDEX IF NOT EXISTS products_subcategory
                ON products(subcategory_name);
            CREATE INDEX IF NOT EXISTS products_state
                ON products(state_name);
            CREATE INDEX IF NOT EXISTS products_city
                ON products(city_name);
            CREATE INDEX IF NOT EXISTS products_locality
                ON products(locality_name);
            CREATE INDEX IF NOT EXISTS products_rental_duration
                ON products(rental_duration);
            CREATE INDEX IF NOT EXISTS products_rental_fee
                ON products(rental_fee);
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS row_count FROM products"
        ).fetchone()
        return int(row["row_count"])

    def revision(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = 'revision'"
        ).fetchone()
        return int(row["value"])

    def _increment_revision(self) -> None:
        self.connection.execute(
            "UPDATE index_metadata SET value = value + 1 "
            "WHERE key = 'revision'"
        )

    def clear(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM products")
            self._increment_revision()

    def upsert(self, rows: list[dict]) -> None:
        if not rows:
            return
        values = [
            (
                str(row["doc_id"]),
                str(row["product_id"]),
                row["content"],
                row.get("main_category_name"),
                row.get("subcategory_name"),
                row.get("state_name"),
                row.get("city_name"),
                row.get("locality_name"),
                row.get("rental_duration"),
                row.get("rental_fee"),
            )
            for row in rows
        ]
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO products (
                    doc_id,
                    product_id,
                    content,
                    main_category_name,
                    subcategory_name,
                    state_name,
                    city_name,
                    locality_name,
                    rental_duration,
                    rental_fee
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    product_id = excluded.product_id,
                    content = excluded.content,
                    main_category_name = excluded.main_category_name,
                    subcategory_name = excluded.subcategory_name,
                    state_name = excluded.state_name,
                    city_name = excluded.city_name,
                    locality_name = excluded.locality_name,
                    rental_duration = excluded.rental_duration,
                    rental_fee = excluded.rental_fee
                """,
                values,
            )
            self._increment_revision()

    def filter_value_index(self) -> dict:
        value_index = {}
        for column in FILTER_COLUMNS:
            rows = self.connection.execute(
                f"SELECT DISTINCT {column} AS value FROM products "
                f"WHERE {column} IS NOT NULL AND TRIM({column}) <> ''"
            ).fetchall()
            value_index[column] = {
                " ".join(str(row["value"]).casefold().split()): row["value"]
                for row in rows
            }
        return value_index

    def _unique_relationship_index(
        self,
        child_column: str,
        parent_columns: tuple[str, ...],
    ) -> dict[str, tuple[str, ...]]:
        allowed_columns = set(FILTER_COLUMNS)
        if child_column not in allowed_columns or not set(parent_columns) <= allowed_columns:
            raise ValueError("Unsupported BM25 relationship column")

        selected_columns = ", ".join((child_column, *parent_columns))
        required_values = " AND ".join(
            f"{column} IS NOT NULL AND TRIM({column}) <> ''"
            for column in (child_column, *parent_columns)
        )
        rows = self.connection.execute(
            f"SELECT DISTINCT {selected_columns} FROM products "
            f"WHERE {required_values}"
        ).fetchall()
        relationships: dict[str, set[tuple[str, ...]]] = {}
        for row in rows:
            normalized_child = " ".join(
                str(row[child_column]).casefold().split()
            )
            relationships.setdefault(normalized_child, set()).add(
                tuple(row[column] for column in parent_columns)
            )
        return {
            child: next(iter(parents))
            for child, parents in relationships.items()
            if len(parents) == 1
        }

    def subcategory_parent_index(self) -> dict[str, str]:
        relationships = self._unique_relationship_index(
            "subcategory_name",
            ("main_category_name",),
        )
        return {
            subcategory: parent[0]
            for subcategory, parent in relationships.items()
        }

    def city_state_index(self) -> dict[str, str]:
        relationships = self._unique_relationship_index(
            "city_name",
            ("state_name",),
        )
        return {
            city: state[0]
            for city, state in relationships.items()
        }

    def locality_location_index(self) -> dict[str, dict[str, str]]:
        relationships = self._unique_relationship_index(
            "locality_name",
            ("city_name", "state_name"),
        )
        return {
            locality: {"city": location[0], "state": location[1]}
            for locality, location in relationships.items()
        }

    def search(self, query: str, resolved_filters: dict, top_k: int) -> list[dict]:
        tokens = tokenize_query(query)
        if not tokens or top_k <= 0:
            return []

        match_query = " OR ".join(f'"{token}"' for token in tokens)
        conditions = ["products_fts MATCH ?"]
        params: list = [match_query]

        for column, value in resolved_filters["categorical"].items():
            if column not in FILTER_COLUMNS:
                continue
            conditions.append(f"p.{column} = ?")
            params.append(value)
        if "min_rental_fee" in resolved_filters:
            conditions.append("p.rental_fee > ?")
            params.append(UNPRICED_RENTAL_FEE_CEILING)
            conditions.append("p.rental_fee >= ?")
            params.append(resolved_filters["min_rental_fee"])
        if "max_rental_fee" in resolved_filters:
            if "min_rental_fee" not in resolved_filters:
                conditions.append("p.rental_fee > ?")
                params.append(UNPRICED_RENTAL_FEE_CEILING)
            conditions.append("p.rental_fee <= ?")
            params.append(resolved_filters["max_rental_fee"])

        params.append(top_k)
        rows = self.connection.execute(
            f"""
            SELECT p.doc_id, p.product_id, bm25(products_fts) AS rank
            FROM products_fts
            JOIN products AS p ON p.rowid = products_fts.rowid
            WHERE {' AND '.join(conditions)}
            ORDER BY rank
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "product_id": row["product_id"],
                "score": -float(row["rank"]),
            }
            for row in rows
        ]

    def browse(
        self,
        resolved_filters: dict,
        top_k: int,
        category_filters: dict[str, str] | None = None,
        exclude_doc_ids: set[str] | None = None,
        offset: int = 0,
        sort_order: str | None = None,
    ) -> list[dict]:
        """Return filtered category rows without requiring a keyword match."""
        if top_k <= 0 or offset < 0:
            return []
        if sort_order not in {None, "price_asc", "price_desc"}:
            raise ValueError(f"Unsupported browse sort order: {sort_order}")

        conditions = []
        params: list = []
        categorical = dict(resolved_filters.get("categorical", {}))
        for column, value in (category_filters or {}).items():
            if column not in FILTER_COLUMNS:
                continue
            existing = categorical.get(column)
            if existing is not None and existing != value:
                return []
            categorical[column] = value

        for column, value in categorical.items():
            if column not in FILTER_COLUMNS:
                continue
            conditions.append(f"{column} = ?")
            params.append(value)
        if "min_rental_fee" in resolved_filters:
            conditions.append("rental_fee > ?")
            params.append(UNPRICED_RENTAL_FEE_CEILING)
            conditions.append("rental_fee >= ?")
            params.append(resolved_filters["min_rental_fee"])
        if "max_rental_fee" in resolved_filters:
            if "min_rental_fee" not in resolved_filters:
                conditions.append("rental_fee > ?")
                params.append(UNPRICED_RENTAL_FEE_CEILING)
            conditions.append("rental_fee <= ?")
            params.append(resolved_filters["max_rental_fee"])

        excluded = sorted(exclude_doc_ids or set())
        if excluded:
            placeholders = ", ".join("?" for _ in excluded)
            conditions.append(f"doc_id NOT IN ({placeholders})")
            params.extend(excluded)

        where_clause = (
            f"WHERE {' AND '.join(conditions)}"
            if conditions
            else ""
        )
        order_clause = {
            "price_asc": (
                "CASE WHEN rental_fee IS NULL OR rental_fee <= ? "
                "THEN 1 ELSE 0 END, rental_fee IS NULL, "
                "rental_fee ASC, rowid DESC"
            ),
            "price_desc": (
                "CASE WHEN rental_fee IS NULL OR rental_fee <= ? "
                "THEN 1 ELSE 0 END, rental_fee IS NULL, "
                "rental_fee DESC, rowid DESC"
            ),
        }.get(sort_order, "rowid DESC")
        if sort_order in {"price_asc", "price_desc"}:
            params.append(UNPRICED_RENTAL_FEE_CEILING)
        params.extend((top_k, offset))
        rows = self.connection.execute(
            f"""
            SELECT doc_id, product_id
            FROM products
            {where_clause}
            ORDER BY {order_clause}
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "product_id": row["product_id"],
                "score": 0.0,
            }
            for row in rows
        ]
