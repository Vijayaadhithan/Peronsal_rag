import re
import sqlite3
from pathlib import Path

from settings import BM25_INDEX_PATH

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
        self.connection = sqlite3.connect(self.path)
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

    def clear(self) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM products")

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
            conditions.append("p.rental_fee >= ?")
            params.append(resolved_filters["min_rental_fee"])
        if "max_rental_fee" in resolved_filters:
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
