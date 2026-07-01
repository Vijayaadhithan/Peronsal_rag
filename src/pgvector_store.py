from __future__ import annotations

import json
from typing import Any

from postgres_store import (
    PostgresRuntimeConfig,
    postgres_connection,
    qualified_table,
    quote_postgres_identifier,
)


class PgVectorCollection:
    """Chroma-compatible subset backed by one isolated pgvector table."""

    def __init__(
        self,
        config: PostgresRuntimeConfig,
        table: str,
        dimensions: int,
        *,
        create: bool = False,
    ):
        self.config = config
        self.table = table
        self.dimensions = dimensions
        if create:
            self.initialize()
        else:
            self._require_table()

    @property
    def name(self) -> str:
        return self.table

    def _qualified(self) -> str:
        return qualified_table(self.config, self.table)

    def _require_table(self) -> None:
        with postgres_connection(self.config, dict_rows=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT to_regclass(%s) AS table_name
                    """,
                    (f"{self.config.schema}.{self.table}",),
                )
                if cursor.fetchone()["table_name"] is None:
                    raise RuntimeError(
                        f"pgvector table {self.config.schema}.{self.table} "
                        "does not exist. Run tenant ingestion first."
                    )

    def initialize(self) -> None:
        index_name = f"{self.table}_embedding_hnsw"
        source_index_name = f"{self.table}_source_file"
        with postgres_connection(self.config) as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except Exception as exc:
                    raise RuntimeError(
                        "The PostgreSQL vector extension is unavailable or the "
                        "configured user cannot enable it. Install pgvector and "
                        "run CREATE EXTENSION vector as a database administrator."
                    ) from exc
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified()} (
                        id TEXT PRIMARY KEY,
                        document TEXT NOT NULL,
                        embedding vector({self.dimensions}) NOT NULL,
                        metadata JSONB NOT NULL
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS
                    {quote_postgres_identifier(index_name)}
                    ON {self._qualified()}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS
                    {quote_postgres_identifier(source_index_name)}
                    ON {self._qualified()} ((metadata ->> 'source_file'))
                    """
                )

    @staticmethod
    def _vector_literal(vector: list[float]) -> str:
        return json.dumps([float(value) for value in vector], separators=(",", ":"))

    def count(self) -> int:
        with postgres_connection(self.config, dict_rows=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT COUNT(*) AS row_count FROM {self._qualified()}"
                )
                return int(cursor.fetchone()["row_count"])

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not (
            len(ids)
            == len(documents)
            == len(embeddings)
            == len(metadatas)
        ):
            raise ValueError("pgvector upsert arrays must have equal lengths")
        rows = []
        for doc_id, document, embedding, metadata in zip(
            ids,
            documents,
            embeddings,
            metadatas,
        ):
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} embedding dimensions, "
                    f"received {len(embedding)}."
                )
            rows.append(
                (
                    str(doc_id),
                    document,
                    self._vector_literal(embedding),
                    json.dumps(metadata, ensure_ascii=False, default=str),
                )
            )
        if not rows:
            return
        with postgres_connection(self.config) as connection:
            with connection.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self._qualified()}
                        (id, document, embedding, metadata)
                    VALUES (%s, %s, %s::vector, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        document = EXCLUDED.document,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    rows,
                )

    def get(
        self,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, list]:
        include = include or []
        conditions = []
        params: list[Any] = []
        if ids is not None:
            if not ids:
                return {"ids": [], "documents": [], "metadatas": []}
            placeholders = ", ".join(["%s"] * len(ids))
            conditions.append(f"id IN ({placeholders})")
            params.extend(str(value) for value in ids)
        for key, value in (where or {}).items():
            conditions.append("metadata ->> %s = %s")
            params.extend((str(key), str(value)))
        where_clause = (
            f"WHERE {' AND '.join(conditions)}" if conditions else ""
        )
        selected = ["id"]
        if "documents" in include:
            selected.append("document")
        if "metadatas" in include:
            selected.append("metadata")
        with postgres_connection(self.config, dict_rows=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"SELECT {', '.join(selected)} FROM {self._qualified()} "
                    f"{where_clause}",
                    params,
                )
                rows = cursor.fetchall()
        rows_by_id = {str(row["id"]): row for row in rows}
        ordered_rows = (
            [rows_by_id[str(value)] for value in ids if str(value) in rows_by_id]
            if ids is not None
            else rows
        )
        result = {"ids": [str(row["id"]) for row in ordered_rows]}
        if "documents" in include:
            result["documents"] = [row["document"] for row in ordered_rows]
        if "metadatas" in include:
            result["metadatas"] = [row["metadata"] for row in ordered_rows]
        return result

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
    ) -> dict[str, list[list]]:
        include = include or []
        all_ids = []
        all_documents = []
        all_metadatas = []
        all_distances = []
        with postgres_connection(self.config, dict_rows=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SET hnsw.iterative_scan = strict_order")
                for embedding in query_embeddings:
                    if len(embedding) != self.dimensions:
                        raise ValueError(
                            f"Expected {self.dimensions} query dimensions, "
                            f"received {len(embedding)}."
                        )
                    cursor.execute(
                        f"""
                        SELECT id, document, metadata,
                               embedding <=> %s::vector AS distance
                        FROM {self._qualified()}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s
                        """,
                        (
                            self._vector_literal(embedding),
                            self._vector_literal(embedding),
                            n_results,
                        ),
                    )
                    rows = cursor.fetchall()
                    all_ids.append([str(row["id"]) for row in rows])
                    all_documents.append([row["document"] for row in rows])
                    all_metadatas.append([row["metadata"] for row in rows])
                    all_distances.append(
                        [float(row["distance"]) for row in rows]
                    )
        result = {"ids": all_ids}
        if "documents" in include:
            result["documents"] = all_documents
        if "metadatas" in include:
            result["metadatas"] = all_metadatas
        if "distances" in include:
            result["distances"] = all_distances
        return result

    def delete(
        self,
        *,
        ids: list[str] | None = None,
        where: dict[str, Any] | None = None,
    ) -> None:
        conditions = []
        params: list[Any] = []
        if ids is not None:
            if not ids:
                return
            placeholders = ", ".join(["%s"] * len(ids))
            conditions.append(f"id IN ({placeholders})")
            params.extend(str(value) for value in ids)
        for key, value in (where or {}).items():
            conditions.append("metadata ->> %s = %s")
            params.extend((str(key), str(value)))
        if not conditions:
            raise ValueError("Refusing to delete pgvector rows without a selector")
        with postgres_connection(self.config) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._qualified()} "
                    f"WHERE {' AND '.join(conditions)}",
                    params,
                )
