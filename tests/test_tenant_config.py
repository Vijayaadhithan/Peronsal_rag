import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tenant_config import (
    TenantRegistry,
    discover_tenant_profiles,
)
from postgres_store import PostgresRuntimeConfig
import vector_store


PROFILE = """
company:
  id: {company}
  planner_adapter: gainr
api:
  key_envs:
    - {key_env}
database:
  host_env: {prefix}_DB_HOST
  port_env: {prefix}_DB_PORT
  database_env: {prefix}_DB_NAME
  user_env: {prefix}_DB_USER
  password_env: {prefix}_DB_PASSWORD
  search_ready_table: search_ready
  content_column: embedding_content
  bm25_column: bm25_content
  search_id_column: id
  result_table: products
  result_id_column: id
storage:
  chroma_dir: {chroma_dir}
  collection_name: company_{company}
  bm25_path: {bm25_path}
payload:
  public_fields:
    - id
    - title
  filter_schema:
    subcategory_name: keyword
rate_limit:
  requests_per_minute: 30
  burst: 5
"""


def write_profile(
    tmp_path: Path,
    company: str,
    *,
    backend: str = "mysql",
    vector_backend: str = "chroma",
) -> None:
    prefix = company.upper()
    (tmp_path / f"{company}.yaml").write_text(
        PROFILE.replace(
            "database:\n",
            f"database:\n  backend: {backend}\n",
        ).replace(
            "storage:\n",
            (
                "storage:\n"
                f"  vector_backend: {vector_backend}\n"
                + (
                    "  vector_dimensions: 768\n"
                    "  pgvector:\n"
                    "    use_company_database: true\n"
                    "    table: alpha_vectors\n"
                    if vector_backend == "pgvector"
                    else ""
                )
            ),
        ).format(
            company=company,
            key_env=f"{prefix}_API_KEY",
            prefix=prefix,
            chroma_dir=tmp_path / "chroma",
            bm25_path=tmp_path / company / "bm25.sqlite3",
        ),
        encoding="utf-8",
    )


def set_database_environment(monkeypatch, company: str) -> None:
    prefix = company.upper()
    monkeypatch.setenv(f"{prefix}_DB_HOST", "localhost")
    monkeypatch.setenv(f"{prefix}_DB_PORT", "3306")
    monkeypatch.setenv(f"{prefix}_DB_NAME", f"db_{company}")
    monkeypatch.setenv(f"{prefix}_DB_USER", company)
    monkeypatch.setenv(f"{prefix}_DB_PASSWORD", "secret")


def test_tenant_profiles_resolve_separate_storage_and_api_keys(
    tmp_path,
    monkeypatch,
):
    for company in ("alpha", "beta"):
        write_profile(tmp_path, company)
        set_database_environment(monkeypatch, company)

    profiles = discover_tenant_profiles(tmp_path)
    registry = TenantRegistry(
        profiles,
        api_keys={"alpha": ["alpha-key"], "beta": ["beta-key"]},
    )

    assert profiles["alpha"].storage.collection_name == "company_alpha"
    assert profiles["alpha"].storage.bm25_path != profiles["beta"].storage.bm25_path
    assert profiles["alpha"].endpoint_slug == "alpha"
    assert profiles["alpha"].payload.request_mapping["query"] == "query"
    assert registry.resolve_api_key("alpha-key").company_id == "alpha"
    assert registry.resolve_api_key("beta-key").company_id == "beta"
    assert registry.resolve_api_key("wrong") is None


def test_tenant_registry_rejects_shared_api_key(tmp_path, monkeypatch):
    for company in ("alpha", "beta"):
        write_profile(tmp_path, company)
        set_database_environment(monkeypatch, company)
    profiles = discover_tenant_profiles(tmp_path)

    with pytest.raises(ValueError, match="share an API key"):
        TenantRegistry(
            profiles,
            api_keys={"alpha": ["same"], "beta": ["same"]},
        )


def test_postgres_company_profile_is_supported(tmp_path, monkeypatch):
    write_profile(
        tmp_path,
        "alpha",
        backend="postgres",
        vector_backend="pgvector",
    )
    set_database_environment(monkeypatch, "alpha")
    monkeypatch.setenv("ALPHA_DB_PORT", "5432")

    profile = discover_tenant_profiles(tmp_path)["alpha"]

    assert isinstance(profile.database, PostgresRuntimeConfig)
    assert profile.database.port == 5432
    assert profile.database.schema == "public"
    assert profile.database.result_type_column == "type"
    assert profile.storage.vector_backend == "pgvector"
    assert profile.storage.pgvector_database is profile.database
    assert profile.storage.pgvector_table == "alpha_vectors"


def test_pgvector_profile_selects_pgvector_collection(
    tmp_path,
    monkeypatch,
):
    write_profile(
        tmp_path,
        "alpha",
        backend="postgres",
        vector_backend="pgvector",
    )
    set_database_environment(monkeypatch, "alpha")
    monkeypatch.setenv("ALPHA_DB_PORT", "5432")
    profile = discover_tenant_profiles(tmp_path)["alpha"]
    captured = {}

    class FakePgVectorCollection:
        def __init__(self, database, table, dimensions, *, create=False):
            captured.update(
                database=database,
                table=table,
                dimensions=dimensions,
                create=create,
            )

    monkeypatch.setattr(
        vector_store,
        "PgVectorCollection",
        FakePgVectorCollection,
    )

    result = vector_store.get_tenant_vector_collection(profile, create=True)

    assert isinstance(result, FakePgVectorCollection)
    assert captured["database"] is profile.database
    assert captured["table"] == "alpha_vectors"
    assert captured["dimensions"] == 768
    assert captured["create"] is True
