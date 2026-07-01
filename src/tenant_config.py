from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from mysql_store import MySQLRuntimeConfig
from postgres_store import PostgresRuntimeConfig
from settings import PROJECT_ROOT


TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
DEFAULT_TENANT_CONFIG_DIR = PROJECT_ROOT / "configs" / "tenants"
GAINR_FILTER_FIELDS = {
    "main_category_name",
    "subcategory_name",
    "state_name",
    "city_name",
    "locality_name",
    "rental_duration",
    "rental_fee",
}


@dataclass(frozen=True)
class TenantStorageConfig:
    chroma_dir: Path
    collection_name: str
    bm25_path: Path
    vector_backend: str = "chroma"
    pgvector_database: PostgresRuntimeConfig | None = None
    pgvector_table: str = "search_vectors"
    vector_dimensions: int = 768


@dataclass(frozen=True)
class TenantRateLimit:
    requests_per_minute: int = 60
    burst: int = 10

    def __post_init__(self) -> None:
        if self.requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than zero")
        if self.burst <= 0:
            raise ValueError("rate-limit burst must be greater than zero")


@dataclass(frozen=True)
class TenantPayloadConfig:
    public_fields: tuple[str, ...]
    field_mapping: dict[str, str] = field(default_factory=dict)
    filter_schema: dict[str, str] = field(default_factory=dict)
    request_mapping: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TenantProfile:
    company_id: str
    database: MySQLRuntimeConfig | PostgresRuntimeConfig
    storage: TenantStorageConfig
    payload: TenantPayloadConfig
    rate_limit: TenantRateLimit
    planner_adapter: str
    api_key_envs: tuple[str, ...]
    config_path: Path
    endpoint_slug: str = ""
    planner_enabled: bool = True
    planner_prompt_context: str = ""


def validate_tenant_id(value: str) -> str:
    tenant_id = value.strip().casefold()
    if not TENANT_ID_RE.fullmatch(tenant_id):
        raise ValueError(
            f"Unsafe company id {value!r}; use lowercase letters, numbers, "
            "underscores, or hyphens."
        )
    return tenant_id


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid tenant YAML: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Tenant profile must contain a YAML object: {path}")
    return raw


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_name(section: dict[str, Any], key: str, default: str) -> str:
    value = str(section.get(key, default)).strip()
    if not value:
        raise ValueError(f"Environment-variable name {key!r} must not be empty")
    return value


def _env_value(
    section: dict[str, Any],
    key: str,
    default_env: str,
    *,
    default: str = "",
) -> str:
    return os.getenv(_env_name(section, key, default_env), default)


def _identifier(section: dict[str, Any], key: str, default: str) -> str:
    value = str(section.get(key, default)).strip()
    if not value or "\x00" in value:
        raise ValueError(f"Database identifier {key!r} must not be empty")
    return value


def load_tenant_profile(path: Path) -> TenantProfile:
    path = path.resolve()
    raw = _read_yaml(path)
    company = dict(raw.get("company", {}))
    company_id = validate_tenant_id(
        str(company.get("id", path.stem))
    )
    if company_id != path.stem:
        raise ValueError(
            f"Tenant profile {path.name} declares company.id={company_id!r}; "
            f"expected {path.stem!r}."
        )
    planner_adapter = str(company.get("planner_adapter", "gainr")).strip()
    if not planner_adapter:
        raise ValueError(f"Tenant {company_id!r} must configure planner_adapter")
    planner = dict(raw.get("planner", {}))
    planner_enabled = bool(planner.get("enabled", True))
    planner_prompt_context = str(
        planner.get("prompt_context", "")
    ).strip()
    if len(planner_prompt_context) > 4000:
        raise ValueError(
            f"Tenant {company_id!r} planner prompt context exceeds 4000 characters"
        )

    database = dict(raw.get("database", {}))
    backend = str(database.get("backend", "mysql")).strip().casefold()
    if backend not in {"mysql", "postgres"}:
        raise ValueError(
            f"Tenant {company_id!r} has unsupported database backend "
            f"{backend!r}."
        )
    default_prefix = "POSTGRES" if backend == "postgres" else "MYSQL"
    default_port = "5432" if backend == "postgres" else "3306"
    try:
        port = int(
            _env_value(
                database,
                "port_env",
                f"{default_prefix}_PORT",
                default=default_port,
            )
        )
    except ValueError as exc:
        raise ValueError(f"Tenant {company_id!r} has an invalid database port") from exc
    common_database_values = dict(
        host=_env_value(
            database,
            "host_env",
            f"{default_prefix}_HOST",
            default="localhost",
        ),
        port=port,
        database=_env_value(
            database,
            "database_env",
            f"{default_prefix}_DATABASE",
        ),
        user=_env_value(database, "user_env", f"{default_prefix}_USER"),
        password=_env_value(
            database,
            "password_env",
            f"{default_prefix}_PASSWORD",
        ),
        search_table=_identifier(database, "search_ready_table", "ads_search_ready"),
        content_column=_identifier(database, "content_column", "embedding_content"),
        bm25_column=_identifier(database, "bm25_column", "bm25_content"),
        search_id_column=_identifier(database, "search_id_column", "id"),
        result_table=_identifier(database, "result_table", "ads"),
        result_id_column=_identifier(database, "result_id_column", "id"),
        result_type_column=_identifier(database, "result_type_column", "type"),
    )
    mysql = (
        PostgresRuntimeConfig(
            **common_database_values,
            schema=_identifier(database, "schema", "public"),
        )
        if backend == "postgres"
        else MySQLRuntimeConfig(**common_database_values)
    )
    if not mysql.database:
        raise ValueError(
            f"Tenant {company_id!r} database environment variable is empty"
        )
    if not mysql.user:
        raise ValueError(f"Tenant {company_id!r} database user is empty")

    storage = dict(raw.get("storage", {}))
    vector_backend = str(
        storage.get("vector_backend", "chroma")
    ).strip().casefold()
    if vector_backend not in {"chroma", "pgvector"}:
        raise ValueError(
            f"Tenant {company_id!r} has unsupported vector backend "
            f"{vector_backend!r}."
        )
    collection_name = str(
        storage.get("collection_name", f"company_{company_id}")
    ).strip()
    if not TENANT_ID_RE.fullmatch(collection_name):
        raise ValueError(
            f"Tenant {company_id!r} has unsafe Chroma collection name "
            f"{collection_name!r}."
        )
    pgvector_database = None
    pgvector_table = "search_vectors"
    vector_dimensions = int(storage.get("vector_dimensions", 768))
    if vector_dimensions <= 0 or vector_dimensions > 2000:
        raise ValueError(
            f"Tenant {company_id!r} vector_dimensions must be between 1 and 2000"
        )
    if vector_backend == "pgvector":
        pgvector = dict(storage.get("pgvector", {}))
        use_company_database = bool(
            pgvector.get("use_company_database", backend == "postgres")
        )
        if use_company_database:
            if not isinstance(mysql, PostgresRuntimeConfig):
                raise ValueError(
                    f"Tenant {company_id!r} pgvector use_company_database "
                    "requires a PostgreSQL company database."
                )
            pgvector_database = mysql
        else:
            vector_prefix = "PGVECTOR"
            pgvector_database = PostgresRuntimeConfig(
                host=_env_value(
                    pgvector,
                    "host_env",
                    f"{vector_prefix}_HOST",
                    default="localhost",
                ),
                port=int(
                    _env_value(
                        pgvector,
                        "port_env",
                        f"{vector_prefix}_PORT",
                        default="5432",
                    )
                ),
                database=_env_value(
                    pgvector,
                    "database_env",
                    f"{vector_prefix}_DATABASE",
                ),
                user=_env_value(
                    pgvector,
                    "user_env",
                    f"{vector_prefix}_USER",
                ),
                password=_env_value(
                    pgvector,
                    "password_env",
                    f"{vector_prefix}_PASSWORD",
                ),
                schema=_identifier(pgvector, "schema", "public"),
            )
        pgvector_table = _identifier(
            pgvector,
            "table",
            f"{company_id}_search_vectors",
        )
        if not pgvector_database.database or not pgvector_database.user:
            raise ValueError(
                f"Tenant {company_id!r} pgvector database and user must be "
                "configured."
            )
    storage_config = TenantStorageConfig(
        chroma_dir=_path(storage.get("chroma_dir", "storage/chroma")),
        collection_name=collection_name,
        bm25_path=_path(
            storage.get(
                "bm25_path",
                f"storage/companies/{company_id}/bm25.sqlite3",
            )
        ),
        vector_backend=vector_backend,
        pgvector_database=pgvector_database,
        pgvector_table=pgvector_table,
        vector_dimensions=vector_dimensions,
    )

    payload = dict(raw.get("payload", {}))
    public_fields = tuple(
        str(value).strip()
        for value in payload.get("public_fields", ())
        if str(value).strip()
    )
    if mysql.result_id_column not in public_fields:
        raise ValueError(
            f"Tenant {company_id!r} payload.public_fields must include "
            f"result id column {mysql.result_id_column!r}."
        )
    field_mapping = {
        str(key).strip(): str(value).strip()
        for key, value in dict(payload.get("field_mapping", {})).items()
        if str(key).strip() and str(value).strip()
    }
    filter_schema = {
        str(key).strip(): str(value).strip().casefold()
        for key, value in dict(payload.get("filter_schema", {})).items()
        if str(key).strip() and str(value).strip()
    }
    allowed_filter_types = {"keyword", "number", "datetime", "boolean"}
    invalid_filter_types = sorted(
        {
            value
            for value in filter_schema.values()
            if value not in allowed_filter_types
        }
    )
    if invalid_filter_types:
        raise ValueError(
            f"Tenant {company_id!r} has unsupported filter types: "
            f"{invalid_filter_types}"
        )
    if planner_adapter == "gainr":
        unsupported_fields = sorted(set(filter_schema) - GAINR_FILTER_FIELDS)
        if unsupported_fields:
            raise ValueError(
                f"Tenant {company_id!r} gainr planner requires canonical "
                f"filter fields; unsupported: {unsupported_fields}"
            )
    request_mapping = {
        "query": "query",
        "cursor": "cursor",
        "page_size": "page_size",
    }
    request_mapping.update(
        {
            str(key).strip(): str(value).strip()
            for key, value in dict(
                payload.get("request_mapping", {})
            ).items()
            if str(key).strip() and str(value).strip()
        }
    )
    invalid_request_fields = sorted(
        set(request_mapping) - {"query", "cursor", "page_size"}
    )
    if invalid_request_fields:
        raise ValueError(
            f"Tenant {company_id!r} has unsupported request mappings: "
            f"{invalid_request_fields}"
        )
    if len(set(request_mapping.values())) != len(request_mapping):
        raise ValueError(
            f"Tenant {company_id!r} request payload fields must be unique."
        )

    rate = dict(raw.get("rate_limit", {}))
    rate_limit = TenantRateLimit(
        requests_per_minute=int(rate.get("requests_per_minute", 60)),
        burst=int(rate.get("burst", 10)),
    )
    api = dict(raw.get("api", {}))
    endpoint_slug = validate_tenant_id(
        str(api.get("endpoint_slug", company_id))
    )
    api_key_envs = tuple(
        str(value).strip()
        for value in api.get("key_envs", ())
        if str(value).strip()
    )
    if not api_key_envs:
        api_key_envs = (f"{company_id.upper()}_API_KEY",)

    return TenantProfile(
        company_id=company_id,
        database=mysql,
        storage=storage_config,
        payload=TenantPayloadConfig(
            public_fields=public_fields,
            field_mapping=field_mapping,
            filter_schema=filter_schema,
            request_mapping=request_mapping,
        ),
        rate_limit=rate_limit,
        planner_adapter=planner_adapter,
        api_key_envs=api_key_envs,
        config_path=path,
        endpoint_slug=endpoint_slug,
        planner_enabled=planner_enabled,
        planner_prompt_context=planner_prompt_context,
    )


def discover_tenant_profiles(
    directory: Path = DEFAULT_TENANT_CONFIG_DIR,
) -> dict[str, TenantProfile]:
    directory = _path(directory)
    if not directory.exists():
        return {}
    profiles: dict[str, TenantProfile] = {}
    for path in sorted(directory.glob("*.yaml")):
        profile = load_tenant_profile(path)
        if profile.company_id in profiles:
            raise ValueError(f"Duplicate tenant profile: {profile.company_id}")
        profiles[profile.company_id] = profile
    validate_tenant_isolation(profiles.values())
    return profiles


def validate_tenant_isolation(profiles: Iterable[TenantProfile]) -> None:
    collection_owners: dict[tuple[Path, str], str] = {}
    pgvector_owners: dict[tuple[str, int, str, str, str], str] = {}
    bm25_owners: dict[Path, str] = {}
    endpoint_owners: dict[str, str] = {}
    for profile in profiles:
        endpoint_slug = profile.endpoint_slug or profile.company_id
        endpoint_owner = endpoint_owners.get(endpoint_slug)
        if endpoint_owner is not None:
            raise ValueError(
                f"Tenants {endpoint_owner!r} and {profile.company_id!r} "
                f"share API endpoint slug {endpoint_slug!r}."
            )
        endpoint_owners[endpoint_slug] = profile.company_id
        if profile.storage.vector_backend == "chroma":
            collection_key = (
                profile.storage.chroma_dir.resolve(),
                profile.storage.collection_name,
            )
            if collection_key in collection_owners:
                raise ValueError(
                    f"Tenants {collection_owners[collection_key]!r} and "
                    f"{profile.company_id!r} share Chroma collection "
                    f"{collection_key[1]!r}."
                )
            collection_owners[collection_key] = profile.company_id
        else:
            database = profile.storage.pgvector_database
            if database is None:
                raise ValueError(
                    f"Tenant {profile.company_id!r} has no pgvector database."
                )
            vector_key = (
                database.host,
                database.port,
                database.database,
                database.schema,
                profile.storage.pgvector_table,
            )
            if vector_key in pgvector_owners:
                raise ValueError(
                    f"Tenants {pgvector_owners[vector_key]!r} and "
                    f"{profile.company_id!r} share pgvector table "
                    f"{vector_key[-1]!r}."
                )
            pgvector_owners[vector_key] = profile.company_id

        bm25_path = profile.storage.bm25_path.resolve()
        if bm25_path in bm25_owners:
            raise ValueError(
                f"Tenants {bm25_owners[bm25_path]!r} and "
                f"{profile.company_id!r} share BM25 path {bm25_path}."
            )
        bm25_owners[bm25_path] = profile.company_id


def api_key_digest(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


class TenantRegistry:
    def __init__(
        self,
        profiles: dict[str, TenantProfile],
        *,
        require_api_keys: bool = True,
        api_keys: dict[str, Iterable[str]] | None = None,
    ):
        if not profiles:
            raise ValueError("At least one tenant profile is required")
        validate_tenant_isolation(profiles.values())
        self._profiles = dict(profiles)
        self._endpoint_profiles = {
            (profile.endpoint_slug or profile.company_id): profile
            for profile in self._profiles.values()
        }
        self._key_owners: dict[str, str] = {}
        supplied = api_keys or {}
        for company_id, profile in self._profiles.items():
            values = [
                value.strip()
                for value in supplied.get(company_id, ())
                if value and value.strip()
            ]
            if not values:
                values = [
                    value.strip()
                    for name in profile.api_key_envs
                    if (value := os.getenv(name, "")).strip()
                ]
            if require_api_keys and not values:
                raise ValueError(
                    f"Tenant {company_id!r} has no configured API key; set one "
                    f"of: {', '.join(profile.api_key_envs)}"
                )
            for value in values:
                digest = api_key_digest(value)
                owner = self._key_owners.get(digest)
                if owner is not None:
                    raise ValueError(
                        f"Tenants {owner!r} and {company_id!r} share an API key"
                    )
                self._key_owners[digest] = company_id

    @property
    def profiles(self) -> dict[str, TenantProfile]:
        return dict(self._profiles)

    def get(self, company_id: str) -> TenantProfile:
        try:
            return self._profiles[validate_tenant_id(company_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown tenant {company_id!r}") from exc

    def resolve_api_key(self, api_key: str) -> TenantProfile | None:
        if not api_key:
            return None
        company_id = self._key_owners.get(api_key_digest(api_key))
        return self._profiles.get(company_id) if company_id else None

    def resolve_endpoint(self, endpoint_slug: str) -> TenantProfile | None:
        try:
            endpoint_slug = validate_tenant_id(endpoint_slug)
        except ValueError:
            return None
        return self._endpoint_profiles.get(endpoint_slug)


def load_tenant_registry(
    directory: Path = DEFAULT_TENANT_CONFIG_DIR,
    *,
    require_api_keys: bool = True,
) -> TenantRegistry:
    return TenantRegistry(
        discover_tenant_profiles(directory),
        require_api_keys=require_api_keys,
    )
