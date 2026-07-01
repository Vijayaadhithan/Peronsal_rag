# Semantic Advertisement Search

Tenant-isolated semantic search for advertisement and ecommerce catalogs. The
same code supports local single-company development and authenticated
multi-company production operation.

The system is designed for users who know what they need but may not know the
product, service, category, brand, or model name. It searches advertisement
titles, descriptions, keywords, taxonomy, attributes, prices, locations, and
rental metadata, then returns canonical rows from that company's MySQL or
PostgreSQL database.

Example:

```text
something portable that can record a wedding clearly from far away
```

This can retrieve cameras, drone cameras, portable recorders, microphones,
gimbals, and related recording equipment without forcing the user to name one
specific product category.

## Production Runbook

This section is the ordered operational path. Run commands from the repository
root.

### A. First Gainr deployment

#### 1. Install the host

Current macOS development machine:

```bash
./scripts/bootstrap_macos.sh
```

Ubuntu with Chroma:

```bash
./scripts/bootstrap_ubuntu.sh
```

Ubuntu when a tenant will use pgvector:

```bash
INSTALL_PGVECTOR=1 ./scripts/bootstrap_ubuntu.sh
```

The bootstrap installs Redis, Ollama, system libraries and Python packages,
creates `.venv`, pulls `embeddinggemma:latest`, and runs the unit tests. It does
not create company databases or import company data.

#### 2. Configure `.env`

If `.env` does not exist:

```bash
cp .env.example .env
```

Set the real values without committing the file:

```text
GEMINI_API_KEY=<secret>
JINA_API_KEY=<secret>
VOYAGE_API_KEY=<secret>

MYSQL_HOST=<gainr-db-host>
MYSQL_PORT=3306
MYSQL_DATABASE=<gainr-database>
MYSQL_USER=<read-only-user>
MYSQL_PASSWORD=<secret>

REDIS_URL=redis://127.0.0.1:6379/0
API_AUTH_ENABLED=false
GAINR_API_KEY=
```

Keep authentication disabled only until the Gainr indexes and company key are
ready. `.env` is ignored by Git.

#### 3. Verify Redis and Ollama

```bash
redis-cli ping
ollama list
```

Expected:

```text
PONG
embeddinggemma:latest
```

If the embedding model is missing:

```bash
ollama pull embeddinggemma:latest
```

#### 4. Review the Gainr profile

Gainr is configured in `configs/tenants/gainr.yaml`. Confirm:

- `company.id: gainr`
- `api.endpoint_slug: gainr`
- MySQL table/column mappings
- `storage.collection_name: company_gainr`
- `storage.bm25_path: storage/companies/gainr/bm25.sqlite3`
- public response fields
- request mapping and rate limits
- `planner.enabled` and `planner.prompt_context`

Do not reuse another company's collection, pgvector table, BM25 path, endpoint
slug or API key.

#### 5. Validate the source database without writing indexes

The company's canonical `search_ready` table must already exist. For the first
Gainr publish from RAG_HT:

```bash
cd /path/to/RAG_HT
./scripts/setup.sh
./scripts/run_scheduled_etl.sh gainr --publish
cd /path/to/Peronsal_rag
```

RAG_HT reads the upstream source database, preprocesses and validates the
company data, then atomically publishes the destination table. Return to this
repository and validate that published table:

```bash
.venv/bin/python src/ingest.py \
  --company gainr \
  --mysql \
  --check \
  --limit 10
```

This is read-only. It validates credentials, tables, content columns and the
primary key.

#### 6. Build Gainr's isolated indexes

```bash
.venv/bin/python src/ingest.py \
  --company gainr \
  --mysql \
  --mysql-batch-size 500 \
  --embed-batch-size 32
```

This reads Gainr's `search_ready` rows, embeds changed/new rows, writes only
`company_gainr`, and builds only Gainr's BM25 file. It does not update the
company database.

Check the result:

```bash
.venv/bin/python src/ingest.py --company gainr --list
.venv/bin/python scripts/doctor.py --company gainr --strict
```

#### 7. Generate and install the Gainr API key

```bash
.venv/bin/python scripts/generate_company_api_key.py --company gainr
```

The command shows the key once. Store it in a secret manager, securely provide
the same key to Gainr, and set:

```text
GAINR_API_KEY=<generated-company-key>
API_AUTH_ENABLED=true
API_RATE_LIMIT_ENABLED=true
```

Never use the Gemini, Jina or Voyage provider keys as a company API key.

#### 8. Start the API

```bash
.venv/bin/python src/run_api.py
```

Keep this terminal running. On a server, place the same command behind
systemd and a TLS reverse proxy.

#### 9. Verify readiness, authentication and indexes

No company secret is needed for process readiness:

```bash
curl http://127.0.0.1:8000/api/v1/ready
```

Verify that the key belongs to the Gainr endpoint:

```bash
curl http://127.0.0.1:8000/api/v1/gainr/auth/verify \
  -H 'X-API-Key: <GAINR_API_KEY>'
```

Verify Gainr's search dependencies:

```bash
curl http://127.0.0.1:8000/api/v1/gainr/health \
  -H 'X-API-Key: <GAINR_API_KEY>'
```

#### 10. Test deterministic search

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gainr/search \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <GAINR_API_KEY>' \
  -d '{"query":"bike in Chennai under 1000","page_size":20}'
```

Expected behavior:

- `execution_path` is `deterministic_filter`
- model token usage is zero
- category/location/price filters are applied
- current rows are fetched from Gainr's result table

#### 11. Test semantic search

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gainr/search \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <GAINR_API_KEY>' \
  -d '{"query":"portable equipment for recording a distant wedding","page_size":20}'
```

Expected behavior:

- hosted query planning runs when the plan is not cached
- vector and BM25 retrieval run in parallel
- Jina reranks first; Voyage models are fallbacks
- `usage` contains this request's provider-reported tokens
- final current rows come from Gainr's database

#### 12. Request the next page

Copy `pagination.next_cursor` from the first response:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gainr/search \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <GAINR_API_KEY>' \
  -d '{"cursor":"<NEXT_CURSOR>","page_size":20}'
```

The cursor is company-bound. It cannot be reused with another company's key.

#### 13. Check monthly usage

```bash
MONTH=$(date -u +%Y-%m)
curl "http://127.0.0.1:8000/api/v1/gainr/usage?month=${MONTH}" \
  -H 'X-API-Key: <GAINR_API_KEY>'
```

The response separates searches, model requests, input tokens, output tokens
and total tokens by provider/model. Deterministic and result-cache hits record
zero model tokens.

### B. Routine Gainr refresh

First let RAG_HT preprocess and atomically publish the newest Gainr
`search_ready` rows:

```bash
cd /path/to/RAG_HT
./scripts/run_scheduled_etl.sh gainr --publish
cd /path/to/Peronsal_rag
```

The guarded RAG_HT command prevents overlapping company ETL runs and
automatically chooses incremental processing when a valid baseline exists.
Then refresh Gainr's retrieval indexes:

```bash
.venv/bin/python src/ingest.py \
  --company gainr \
  --mysql \
  --mysql-batch-size 500 \
  --embed-batch-size 32
```

The command scans the configured source but embeds only changed/new content;
unchanged hashes are skipped.

Rebuild only BM25 without embedding:

```bash
.venv/bin/python src/ingest.py \
  --company gainr \
  --mysql-bm25-only \
  --mysql-batch-size 5000
```

Force every row through embeddings again:

```bash
.venv/bin/python src/ingest.py \
  --company gainr \
  --mysql \
  --mysql-force-reembed
```

When the database snapshot has authoritative deletions, schedule a maintenance
rebuild:

```bash
.venv/bin/python src/ingest.py \
  --company gainr \
  --mysql \
  --mysql-replace-source
```

`--mysql-replace-source` clears only Gainr's configured vector source and BM25
file before rebuilding. It never deletes source database rows. Because the
current replacement is not an atomic generation swap, run it during a
maintenance window.

After any refresh:

```bash
.venv/bin/python scripts/doctor.py --company gainr --strict
```

### C. Onboard another company

#### 1. Create its profile

```bash
cp configs/tenants/example-company.yaml.example \
  configs/tenants/acme.yaml
```

Edit `configs/tenants/acme.yaml` so `company.id` exactly matches `acme`.
Configure:

- unique `api.endpoint_slug`
- unique API-key environment-variable name
- MySQL or PostgreSQL source/result tables
- Chroma or pgvector storage
- unique BM25 path
- canonical filter mapping
- request-field mapping
- public result fields and response-field mapping
- company rate limit
- planner enablement and company prompt context

#### 2. Add database credentials to the server environment

PostgreSQL example:

```text
ACME_POSTGRES_HOST=<host>
ACME_POSTGRES_PORT=5432
ACME_POSTGRES_DATABASE=<database>
ACME_POSTGRES_USER=<read-only-user>
ACME_POSTGRES_PASSWORD=<secret>
```

The YAML profile declares which environment-variable names to read.

#### 3. Prepare pgvector only when selected

Run once in the configured vector database as an authorized database user:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Chroma tenants do not need this.

When `pgvector.use_company_database: true`, the configured PostgreSQL user also
needs create/read/write privileges for that tenant's vector table/schema. To
keep the source/result tables strictly read-only, set
`use_company_database: false` and configure separate `PGVECTOR_*` credentials
for a dedicated vector database.

#### 4. Validate and ingest only that company

```bash
.venv/bin/python src/ingest.py \
  --company acme \
  --mysql \
  --check \
  --limit 10

.venv/bin/python src/ingest.py \
  --company acme \
  --mysql \
  --mysql-batch-size 500 \
  --embed-batch-size 32

.venv/bin/python scripts/doctor.py --company acme --strict
```

The `--mysql` flag is the backward-compatible database-ingestion flag and also
works when the selected profile uses PostgreSQL.

#### 5. Generate, store and verify its key

```bash
.venv/bin/python scripts/generate_company_api_key.py --company acme
```

Set the environment variable named by `api.key_envs`, restart the API so it
reloads profiles/secrets, then verify:

```bash
curl http://127.0.0.1:8000/api/v1/acme/auth/verify \
  -H 'X-API-Key: <ACME_API_KEY>'
```

#### 6. Test its configured payload

If Acme maps `query` to `search_text` and `page_size` to `limit`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/acme/search \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: <ACME_API_KEY>' \
  -d '{"search_text":"portable camera","limit":20}'
```

### D. Rotate a company API key

Declare both environment-variable names temporarily:

```yaml
api:
  key_envs:
    - GAINR_API_KEY_CURRENT
    - GAINR_API_KEY_NEXT
```

Then:

1. Generate and store the next key.
2. Restart the API with both environment variables populated.
3. Verify the new key with `/api/v1/gainr/auth/verify`.
4. Move the client to the new key.
5. Remove the old environment variable from the profile and secret manager.
6. Restart the API again.

API keys cannot be shared across company profiles; startup fails if a duplicate
is detected.

### E. Operational troubleshooting

- `401 Missing API key`: send the `X-API-Key` header.
- `401 Invalid API key`: check the configured secret and restart after changes.
- `403 API key does not match`: the key belongs to another company endpoint.
- `404 Unknown company endpoint`: check `api.endpoint_slug` and restart.
- `429 Company rate limit exceeded`: inspect that tenant's rate policy.
- `503` with missing vector/BM25 index: run company ingestion and the doctor.
- Ollama failure: run `ollama list`, then pull `embeddinggemma:latest`.
- Redis failure: run `redis-cli ping`; search falls back to process memory.
- PostgreSQL pgvector failure: verify the extension, schema, table permissions
  and configured embedding dimension.

Run the full local verification suite:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests scripts
```

## Current Stack

- MySQL or PostgreSQL: company source and canonical advertisement records
- Ollama: local embeddings only
- Gemini API: hosted structured query extraction
- Chroma or PostgreSQL/pgvector: one isolated vector store per company
- SQLite FTS5: one persistent BM25 keyword index per company
- Jina `jina-reranker-v2-base-multilingual`: primary hosted reranker
- Voyage `rerank-2.5`: first hosted fallback, limited locally to 3 RPM
- Voyage `rerank-2.5-lite`: second hosted fallback, limited locally to 3 RPM
- Python: ingestion, retrieval, evaluation, and CLI orchestration

Defaults are configured in `config.yaml`:

- Local compatibility collection: `local_data`
- Production Gainr collection: `company_gainr`
- Embedding model: `embeddinggemma:latest`
- Query models, in fallback order: `gemini-3.1-flash-lite`,
  `gemma-4-26b-a4b-it`, `gemma-4-31b-it`
- Deterministic fast path: enabled
- Normalized query-plan cache: 500 entries for 15 minutes
- Rerank order: Jina -> Voyage 2.5 -> Voyage 2.5 Lite
- Vector candidates: 30
- BM25 candidates: 30
- API reranker candidates: 60
- API primary ranked results: 60
- API related tail: up to 140
- API combined result window: 200
- CLI final results: 10

## Search Design

The system separates hard constraints from semantic intent.

### Hard Filters

Hard filters are applied only when they are explicitly requested or safely
derived from a unique database relationship:

- Offer ads versus wanted ads
- State, city, and locality
- Rental duration
- Minimum and maximum rental fee
- Explicit main category or subcategory

Advertisement intent uses the canonical `ads.type` field:

- `type = 1`: offer advertisement
- `type = 2`: wanted/request advertisement

`I need a bike` means the searcher wants an available offer, so results must be
`type = 1`. `Someone looking for bikes` asks for another person's request, so
results must be `type = 2`. The deterministic planner enforces this perspective
after LLM extraction. `ads.status` is a separate lifecycle field and must not
be used to infer offer versus wanted intent.

The same perspective applies to services:

- `I need a wedding photographer` -> `offer` (`type = 1`): show providers.
- `find photographers available for hire` -> `offer` (`type = 1`).
- `show people looking for wedding photographers` -> `wanted` (`type = 2`):
  show customer/request ads.
- `find customers who need photography services` -> `wanted` (`type = 2`).

For example, `Quad Bike in Bangalore per hour` can safely resolve to:

```json
{
  "main_category": "Sports & Toys",
  "subcategory": "Quad Bike",
  "state": "Karnataka",
  "city": "Bengaluru",
  "rental_duration": "Per Hour"
}
```

### Soft Category Hints

When a category is inferred from functionality rather than stated by the user,
it becomes a ranking hint instead of a hard filter.

For example:

```text
something portable that records distant subjects
```

The model may infer `Camera`, but Camera is not used to eliminate recorders,
drone cameras, microphones, or other potentially relevant products.

### DB-Backed Resolution

The persistent BM25 product table is also used to build safe relationship maps:

- Subcategory to main category
- City to state
- Locality to city and state

A parent is derived only when the indexed relationship is unique. Ambiguous
subcategories such as `Technician`, `Consultant`, or `Others` do not force an
incorrect main category.

The planner also handles:

- Singular and plural category names
- Common city aliases such as Bangalore/Bengaluru and Bombay/Mumbai
- Conservative typo recovery such as Coimbtore/Coimbatore
- Hourly, daily, weekly, monthly, and per-ride language
- Price ranges and upper/lower budgets
- Searcher intent versus wanted-ad intent

### Deterministic Fast Path

Simple explicit catalog queries bypass hosted query extraction, embeddings,
BM25 relevance search, and cross-encoder reranking. Examples include:

```text
bike
bikes in Chennai under 1000
1000 rent car
car rent 1000 in Chennai
camera per day
someone looking for bikes
bke
bkes in chni under 1000
```

The fast path requires an explicit indexed main category or subcategory. Every
remaining term must be explainable as a validated location, duration, price,
ad intent, or harmless request word. Descriptive attributes keep the semantic
pipeline active:

```text
red bike with ABS
portable camera for distant subjects
vehicle for recreational driving on rough terrain
```

High-confidence, unique typo matches are corrected against the actual indexed
taxonomy and locations before fast-path selection. For example, `bke` maps to
`Bike`, while `chni` after `in` maps to `Chennai`. The API reports these
decisions in `interpreted_query.query_corrections`. If two catalog values are
similarly close, no correction is forced and the semantic path remains
available. Set `query_extraction.fuzzy_matching: false` to disable this.

Compact queries are order-independent. When a simple validated category query
contains exactly one standalone amount, the fast path treats it as the maximum
rental budget. Values that look like quantities, model years, or specifications
such as `2 cars`, `2020 car`, and `1000 cc car` are not treated as budgets.

Normalized query plans are cached for 15 minutes with a 500-entry LRU bound.
Case and repeated whitespace do not create different cache keys. The cache
skips only planning; semantic retrieval and reranking still run for a repeated
descriptive query.

## End-to-End Search Flow

1. Normalize the query and test conservative spelling corrections against
   indexed taxonomy and filter values.
2. Test the deterministic grammar; if it does not match, check the normalized
   query-plan cache.
3. Simple explicit queries browse matching products directly and stop here.
4. Descriptive queries use the first available configured Google model to
   produce semantic and keyword queries plus possible constraints.
5. Deterministic validation corrects duration, price, ad type, taxonomy,
   aliases, spelling, and hierarchy relationships.
6. Explicit constraints become hard filters. Guessed categories become soft
   hints.
7. Chroma or pgvector search and SQLite FTS5/BM25 retrieve primary candidates
   concurrently.
8. Reciprocal Rank Fusion combines their ranks, and `ads.type` removes
   offer/wanted mismatches.
9. Jina orders the 60 primary candidates. Voyage `rerank-2.5` and then
   `rerank-2.5-lite` are tried on failure or provider throttling.
10. A stable related tail is selected using whichever validated category,
    location, duration, or price filters are available.
11. Full canonical database records are returned in primary-then-related order.

The extraction model never generates product records. Returned data always
comes from the configured company database.

## Architecture Boundaries

`ProductSearchEngine` in `src/search_engine.py` exposes the complete:

```text
cache/fast-plan -> filter browse
                -> semantic retrieve -> rerank -> append filtered tail
```

flow. The CLI and evaluation tools use this same implementation.

Provider protocols are defined in `src/providers.py`:

- `EmbeddingProvider`
- `StructuredQueryProvider`
- `RerankingProvider`

Ollama provides embeddings and the Google Generative Language API provides
schema-constrained query plans using Gemma or Gemini models. The reranker chain
uses the same boundary, so filtering, retrieval, fusion, evaluation, and
database mapping remain provider-independent.

## Multi-Company Production Boundary

`RAG_HT` reads and preprocesses each company's source data and publishes that
company's canonical `search_ready` table. This repository embeds and searches
that table.

Company profiles live under `configs/tenants/`. Each profile owns:

- database backend, credential environment-variable names, and table mappings
- either a unique Chroma collection or unique pgvector table
- a unique SQLite BM25 path
- public response fields and optional field renaming
- declared filter fields
- per-minute and burst rate limits

Each profile exposes a separate configurable company endpoint such as
`/api/v1/gainr/search`. `X-API-Key` must resolve to the same company as the
endpoint; an Alpha key cannot call the Gainr endpoint. Request field names can
also be mapped per company while the internal query/cursor/page-size contract
and search flow remain unchanged. Search sessions, document IDs, Redis
namespaces, vector stores, BM25 files, and database connections are
company-bound. A cursor created for one company is invalid for another company.

Enable tenant mode:

```text
API_AUTH_ENABLED=true
API_RATE_LIMIT_ENABLED=true
GAINR_API_KEY=replace-with-a-long-random-secret
```

Generate a new company credential:

```bash
.venv/bin/python scripts/generate_company_api_key.py --company gainr
```

The secret is displayed once. Store it in the server secret manager under one
of the profile's `api.key_envs`, deliver it securely to the company, and never
store it in source control. The client proves possession on every request with
`X-API-Key`; the server hashes the received key, resolves its company, and also
checks that it matches the URL's endpoint slug. Verify onboarding without
opening the search indexes:

```bash
curl http://127.0.0.1:8000/api/v1/gainr/auth/verify \
  -H 'X-API-Key: GAINR_API_KEY'
```

Validate one company's database and isolated BM25 index:

```bash
.venv/bin/python scripts/doctor.py --company gainr --strict
```

Add a company by copying `configs/tenants/gainr.yaml` and changing the company
ID, endpoint slug, request-field mapping, secret environment-variable names,
database mapping, storage names, public response payload, and rate policy.
Endpoint slugs, Chroma collections, pgvector tables, and BM25 paths cannot be
shared. `configs/tenants/example-company.yaml.example` demonstrates a different
endpoint/request payload together with PostgreSQL and pgvector.

Company source payloads may be completely different, but their `RAG_HT`
adapter must map searchable filter concepts into the canonical planner fields
used here (`main_category_name`, `subcategory_name`, location fields,
`rental_duration`, and `rental_fee`). Public result fields remain freely
configurable through `payload.public_fields` and `payload.field_mapping`.
`planner.enabled` controls hosted query planning for that tenant, while
`planner.prompt_context` adds bounded company-specific domain instructions.
Deterministic rules always run first and remain common to every tenant that
maps its data to the canonical filter contract.

One embedding provider and one reranker chain are shared across all companies.
Tenant engines are opened lazily and bounded by
`API_TENANT_ENGINE_CACHE_SIZE`, which keeps memory bounded on the 16 GB host.
Different company engines can process requests concurrently. Within each
semantic search, vector and BM25 retrieval also run concurrently and retain the
existing RRF, ad-intent filtering, reranking, and canonical-row fetch flow.
Each company can run up to `API_TENANT_MAX_CONCURRENT_SEARCHES` searches in
parallel (default 4); excess requests wait for a slot and remain subject to
that company's API rate policy.

Use Chroma when the priority is the cheapest and simplest single-host
deployment. Use pgvector when PostgreSQL is already operated for that company
and database-managed backup, replication, and SQL visibility are worth the
additional index memory and operations. Both preserve a hard per-company
storage boundary. OpenSearch is not required.

## Setup

### One-command macOS bootstrap

On a new Mac with Homebrew installed:

```bash
./scripts/bootstrap_macos.sh
```

This installs and starts Ollama and Redis, creates `.venv`, installs Python
dependencies, creates `.env` from `.env.example` when needed, pulls
`embeddinggemma:latest`, prefetches the ModernBERT reranker, and runs the
environment doctor.

MySQL is required for the populated `ads_search_ready` and `ads` tables. To
also install and start an empty local MySQL server, use:

```bash
INSTALL_MYSQL=1 ./scripts/bootstrap_macos.sh
```

Installing MySQL does not create or populate the application tables. To skip
the model download during a lightweight setup, set `SKIP_MODEL_PREFETCH=1`.
Rerun all infrastructure checks at any time with:

```bash
.venv/bin/python scripts/doctor.py --strict
```

### One-command Ubuntu bootstrap

On Ubuntu:

```bash
./scripts/bootstrap_ubuntu.sh
```

The script installs the non-database runtime requirements: Redis, Ollama,
compiler/build tools, `libgomp1`, `libpq-dev`, Git, curl, and certificate
packages. It creates `.venv`, installs runtime and test dependencies, pulls the
embedding model, and runs unit tests. The unused local reranker is not
downloaded unless `SKIP_LOCAL_RERANKER=0` is explicitly supplied.

If this host will store vectors in PostgreSQL, install pgvector too:

```bash
INSTALL_PGVECTOR=1 ./scripts/bootstrap_ubuntu.sh
```

Useful installation switches:

```text
SKIP_LOCAL_RERANKER=0  Also download the optional local fallback model
SKIP_TESTS=1           Install runtime dependencies only
INSTALL_OLLAMA=0       Use an Ollama service installed elsewhere
PYTHON_BIN=python3.12  Select a specific Python executable
```

Beyond Python and the company database, the production host needs Redis,
Ollama, the OS libraries installed by the script, and pgvector only when
`storage.vector_backend: pgvector`. No OpenSearch, Neo4j, Java, or separate
vector service is needed. Keep one API worker on a 16 GB CPU host when local
reranking is enabled; the hosted primary path avoids loading local model
weights during normal operation.

### Manual setup

Create `.env`:

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_KEEP_ALIVE=-1
GEMINI_API_KEY=your-gemini-api-key
GEMINI_API_BASE_URL=https://generativelanguage.googleapis.com/v1beta
GEMINI_TIMEOUT_SECONDS=10

REDIS_ENABLED=true
REDIS_URL=redis://127.0.0.1:6379/0
REDIS_KEY_PREFIX=semantic_ads
REDIS_RESULT_CACHE_ENABLED=true
REDIS_RESULT_CACHE_TTL_SECONDS=300

API_AUTH_ENABLED=false
API_RATE_LIMIT_ENABLED=true
API_TENANT_CONFIG_DIR=configs/tenants
API_TENANT_ENGINE_CACHE_SIZE=8
GAINR_API_KEY=

# Primary followed by two independently throttled Voyage fallbacks.
RERANK_PROVIDER_ORDER=jina,voyage-2.5,voyage-2.5-lite
VOYAGE_API_KEY=
JINA_API_KEY=

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=rag_ht_test
MYSQL_USER=root
MYSQL_PASSWORD=your-local-password
MYSQL_TABLE=ads_search_ready
MYSQL_CONTENT_COLUMN=embedding_content
MYSQL_BM25_COLUMN=bm25_content
MYSQL_SEARCH_ID_COLUMN=id
MYSQL_RESULT_TABLE=ads
MYSQL_RESULT_ID_COLUMN=id
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Install the local Ollama embedding model:

```bash
ollama pull embeddinggemma:latest
```

Hosted reranking sends the query and candidate text to the selected provider.
The local ModernBERT implementation remains available by explicitly adding
`local` to `RERANK_PROVIDER_ORDER`, but it is not part of the production
three-model chain.

## Company Database Contract

Each RAG_HT company adapter publishes a canonical search-ready table in MySQL
or PostgreSQL. Table and column names are configured per tenant.

Important fields include:

- `id`
- `embedding_content`
- `bm25_content`
- `main_category_name`
- `subcategory_name`
- `state_name`
- `city_name`
- `locality_name`
- `rental_duration`
- `rental_fee`

`embedding_content` is labelled semantic text containing title, description,
metadata, taxonomy, location, attributes, and selected values.

`bm25_content` contains exact searchable terms such as IDs, brands, models,
keywords, category names, location names, and attribute values.

The final result source is the configured canonical result table, joined by
the configured result ID after reranking. `result_type_column` must identify
offer versus wanted intent (`1=offer`, `2=wanted`) even when the company's
physical column is named differently.

## Database Ingestion

Ingestion never searches every company's database together. `--company`
selects one profile, opens only that profile's MySQL/PostgreSQL connection,
streams its `search_ready` rows, and writes only its configured vector
collection/table and BM25 file. Stable document IDs include database backend,
company, database, table, and row identity. Vector metadata also includes
`company_id`.

Changed and new rows are embedded; unchanged content hashes are skipped. The
company database remains read-only. `--mysql-replace-source` treats the current
table as an authoritative snapshot by clearing that company's vector/BM25
indexes before rebuilding. Despite the legacy CLI flag name, a PostgreSQL
profile uses PostgreSQL.

Validate the configured local database without embedding:

```bash
.venv/bin/python src/ingest.py --mysql --check --limit 10
```

Run a small ingestion:

```bash
.venv/bin/python src/ingest.py --mysql --limit 100
```

Run full incremental ingestion:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-batch-size 500 --embed-batch-size 32
```

Production tenant-isolated ingestion (the profile selects MySQL/PostgreSQL and
Chroma/pgvector):

```bash
.venv/bin/python src/ingest.py \
  --company gainr \
  --mysql \
  --mysql-batch-size 500 \
  --embed-batch-size 32
```

This writes to the vector backend and BM25 file declared by the company
profile. It includes `company_id` in vector metadata and document identity.
Run it once before enabling tenant mode because legacy local indexes are
separate from company indexes.

Rows are streamed from the company database. Stable document IDs and content
hashes allow
unchanged rows to be skipped when ingestion is resumed.

Rebuild only BM25:

```bash
.venv/bin/python src/ingest.py --mysql-bm25-only --mysql-batch-size 5000
```

Replace the database source inside the configured vector backend:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-replace-source
```

Force all rows to be embedded again:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-force-reembed
```

These commands do not update or delete company database source rows.

## Local File Ingestion

Files placed in `data/raw_docs/` can also be indexed.

Supported extensions:

- `.pdf`
- `.csv`
- `.tsv`
- `.xlsx`
- `.xlsm`

Validate:

```bash
.venv/bin/python src/ingest.py --check
```

Ingest:

```bash
.venv/bin/python src/ingest.py
```

## Run Search

Run one query on the current machine:

```bash
.venv/bin/python src/chat.py \
  --query "bike in Chennai under 1000" \
  --limit 5
```

Run against an indexed company profile:

```bash
.venv/bin/python src/chat.py \
  --company gainr \
  --query "portable camera for a wedding" \
  --limit 5
```

Open the interactive shell:

```bash
.venv/bin/python src/chat.py
```

Example queries:

```text
something portable that can record a wedding clearly from far away in Chennai for a day
A vehicle for recreational driving on rough terrain.
Quad Bike for Hourly Rent in Bangalore
someone looking for bikes in 1000 range per hour
need a bike for a week within 800
```

Deterministic catalog queries use the existing database browse path and do not
call embeddings or a reranker. Semantic queries run vector and BM25 retrieval
in parallel, then use Jina, Voyage 2.5, or Voyage 2.5 Lite. The selected model
and fallback attempts are included in logs and timings.

## HTTP API

Start one API process from the repository root:

```bash
.venv/bin/python src/run_api.py
```

Expected startup output:

```text
INFO: Waiting for application startup.
INFO: Redis cache connected key_prefix=semantic_ads
INFO: Initializing the configured reranker chain...
INFO: Reranker chain ready model_order=jina:jina-reranker-v2-base-multilingual -> voyage-2.5:rerank-2.5 -> voyage-2.5-lite:rerank-2.5-lite ...
INFO: Preloading the Ollama embedding model...
INFO: Ollama embedding model ready in 129 ms.
INFO: Application startup complete.
INFO: Uvicorn running on http://127.0.0.1:8000
```

Each new search prints a correlated flow using an eight-character search ID:

```text
INFO: [search:a1b2c3d4] step=search status=start query_chars=39 limit=200
INFO: [search:a1b2c3d4] step=plan status=start query_chars=39 models=gemini-3.1-flash-lite -> ...
INFO: step=query_model status=attempt model=gemini-3.1-flash-lite position=1/3
INFO: step=query_model status=success model=gemini-3.1-flash-lite duration_ms=1120
INFO: [search:a1b2c3d4] step=plan status=complete model=gemini-3.1-flash-lite ...
INFO: [search:a1b2c3d4] step=retrieve status=complete vector=120 bm25=120 candidates=60 ...
INFO: step=reranker_provider status=success provider=jina model=jina-reranker-v2-base-multilingual ...
INFO: [search:a1b2c3d4] step=rerank status=complete provider=jina results=60 ...
INFO: [search:a1b2c3d4] step=related_tail status=complete primary=60 related=140 ...
INFO: [search:a1b2c3d4] step=database_map status=complete rows=200
INFO: [search:a1b2c3d4] step=search status=complete products=200 duration_ms=2840
```

Fast-path logs contain `path=deterministic_filter`, `model=none`, and
`step=fast_filter`. Repeated normalized queries log
`step=plan status=cache_hit`.

Query plans are cached for 15 minutes by default. A bounded in-process LRU is
checked first; Redis is the shared backing cache, allowing a plan to survive an
API restart or be reused by another worker. Redis keys contain a SHA-256 digest
of the normalized query rather than the query text. If Redis is stopped,
search continues with the in-process cache and periodically retries Redis.
Disable Redis with `REDIS_ENABLED=false`.

Successful search result ordering is cached separately in Redis for five
minutes. This cache contains product IDs, ranking tiers, and interpreted
filters—not full product descriptions or photos. A repeated query therefore
skips planning, embeddings, BM25, fusion, and reranking, while current
canonical rows and visibility state are fetched again from the company
database. Result keys
include the BM25 index revision and automatically change after ingestion.
Disable this layer with `REDIS_RESULT_CACHE_ENABLED=false`.

When a model is exhausted or temporarily unavailable, a warning shows the HTTP
status and next model. Logs include model names, stage timings, candidate
counts, and filter field names. They intentionally omit the API key, raw query,
filter values, and product contents. Set `API_LOG_LEVEL=warning` in `.env` for
quiet operation, or `API_LOG_LEVEL=debug` for library diagnostics.

The API initializes the three hosted rerankers without loading ModernBERT.
Hosted calls can execute concurrently across companies. The API also preloads
`embeddinggemma:latest`.
`OLLAMA_KEEP_ALIVE=-1` keeps it resident until Ollama is stopped. Keep one API
process running so requests reuse embeddings and any loaded local fallback.

Interactive OpenAPI documentation is available at
`http://127.0.0.1:8000/docs`. Check readiness with:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

With tenant mode enabled, check the company-specific endpoint:

```bash
curl http://127.0.0.1:8000/api/v1/gainr/health \
  -H 'X-API-Key: YOUR_COMPANY_API_KEY'
```

Example health output:

```json
{
  "status": "ok",
  "app": "Local Data Assistant",
  "indexed_products": 250117,
  "max_result_window": 200,
  "session_ttl_seconds": 600,
  "reranker_model": "jina:jina-reranker-v2-base-multilingual -> voyage-2.5:rerank-2.5 -> voyage-2.5-lite:rerank-2.5-lite",
  "reranker_loaded": true,
  "reranker_load_ms": 2796.68,
  "embedding_warmup": {
    "embedding_model": {
      "model": "embeddinggemma:latest",
      "total_ms": 128.53,
      "load_ms": 83.12
    }
  },
  "redis_enabled": true,
  "redis_connected": true,
  "query_plan_cache_backend": "redis+memory",
  "result_cache_enabled": true,
  "result_cache_ttl_seconds": 300
}
```

### First search batch

Gainr uses `api.endpoint_slug: gainr` and the canonical request field names in
`configs/tenants/gainr.yaml`. Send a query and the number of products the UI
wants initially:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gainr/search \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_COMPANY_API_KEY' \
  -d '{"query":"bike in Chennai under 1000 per day"}'
```

`page_size` defaults to 20. It can be included explicitly as
`"page_size": 20`; values above 20 are rejected.

Another company can use a different endpoint and request payload:

```yaml
api:
  endpoint_slug: acme
payload:
  request_mapping:
    query: search_text
    cursor: continuation_token
    page_size: limit
```

That client sends:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/acme/search \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: ACME_API_KEY' \
  -d '{"search_text":"portable camera","limit":20}'
```

The endpoint slug, API key, database connection, vector table/collection, BM25
file, Redis namespace, cursor sessions, input mapping, and response field
mapping all resolve from the same tenant profile. The legacy
`/api/v1/search` route remains available for compatibility, but production
clients should use their company-specific path.

### Per-company monthly usage

Every API search records its execution path. Provider-reported Google, Jina,
and Voyage token counts are aggregated by UTC month, company, provider, model,
operation, and status in `storage/usage.sqlite3`. Raw queries and product text
are never stored in this ledger. Deterministic and cache-hit searches record
zero model tokens.

The search response includes usage for that request. A company can retrieve
only its own monthly totals:

```bash
curl 'http://127.0.0.1:8000/api/v1/gainr/usage?month=2026-07' \
  -H 'X-API-Key: GAINR_API_KEY'
```

The response contains `searches`, `model_requests`, input/output/total tokens,
and a provider/model breakdown. The endpoint/API-key ownership check is the
same one used for search, so one company cannot request another company's
ledger.

The response shape is:

```json
{
  "search_id": "9fdc4b42-0867-442b-92ef-c14678f2c668",
  "query": "bike in Chennai under 1000 per day",
  "cached": false,
  "items": [
    {
      "result_tier": "filtered",
      "id": "231049",
      "title": "Bajaj Pulsar 220 Bike for Daily Rent",
      "rental_duration": "Per Day",
      "rental_fee": "750.00"
    }
  ],
  "interpreted_query": {
    "semantic_query": "bike in Chennai under 1000 per day",
    "keyword_query": "bike in Chennai under 1000 per day",
    "target_ad_type": "offer",
    "execution_path": "deterministic_filter",
    "plan_cache_hit": false,
    "result_cache_hit": false,
    "query_corrections": []
  },
  "applied_filters": {
    "categorical": {
      "main_category_name": "Automobiles",
      "subcategory_name": "Bike",
      "state_name": "Tamil Nadu",
      "city_name": "Chennai",
      "rental_duration": "Per Day"
    },
    "max_rental_fee": 1000
  },
  "unresolved_filters": {},
  "timings_ms": {
    "planning": 136,
    "vector_search": 0,
    "bm25_search": 0,
    "related_tail": 268,
    "reranker_load": 0,
    "reranking": 0,
    "total": 611,
    "query_model_total": 0,
    "query_model_load": 0,
    "embedding_model_total": 0,
    "embedding_model_load": 0
  },
  "pagination": {
    "page_size": 20,
    "returned": 20,
    "offset": 0,
    "total_results": 200,
    "has_more": true,
    "next_cursor": "NEXT_CURSOR"
  }
}
```

`items` come from canonical rows in the configured company result table, but
the public API
returns an explicit field allowlist. Internal user IDs, phone numbers, hidden
contact data, keywords, and administrative fields are not serialized. Rows
with a non-null `deleted_at` are excluded. The API does not guess visibility
from `ads.status`, because wanted rows use a different status lifecycle.
`result_tier` is the only synthetic result field: `filtered` identifies direct
fast-path results, `ranked` identifies primary cross-encoder results, and
`related` identifies the semantic pipeline's filtered tail. Fast-path result
windows contain only `filtered` items. For semantic searches, pages 1–3 contain
the 60 primary ranked results and page 4 onward contains the related tail. If
fewer than 60 primary results are available, the related tier begins
immediately after the last primary result.

### Infinite scroll / next batch

When `has_more` is `true`, send the returned cursor instead of the query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/gainr/search \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: YOUR_COMPANY_API_KEY' \
  -d '{"cursor":"PASTE_NEXT_CURSOR_HERE"}'
```

Abbreviated page-2 output:

```json
{
  "query": "bike in Chennai under 1000 per day",
  "cached": true,
  "items": [
    {
      "result_tier": "filtered",
      "id": "10094",
      "title": "Honda Dream Yuga Bike for Daily Rent",
      "rental_fee": "600.00"
    }
  ],
  "pagination": {
    "page_size": 20,
    "returned": 20,
    "offset": 20,
    "total_results": 200,
    "has_more": true,
    "next_cursor": "NEXT_CURSOR"
  }
}
```

The measured page-2 response time was 1.1 ms because it used the cached result
window.

The server constructs the configured combined result window once and keeps it
in memory for 10 minutes by default. Cursor requests return stable slices from
that window, so scrolling does not repeat query extraction, embeddings,
retrieval, reranking, or related-tail selection. Cursor responses have
`"cached": true`. Repeating a query can also return `"cached": true` when its
Redis result-ID cache is hit; `interpreted_query.result_cache_hit` distinguishes
that case. A cursor is intentionally opaque; the frontend should store
and return it unchanged. An expired cursor returns HTTP `410`, and the frontend
must start a new search using `query`.

Pagination covers the configured result window, not an unlimited SQL catalog
scan. The default maximum is 200 products before API visibility filtering,
served 20 at a time. If more than 200 products match `bike`, the current cursor
cannot continue beyond that window. Full-catalog scrolling would require
database keyset pagination and a different API/session design.

`K` controls primary candidate-pool depth; it is not the end of the catalog.
The API fuses 60 primary candidates, reranks them, and places those 60 first.
Positions 61–200 are a stable tail matching whichever validated intent
fields are present. For example, a query containing only `city=Chennai` can
produce a Chennai tail; it does not also require a category, state, duration,
and price. A completely unfiltered query does not receive a random catalog
tail. Primary IDs are deduplicated from the tail.

API behavior is configured under `api` in `config.yaml`:

- `default_page_size`: used when the payload omits `page_size`
- `max_page_size`: validation ceiling for one response
- `max_results`: maximum combined ranked-plus-related window available to scroll
- `session_ttl_seconds`: lifetime of an in-memory cursor
- `max_sessions`: memory bound for active searches
- `tenant_max_concurrent_searches`: parallel search bound inside one company
- `usage_tracking_enabled` / `usage_db_path`: monthly model-token ledger

Set `API_CORS_ORIGINS` in `.env` when a browser frontend runs on a different
origin. Use a comma-separated allowlist; do not use `*` with private data.

### Hosted reranking

The production chain contains exactly three hosted models:

```text
jina-reranker-v2-base-multilingual
  -> rerank-2.5
  -> rerank-2.5-lite
```

Jina receives normal traffic. A timeout, connection failure, provider error, or
HTTP 429 advances to Voyage 2.5 and then Voyage 2.5 Lite. Each Voyage model has
its own process-wide rolling 3-RPM budget configured by
`VOYAGE_RERANK_RPM_PER_MODEL`. The shared limiter covers all company engines in
that API process, so simultaneous tenants cannot each consume a separate
three-request allowance. Keep one API process on the 16 GB deployment unless
the provider limit is moved into shared Redis coordination.

Jina counts the full reranking input, not merely one search request. With
roughly 60 candidates averaging about 1,500 characters, the old ceiling sent
approximately 88,750 characters and around 20K+ tokens per semantic search.
The configured 800-character ceiling reduces that payload while preserving the
leading labelled title/category/description content. A verified local semantic
run after this change reported 14,546 Jina tokens plus 1,264 Google planning
tokens.

### Hosted query planning

The Google API is used only for structured extraction when a query does not
qualify for the deterministic fast path and its normalized plan is not cached.
The configured order is `gemini-3.1-flash-lite`, `gemma-4-26b-a4b-it`,
then `gemma-4-31b-it`. A request moves to the next model only for HTTP
429 (quota/rate limit), a temporary HTTP 5xx provider failure, a connection
failure, or a per-model timeout. `GEMINI_TIMEOUT_SECONDS` defaults to 10
seconds, so one stalled model no longer blocks planning for 60 seconds.
Authentication, permission, and malformed-request errors do not trigger
fallback. Quota numbers are not hardcoded because Google applies them per
project and model and may change them; the provider's HTTP response is the
source of truth. Local embeddings, filtering, BM25, reranking, and canonical
database result retrieval are unchanged. The API key is read from `.env` and sent
in the `X-goog-api-key` header.

`gemini-3.1-flash-live-preview` is intentionally excluded. It is a voice-first
Live API model that exposes bidirectional generation rather than the
`generateContent` endpoint used here, and it does not support structured JSON
outputs. It is appropriate for a separate real-time audio interface, not this
request-response query planner.

This removes the local 12B model's memory and cold-start cost. The tradeoff is
that each raw search query, including any stated location and budget, is sent to
Google. It also adds network dependency, provider rate limits, and usage cost.
If every configured Google model is unavailable, the planner falls back to the
original query with no model-extracted filters.

To release Ollama memory after stopping the API:

```bash
ollama stop embeddinggemma:latest
```

## Evaluation

Run deterministic unit tests:

```bash
.venv/bin/python -m pytest -q
```

Run query-understanding scenarios:

```bash
.venv/bin/python src/evaluate_queries.py
```

Cases are stored in `eval/query_cases.json`.

Run end-to-end labeled retrieval:

```bash
.venv/bin/python src/evaluate_retrieval.py
```

Cases are stored in `eval/retrieval_cases.json`. The evaluator reports passed
cases and Mean Reciprocal Rank.

Latest verified results:

```text
109 unit tests passed
5/5 end-to-end retrieval cases passed
MRR = 0.672
```

Compile-check:

```bash
python3 -m compileall -q src tests
```

## Index Management

List indexed sources:

```bash
.venv/bin/python src/ingest.py --list
```

Delete one source from the configured vector backend:

```bash
.venv/bin/python src/ingest.py --delete "source-name"
```

Clear the configured vector backend:

```bash
.venv/bin/python src/ingest.py --clear
```

These operations affect the configured vector backend only. They do not delete
source files, BM25 rows, or company database records.

## Project Structure

```text
src/
  chat.py                 Interactive CLI
  api.py                  HTTP API and cursor pagination
  run_api.py              API process entry point
  search_engine.py        Reusable end-to-end search service
  query_planner.py        LLM extraction and deterministic validation
  retrieval.py            Vector, BM25, RRF, and ad-type filtering
  reranker.py             Jina -> Voyage 2.5 -> Voyage 2.5 Lite chain
  providers.py            Replaceable model-provider protocols
  gemini_client.py        Hosted structured-query provider
  ollama_client.py        Local embedding provider
  redis_cache.py          Optional shared Redis query-plan cache
  rate_limit.py           Per-company Redis/local token bucket
  usage_store.py          Per-company monthly model-token ledger
  tenant_config.py        Tenant profiles, API-key registry, isolation checks
  bm25_index.py           Persistent SQLite FTS5 index
  database_store.py       MySQL/PostgreSQL database dispatch
  mysql_store.py          MySQL reads and canonical record lookup
  postgres_store.py       PostgreSQL reads and canonical record lookup
  vector_store.py         Chroma/pgvector backend dispatch
  pgvector_store.py       Tenant-isolated pgvector collection adapter
  ingestion_service.py    Incremental ingestion workflows
  document_processing.py  Source and metadata preparation
  evaluate_queries.py     Query-plan evaluation
  evaluate_retrieval.py   End-to-end retrieval evaluation
eval/
  query_cases.json
  retrieval_cases.json
configs/tenants/
  gainr.yaml              Gainr DB, storage, payload, and rate policy
  example-company.yaml.example
                           PostgreSQL and pgvector profile example
scripts/
  generate_company_api_key.py
                           One-time tenant credential generator
```

## Remaining Production Work

- Run a live PostgreSQL/pgvector integration test against the exact production
  PostgreSQL major version, permissions, backup policy, and embedding
  dimensions. Unit tests cover configuration and adapters but this repository
  does not include a PostgreSQL service fixture.
- Add generation-based index builds with atomic promotion so a failed full
  company reindex cannot expose a partially replaced vector/BM25 pair.
- Add a durable asynchronous ingestion job API, progress state, retry policy,
  and deletion manifest. The current CLI is resumable but still operator-run.
- Benchmark deterministic, semantic, cached, and hosted-provider-fallback
  paths on the target 16 GB Ubuntu host before setting concurrency and rate
  limits.
- Add service units, TLS/reverse-proxy configuration, encrypted secret
  injection, Redis/PostgreSQL backup monitoring, and provider usage alerts.
- The labeled retrieval set is intentionally small and must grow before making
  production-quality claims.
- Soft category boosts and candidate counts still require benchmark-driven
  tuning.
- The HTTP API excludes soft-deleted rows but does not interpret `ads.status`.
  Confirm the complete status/visibility policy with the owning team before
  production deployment. The HTTP API and chat command apply the configured
  public-field and soft-delete presentation rules.
- City aliases are not a complete geographic knowledge base.
- Exact-title diversification can hide multiple legitimate listings with the
  same title; business-specific deduplication should eventually use seller,
  location, price, and availability.
