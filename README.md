# Semantic Advertisement Search

Local-first semantic search for advertisement and ecommerce catalogs.

The system is designed for users who know what they need but may not know the
product, service, category, brand, or model name. It searches advertisement
titles, descriptions, keywords, taxonomy, attributes, prices, locations, and
rental metadata, then returns canonical rows from MySQL.

Example:

```text
something portable that can record a wedding clearly from far away
```

This can retrieve cameras, drone cameras, portable recorders, microphones,
gimbals, and related recording equipment without forcing the user to name one
specific product category.

## Current Stack

- MySQL: source and canonical advertisement records
- Ollama: local embeddings and structured query extraction
- Chroma: persistent vector index
- SQLite FTS5: persistent BM25 keyword index
- `BAAI/bge-reranker-large`: local cross-encoder reranking
- Python: ingestion, retrieval, evaluation, and CLI orchestration

Defaults are configured in `config.yaml`:

- Collection: `local_data`
- Embedding model: `embeddinggemma:latest`
- Query model: `gemma4:12b`
- Reranker: `BAAI/bge-reranker-large`
- Vector candidates: 30
- BM25 candidates: 30
- Hybrid candidates: 60
- Final results: 10

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

## End-to-End Search Flow

1. `gemma4:12b` rewrites the request into a semantic query, keyword query, ad
   intent, and possible structured constraints.
2. Deterministic validation corrects duration, price, ad type, taxonomy,
   aliases, spelling, and hierarchy relationships.
3. Explicit constraints become hard filters. Guessed categories become soft
   hints.
4. Chroma retrieves semantically similar advertisement documents.
5. SQLite FTS5 retrieves exact brands, models, keywords, categories, and
   attributes using BM25.
6. Reciprocal Rank Fusion combines vector and BM25 ranks without comparing
   their incompatible raw scores.
7. The canonical `ads.type` value removes offer/wanted mismatches before
   reranking.
8. `BAAI/bge-reranker-large` scores the original user request against each
   complete advertisement document.
9. Repeated exact titles are diversified so duplicates do not occupy all final
   result positions.
10. Only ranked advertisement IDs are retained.
11. Full records are fetched from the canonical `ads` table while preserving
    reranker order.

The extraction model never generates product records. Returned data always
comes from MySQL.

## Architecture Boundaries

`ProductSearchEngine` in `src/search_engine.py` exposes the complete:

```text
plan -> retrieve -> filter -> rerank -> map IDs -> fetch ads
```

flow. The CLI and evaluation tools use this same implementation.

Provider protocols are defined in `src/providers.py`:

- `EmbeddingProvider`
- `StructuredQueryProvider`
- `RerankingProvider`

Ollama is the current local provider. A hosted API can later implement these
interfaces without replacing filtering, retrieval, fusion, evaluation, or
MySQL mapping.

## Setup

Create `.env`:

```bash
OLLAMA_BASE_URL=http://localhost:11434

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

Install local Ollama models:

```bash
ollama pull embeddinggemma:latest
ollama pull gemma4:12b
```

The BGE reranker loads from the local Hugging Face cache. If it is not cached,
Transformers downloads it on first use.

## MySQL Data Contract

The search index is built from `ads_search_ready`.

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

The final result source is the canonical `ads` table, joined by advertisement
ID after reranking.

## MySQL Ingestion

Validate MySQL configuration without embedding:

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

Rows are streamed from MySQL. Stable document IDs and content hashes allow
unchanged rows to be skipped when ingestion is resumed.

Rebuild only BM25:

```bash
.venv/bin/python src/ingest.py --mysql-bm25-only --mysql-batch-size 5000
```

Replace the MySQL source inside Chroma:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-replace-source
```

Force all rows to be embedded again:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-force-reembed
```

These commands do not update or delete MySQL source rows.

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

```bash
.venv/bin/python src/chat.py
```

Example queries:

```text
something portable that can record a wedding clearly from far away in Chennai for a day
Quad Bike for Hourly Rent in Bangalore
someone looking for bikes in 1000 range per hour
need a bike for a week within 800
```

The reranker is loaded once per process and reused for later queries.

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
47 unit tests passed
8/8 query-plan cases passed
4/4 end-to-end retrieval cases passed
MRR = 0.750
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

Delete one Chroma source:

```bash
.venv/bin/python src/ingest.py --delete "source-name"
```

Clear Chroma:

```bash
.venv/bin/python src/ingest.py --clear
```

These operations affect Chroma only. They do not delete source files, BM25
rows, or MySQL records.

## Project Structure

```text
src/
  chat.py                 Interactive CLI
  search_engine.py        Reusable end-to-end search service
  query_planner.py        LLM extraction and deterministic validation
  retrieval.py            Vector, BM25, RRF, and ad-type filtering
  reranker.py             BGE cross-encoder adapter
  providers.py            Replaceable model-provider protocols
  ollama_client.py        Local Ollama provider
  bm25_index.py           Persistent SQLite FTS5 index
  mysql_store.py          MySQL reads and canonical record lookup
  ingestion_service.py    Incremental ingestion workflows
  document_processing.py  Source and metadata preparation
  evaluate_queries.py     Query-plan evaluation
  evaluate_retrieval.py   End-to-end retrieval evaluation
eval/
  query_cases.json
  retrieval_cases.json
```

## Current Limitations

- Query extraction and local embedding latency depend on Ollama model size and
  hardware.
- The first reranking request pays the BGE model-loading cost.
- The labeled retrieval set is intentionally small and must grow before making
  production-quality claims.
- Soft category boosts and candidate counts still require benchmark-driven
  tuning.
- The current canonical lookup does not exclude records based on `status` or
  `deleted_at`. A production deployment must define the advertisement
  visibility policy with the owning team.
- City aliases are not a complete geographic knowledge base.
- Exact-title diversification can hide multiple legitimate listings with the
  same title; business-specific deduplication should eventually use seller,
  location, price, and availability.
