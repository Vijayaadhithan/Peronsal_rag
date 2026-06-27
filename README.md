# Local Data Assistant

Local RAG assistant using:

- MySQL as the ads source table
- Ollama for embeddings and structured query extraction
- Chroma for local vector storage
- BM25 + vector search + BGE cross-encoder reranking

Current defaults are in `config.yaml`:

- Collection: `local_data`
- Embedding model: `embeddinggemma:latest`
- Query-extraction model: `gemma3:1b`
- Reranker: `BAAI/bge-reranker-large`
- Chroma storage: `storage/chroma/`

## Setup

Create or update `.env` with your local settings:

```bash
OLLAMA_BASE_URL=http://localhost:11434

MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=rag_ht_test
MYSQL_USER=root
MYSQL_PASSWORD=your-local-password
MYSQL_TABLE=ads_search_ready
MYSQL_CONTENT_COLUMN=embedding_content
MYSQL_SEARCH_ID_COLUMN=id
MYSQL_RESULT_TABLE=ads
MYSQL_RESULT_ID_COLUMN=id
```

Install dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Make sure Ollama has the embedding and chat models:

```bash
ollama pull embeddinggemma:latest
ollama pull gemma3:1b
```

Use the embedding model configured in `config.yaml`. The BGE reranker is
downloaded to the local Hugging Face cache on the first search.

## MySQL Ingestion

The current MySQL path reads:

```sql
SELECT embedding_content FROM ads_search_ready
```

In practice, `embedding_content` is labelled semantic text, not JSON. A row starts like:

```text
Title: ...
Description: ...
Listing meta title: ...
Main category: ...
Subcategory: ...
State: ...
City: ...
Locality: ...
Selected attributes: ...
```

During ingestion:

- `embedding_content` becomes the Chroma document text.
- Known labels are split onto separate lines before embedding.
- Label values are copied into metadata, such as `content_title`, `content_main_category`, `content_subcategory`, `content_state`, `content_city`, and `content_locality`.
- Other MySQL table columns are also stored as metadata.
- Rows are streamed from MySQL, so the process does not load all 250k rows into memory.
- Chroma upserts happen in batches.
- `bm25_content` is maintained in the persistent local
  `storage/bm25.sqlite3` FTS5 index.
- Each row gets a stable Chroma ID and `source_content_hash`.
- Re-running ingestion skips rows already indexed with the same embedding model and same content hash.
- If a MySQL row changes, only that changed row is re-embedded and upserted.

Check the table and planned row count without creating embeddings:

```bash
.venv/bin/python src/ingest.py --mysql --check --limit 10
```

Run a small smoke ingestion first:

```bash
.venv/bin/python src/ingest.py --mysql --limit 100
```

Run the full ingestion:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-batch-size 500 --embed-batch-size 32
```

This command is resumable. If it stops halfway, run the same command again; already indexed unchanged rows will be skipped.

Rebuild only the BM25 index without embeddings or Chroma changes:

```bash
.venv/bin/python src/ingest.py --mysql-bm25-only --mysql-batch-size 5000
```

Rebuild the MySQL source inside Chroma:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-replace-source --mysql-batch-size 500 --embed-batch-size 32
```

`--mysql-replace-source` deletes only the indexed Chroma chunks for `mysql:rag_ht_test.ads_search_ready`. It does not delete or update rows in MySQL.

Force re-embedding without deleting first:

```bash
.venv/bin/python src/ingest.py --mysql --mysql-force-reembed --mysql-batch-size 500 --embed-batch-size 32
```

Use `--mysql-force-reembed` when the embedding model behavior changed but the model name stayed the same, or when you want to refresh vectors even though the source text is unchanged.

## Local File Ingestion

The older local-file ingestion still works for files placed in `data/raw_docs/`.

Supported types:

- `.pdf`
- `.csv`
- `.tsv`
- `.xlsx`
- `.xlsm`

Validate local files:

```bash
.venv/bin/python src/ingest.py --check
```

Ingest local files:

```bash
.venv/bin/python src/ingest.py
```

## Index Management

List indexed sources:

```bash
.venv/bin/python src/ingest.py --list
```

Delete one indexed source from Chroma:

```bash
.venv/bin/python src/ingest.py --delete "source-name"
```

Clear the whole Chroma collection:

```bash
.venv/bin/python src/ingest.py --clear
```

Use `--yes` to skip delete confirmations:

```bash
.venv/bin/python src/ingest.py --clear --yes
```

These commands affect only Chroma index data. They do not delete source files or MySQL rows.

## Product Search

After ingestion:

```bash
.venv/bin/python src/chat.py
```

The search path is:

1. `gemma3:1b` converts the user request into structured JSON containing a
   semantic query, a keyword query, and explicit category/location/duration/price
   filters.
2. Chroma retrieves the nearest 100 `embedding_content` candidates without an
   expensive metadata scan, then exact filters are applied locally.
3. SQLite FTS5 executes BM25 against the persistent `bm25_content` index only
   after a query is submitted. Chat startup does not load or rebuild the corpus.
4. Explicit filters are resolved against indexed metadata and applied to both
   retrieval paths. Unrecognized filters are ignored rather than eliminating
   valid products.
5. Vector and BM25 candidates are merged and scored by
   `BAAI/bge-reranker-large`.
6. Only the ranked product IDs are retained.
7. Canonical rows are returned from `ads` with a parameterized
   `SELECT * FROM ads WHERE id IN (...)`, preserving reranker order.

The query-extraction LLM does not generate the final response. Product data
always comes from the canonical `ads` table. The cached BGE reranker is loaded
locally on the first query, without a Hugging Face network check.

## Verification

Run tests:

```bash
.venv/bin/python -m pytest -q
```

Compile-check source and tests:

```bash
python3 -m compileall -q src tests
```
