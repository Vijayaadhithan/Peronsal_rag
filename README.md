# Personal RAG

A fully local PDF question-answering pipeline using Ollama, Chroma, BM25, and
FlashRank. Source documents and generated embeddings remain on this machine.

## Models

The checked-in configuration uses models already installed locally:

- `embeddinggemma:latest` for embeddings
- `gemma4:12b` for answers

Answers stream as they are generated. Optional model reasoning is disabled with
`llm.think: false` in `config.yaml` to reduce local response latency. The LLM is
kept loaded for 30 minutes; the first answer after a cold start can still take
longer while Ollama loads the 12B model.

`bge-m3:latest` is also suitable for embeddings, but changing embedding models
requires deleting `storage/chroma/` and ingesting again because vector dimensions
may differ.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Ensure the Ollama application/service is running, then confirm the models:

```bash
ollama list
```

Place PDFs in `data/raw_docs/`. This directory is ignored by Git to avoid
committing private or large documents.

## Validate And Run

Validate PDF text extraction without starting Ollama:

```bash
python src/ingest.py --check
```

Build or refresh the local index, then start chat:

```bash
python src/ingest.py
python src/chat.py
```

Ingestion is safe to rerun: unchanged PDFs are skipped, while changed PDFs have
their chunks replaced rather than duplicated. PDFs with no extractable text are
skipped and need OCR first.

## Manage Indexed Documents

The generated Chroma database is stored in `storage/chroma/` and ignored by Git.
Use the CLI rather than editing its database files directly:

```bash
# Show indexed filenames and chunk counts
python src/ingest.py --list
```

Example output:

```text
Collection: project_docs (1693 chunks)

  450 chunks  reference-book.pdf
  301 chunks  programming-guide.pdf
    2 chunks  sample-document.pdf
```

Remove one document from search results while retaining its original PDF:

```bash
python src/ingest.py --delete "sample-document.pdf"
```

The command asks for confirmation:

```text
Delete 2 indexed chunks for 'sample-document.pdf'? The PDF will be kept. [y/N]: y
Deleted 2 chunks for 'sample-document.pdf'.
```

Delete the complete index while retaining all original PDFs:

```bash
python src/ingest.py --clear
```

For scripts, add `--yes` to either deletion command to skip confirmation. Use
this carefully:

```bash
python src/ingest.py --delete "sample-document.pdf" --yes
python src/ingest.py --clear --yes
```

Delete a source PDF separately from `data/raw_docs/` if it should also be removed
from disk. Running `python src/ingest.py` again will re-index every PDF still in
that folder, including an indexed document that was previously deleted.
