#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f /etc/os-release ]]; then
  echo "Cannot identify this Linux distribution."
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This installer supports Ubuntu. Detected: ${ID:-unknown}"
  exit 1
fi

if [[ "$(id -u)" -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-1}"
INSTALL_PGVECTOR="${INSTALL_PGVECTOR:-0}"
SKIP_LOCAL_RERANKER="${SKIP_LOCAL_RERANKER:-1}"
SKIP_TESTS="${SKIP_TESTS:-0}"

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  libgomp1 \
  libpq-dev \
  pkg-config \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  redis-server

"${SUDO[@]}" systemctl enable --now redis-server

if [[ "$INSTALL_OLLAMA" == "1" ]] && ! command -v ollama >/dev/null 2>&1; then
  OLLAMA_INSTALLER="$(mktemp)"
  curl -fsSL https://ollama.com/install.sh -o "$OLLAMA_INSTALLER"
  sh "$OLLAMA_INSTALLER"
  rm -f "$OLLAMA_INSTALLER"
fi

if command -v systemctl >/dev/null 2>&1 && command -v ollama >/dev/null 2>&1; then
  "${SUDO[@]}" systemctl enable --now ollama
fi

if [[ "$INSTALL_PGVECTOR" == "1" ]]; then
  if ! command -v pg_config >/dev/null 2>&1; then
    echo "pg_config is missing. Install PostgreSQL server development files first."
    exit 1
  fi
  PG_MAJOR="$(pg_config --version | awk '{print $2}' | cut -d. -f1)"
  if ! "${SUDO[@]}" apt-get install -y "postgresql-${PG_MAJOR}-pgvector"; then
    "${SUDO[@]}" apt-get install -y "postgresql-server-dev-${PG_MAJOR}"
    PGVECTOR_DIR="$(mktemp -d)"
    git clone --depth 1 --branch v0.8.4 \
      https://github.com/pgvector/pgvector.git "$PGVECTOR_DIR"
    make -C "$PGVECTOR_DIR"
    "${SUDO[@]}" make -C "$PGVECTOR_DIR" install
    rm -rf "$PGVECTOR_DIR"
  fi
fi

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
if [[ "$SKIP_TESTS" == "1" ]]; then
  .venv/bin/python -m pip install -r requirements.txt
else
  .venv/bin/python -m pip install -r requirements-dev.txt
fi

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env. Configure database and hosted API credentials before startup."
fi

if command -v ollama >/dev/null 2>&1; then
  ollama pull embeddinggemma:latest
fi

if [[ "$SKIP_LOCAL_RERANKER" != "1" ]]; then
  .venv/bin/python scripts/prefetch_models.py
fi

if [[ "$SKIP_TESTS" != "1" ]]; then
  .venv/bin/python -m pytest -q
fi

echo
echo "Ubuntu bootstrap complete."
echo "Required external configuration:"
echo "  - company MySQL/PostgreSQL credentials"
echo "  - GEMINI_API_KEY"
echo "  - JINA_API_KEY (primary hosted reranker)"
echo "  - VOYAGE_API_KEY (rerank-2.5 and rerank-2.5-lite fallbacks)"
echo "  - company API keys"
if [[ "$INSTALL_PGVECTOR" == "1" ]]; then
  echo "Run 'CREATE EXTENSION IF NOT EXISTS vector;' once in each vector database."
fi
echo
echo "Local smoke test:"
echo "  .venv/bin/python src/chat.py --query 'bike in Chennai under 1000' --limit 5"
