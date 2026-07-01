import threading
import time
from contextlib import contextmanager
import logging
from collections import deque

import requests

from settings import (
    JINA_API_KEY,
    JINA_RERANK_MODEL,
    JINA_RERANK_URL,
    RERANK_API_TIMEOUT_SECONDS,
    RERANK_BATCH_SIZE,
    RERANK_MAX_DOCUMENT_CHARS,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RERANK_PROVIDER_ORDER,
    RERANK_USE_FP16,
    VOYAGE_API_KEY,
    VOYAGE_RERANK_LITE_MODEL,
    VOYAGE_RERANK_MODEL,
    VOYAGE_RERANK_RPM_PER_MODEL,
    VOYAGE_RERANK_URL,
)

LOGGER = logging.getLogger("uvicorn.error")


class RequestWindowLimiter:
    """Thread-safe rolling one-minute request budget."""

    def __init__(self, requests_per_minute: int, clock=time.monotonic):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be greater than zero")
        self.requests_per_minute = requests_per_minute
        self.clock = clock
        self._requests = deque()
        self._lock = threading.Lock()

    def allow(self) -> tuple[bool, float]:
        now = self.clock()
        cutoff = now - 60
        with self._lock:
            while self._requests and self._requests[0] <= cutoff:
                self._requests.popleft()
            if len(self._requests) >= self.requests_per_minute:
                return False, max(60 - (now - self._requests[0]), 0)
            self._requests.append(now)
            return True, 0.0


class SharedReranker:
    """Loads one reranker instance for every tenant engine in the process."""

    def __init__(self, loader=None):
        self.loader = loader or load_reranker
        self.ranker = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def ensure(self):
        if self.ranker is not None:
            return self.ranker, 0.0
        with self._lock:
            if self.ranker is not None:
                return self.ranker, 0.0
            started = time.perf_counter()
            self.ranker = self.loader()
            return self.ranker, time.perf_counter() - started

    @contextmanager
    def inference_guard(self):
        if getattr(self.ranker, "supports_parallel", False):
            yield
            return
        with self._inference_lock:
            yield


class TransformerCrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        use_fp16: bool = False,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Transformer reranking requires torch and transformers. "
                "Install requirements.txt first."
            ) from exc

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                local_files_only=True,
            )
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if use_fp16 and self.device.type != "cpu":
            self.model = self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

    def compute_score(self, pairs, batch_size=None, max_length=None):
        if not pairs:
            return []
        batch_size = batch_size or self.batch_size
        max_length = max_length or self.max_length
        scores = []

        with self.torch.inference_mode():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self.tokenizer(
                    [pair[0] for pair in batch],
                    [pair[1] for pair in batch],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                }
                logits = self.model(**encoded).logits.view(-1).float()
                scores.extend(logits.cpu().tolist())
        return scores


class HostedReranker:
    def __init__(
        self,
        *,
        name: str,
        url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = RERANK_API_TIMEOUT_SECONDS,
        max_document_chars: int = RERANK_MAX_DOCUMENT_CHARS,
        provider_name: str | None = None,
        requests_per_minute: int | None = None,
        clock=time.monotonic,
    ):
        self.name = name
        self.provider_name = provider_name or name
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_document_chars = max_document_chars
        self.request_limiter = (
            RequestWindowLimiter(requests_per_minute, clock=clock)
            if requests_per_minute is not None
            else None
        )
        self._state = threading.local()

    @property
    def last_usage(self) -> dict[str, int]:
        return dict(getattr(self._state, "last_usage", {}))

    def _payload(self, query: str, documents: list[str]) -> dict:
        if self.provider_name == "voyage":
            return {
                "model": self.model,
                "query": query,
                "documents": documents,
                "return_documents": False,
                "truncation": True,
            }
        return {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": len(documents),
            "return_documents": False,
        }

    def compute_score(self, pairs, batch_size=None, max_length=None):
        self._state.last_usage = {}
        if not pairs:
            return []
        queries = {str(pair[0]) for pair in pairs}
        if len(queries) != 1:
            raise RuntimeError(
                f"{self.name} reranker requires one query per request."
            )
        query = str(pairs[0][0])
        documents = [
            str(pair[1])[: self.max_document_chars]
            for pair in pairs
        ]
        if self.request_limiter is not None:
            allowed, retry_after = self.request_limiter.allow()
            if not allowed:
                raise RuntimeError(
                    f"{self.name} local request budget exhausted; "
                    f"retry_after={retry_after:.1f}s"
                )
        try:
            response = requests.post(
                self.url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=self._payload(query, documents),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            total_tokens = int(usage.get("total_tokens", 0) or 0)
            self._state.last_usage = {
                "input_tokens": total_tokens,
                "output_tokens": 0,
                "total_tokens": total_tokens,
            }
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            reason = f"http_{status}" if status else type(exc).__name__
            raise RuntimeError(
                f"{self.name} reranker unavailable: {reason}"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                f"{self.name} reranker returned invalid JSON"
            ) from exc
        results = payload.get("data") or payload.get("results")
        if not isinstance(results, list):
            raise RuntimeError(
                f"{self.name} reranker response has no result list"
            )
        scores: list[float | None] = [None] * len(documents)
        for result in results:
            try:
                index = int(result["index"])
                score = float(
                    result.get(
                        "relevance_score",
                        result.get("score"),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"{self.name} reranker returned an invalid result"
                ) from exc
            if 0 <= index < len(scores):
                scores[index] = score
        if any(score is None for score in scores):
            raise RuntimeError(
                f"{self.name} reranker did not score every document"
            )
        return [float(score) for score in scores]


class LazyLocalReranker:
    name = "local"

    def __init__(self):
        self._ranker = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _ensure(self):
        if self._ranker is not None:
            return self._ranker
        with self._lock:
            if self._ranker is None:
                self._ranker = load_local_reranker()
        return self._ranker

    def compute_score(self, pairs, batch_size=None, max_length=None):
        with self._inference_lock:
            return self._ensure().compute_score(
                pairs,
                batch_size=batch_size,
                max_length=max_length,
            )


class FallbackReranker:
    supports_parallel = True

    def __init__(self, providers):
        if not providers:
            raise ValueError("At least one reranker provider is required.")
        self.providers = providers
        self._state = threading.local()

    @property
    def last_provider(self) -> str:
        return getattr(self._state, "last_provider", "")

    @property
    def last_attempts(self) -> list[dict]:
        return list(getattr(self._state, "last_attempts", []))

    @property
    def model_label(self) -> str:
        labels = []
        for provider in self.providers:
            model = getattr(provider, "model", RERANK_MODEL)
            labels.append(f"{provider.name}:{model}")
        return " -> ".join(labels)

    def compute_score(self, pairs, batch_size=None, max_length=None):
        self._state.last_attempts = []
        self._state.last_provider = ""
        failures = []
        for provider in self.providers:
            started = time.perf_counter()
            try:
                scores = provider.compute_score(
                    pairs,
                    batch_size=batch_size,
                    max_length=max_length,
                )
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                reason = str(exc).split(":", 1)[-1].strip()
                self._state.last_attempts.append(
                    {
                        "provider": provider.name,
                        "model": getattr(provider, "model", RERANK_MODEL),
                        "status": "fallback",
                        "reason": reason,
                        "duration_ms": elapsed_ms,
                        "usage": getattr(provider, "last_usage", {}),
                    }
                )
                failures.append(f"{provider.name}={reason}")
                LOGGER.warning(
                    "step=reranker_provider status=fallback provider=%s "
                    "reason=%s duration_ms=%.0f",
                    provider.name,
                    reason,
                    elapsed_ms,
                )
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._state.last_provider = provider.name
            self._state.last_attempts.append(
                {
                    "provider": provider.name,
                    "model": getattr(provider, "model", RERANK_MODEL),
                    "status": "success",
                    "duration_ms": elapsed_ms,
                    "usage": getattr(provider, "last_usage", {}),
                }
            )
            LOGGER.info(
                "step=reranker_provider status=success provider=%s "
                "model=%s duration_ms=%.0f",
                provider.name,
                getattr(provider, "model", RERANK_MODEL),
                elapsed_ms,
            )
            return scores
        raise RuntimeError(
            "All reranker providers failed: " + "; ".join(failures)
        )


def load_local_reranker():
    return TransformerCrossEncoderReranker(
        RERANK_MODEL,
        use_fp16=RERANK_USE_FP16,
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
    )


def load_reranker():
    providers = []
    for name in RERANK_PROVIDER_ORDER:
        if name in {"voyage", "voyage-2.5", "voyage-2.5-lite"}:
            if VOYAGE_API_KEY:
                model = (
                    VOYAGE_RERANK_LITE_MODEL
                    if name == "voyage-2.5-lite"
                    else VOYAGE_RERANK_MODEL
                )
                providers.append(
                    HostedReranker(
                        name=name,
                        provider_name="voyage",
                        url=VOYAGE_RERANK_URL,
                        api_key=VOYAGE_API_KEY,
                        model=model,
                        requests_per_minute=(
                            VOYAGE_RERANK_RPM_PER_MODEL
                        ),
                    )
                )
            else:
                LOGGER.info(
                    "Voyage reranker skipped because VOYAGE_API_KEY is unset."
                )
        elif name == "jina":
            if JINA_API_KEY:
                providers.append(
                    HostedReranker(
                        name="jina",
                        url=JINA_RERANK_URL,
                        api_key=JINA_API_KEY,
                        model=JINA_RERANK_MODEL,
                    )
                )
            else:
                LOGGER.info(
                    "Jina reranker skipped because JINA_API_KEY is unset."
                )
        elif name == "local":
            providers.append(LazyLocalReranker())
    if not providers:
        raise RuntimeError(
            "No reranker is configured. Set a hosted API key or include local "
            "in RERANK_PROVIDER_ORDER."
        )
    return FallbackReranker(providers)


# Backward-compatible import for existing callers.
BGEReranker = TransformerCrossEncoderReranker


def rerank(
    query,
    candidates,
    ranker,
    top_k=6,
    diversity_top_k=None,
):
    if not candidates:
        return []

    pairs = [[query, candidate["text"]] for candidate in candidates]
    scores = ranker.compute_score(
        pairs,
        batch_size=RERANK_BATCH_SIZE,
        max_length=RERANK_MAX_LENGTH,
    )
    if isinstance(scores, (int, float)):
        scores = [scores]

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    diversity_top_k = top_k if diversity_top_k is None else diversity_top_k
    primary = []
    deferred = []
    seen_titles = set()
    for position, (candidate, score) in enumerate(ranked):
        metadata = candidate.get("metadata") or {}
        title = metadata.get("content_title") or metadata.get("title")
        normalized_title = " ".join(str(title).casefold().split()) if title else None
        if (
            len(primary) < diversity_top_k
            and normalized_title
            and normalized_title in seen_titles
        ):
            deferred.append((position, candidate, score))
            continue
        if normalized_title:
            seen_titles.add(normalized_title)
        if len(primary) < diversity_top_k:
            primary.append((position, candidate, score))
        else:
            deferred.append((position, candidate, score))

    ordered = primary + sorted(deferred, key=lambda item: item[0])
    results = []
    for _, candidate, score in ordered[:top_k]:
        results.append(
            {
                "id": candidate["id"],
                "text": candidate["text"],
                "metadata": candidate["metadata"],
                "score": float(score),
            }
        )
    return results
