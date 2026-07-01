import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import reranker
from reranker import (
    FallbackReranker,
    HostedReranker,
    RequestWindowLimiter,
)


class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "data": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.2},
            ]
        }


def test_voyage_scores_are_restored_to_input_order(monkeypatch):
    captured = {}

    def post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(reranker.requests, "post", post)
    provider = HostedReranker(
        name="voyage",
        url="https://voyage.example/rerank",
        api_key="secret",
        model="rerank-2.5-lite",
    )

    scores = provider.compute_score(
        [["query", "first"], ["query", "second"]]
    )

    assert scores == [0.2, 0.9]
    assert captured["json"]["return_documents"] is False
    assert captured["json"]["model"] == "rerank-2.5-lite"


def test_hosted_reranker_captures_provider_usage(monkeypatch):
    class UsageResponse(FakeResponse):
        def json(self):
            payload = super().json()
            payload["usage"] = {"total_tokens": 321}
            return payload

    monkeypatch.setattr(
        reranker.requests,
        "post",
        lambda *_args, **_kwargs: UsageResponse(),
    )
    provider = HostedReranker(
        name="jina",
        url="https://jina.example/rerank",
        api_key="secret",
        model="jina-reranker-v2-base-multilingual",
    )

    provider.compute_score([["query", "first"], ["query", "second"]])

    assert provider.last_usage == {
        "input_tokens": 321,
        "output_tokens": 0,
        "total_tokens": 321,
    }


def test_reranker_chain_uses_next_provider_after_failure():
    class Provider:
        def __init__(self, name, result=None):
            self.name = name
            self.model = name
            self.result = result

        def compute_score(self, *_args, **_kwargs):
            if self.result is None:
                raise RuntimeError("temporary failure")
            return self.result

    chain = FallbackReranker(
        [
            Provider("voyage"),
            Provider("jina", [0.7, 0.1]),
            Provider("local", [0.1, 0.2]),
        ]
    )

    assert chain.compute_score([["q", "a"], ["q", "b"]]) == [0.7, 0.1]
    assert chain.last_provider == "jina"
    assert [attempt["provider"] for attempt in chain.last_attempts] == [
        "voyage",
        "jina",
    ]


def test_request_window_limiter_enforces_per_model_budget():
    now = [100.0]
    limiter = RequestWindowLimiter(3, clock=lambda: now[0])

    assert limiter.allow() == (True, 0.0)
    assert limiter.allow() == (True, 0.0)
    assert limiter.allow() == (True, 0.0)
    allowed, retry_after = limiter.allow()
    assert allowed is False
    assert retry_after == 60.0

    now[0] = 161.0
    assert limiter.allow() == (True, 0.0)


def test_voyage_model_budget_is_counted_independently(monkeypatch):
    calls = []

    def post(*_args, **_kwargs):
        calls.append(1)
        return FakeResponse()

    monkeypatch.setattr(reranker.requests, "post", post)
    quality = HostedReranker(
        name="voyage-2.5",
        provider_name="voyage",
        url="https://voyage.example/rerank",
        api_key="secret",
        model="rerank-2.5",
        requests_per_minute=1,
    )
    lite = HostedReranker(
        name="voyage-2.5-lite",
        provider_name="voyage",
        url="https://voyage.example/rerank",
        api_key="secret",
        model="rerank-2.5-lite",
        requests_per_minute=1,
    )
    pairs = [["query", "first"], ["query", "second"]]

    quality.compute_score(pairs)
    lite.compute_score(pairs)

    assert len(calls) == 2
    try:
        quality.compute_score(pairs)
    except RuntimeError as exc:
        assert "request budget exhausted" in str(exc)
    else:
        raise AssertionError("quality model should have exhausted its budget")
