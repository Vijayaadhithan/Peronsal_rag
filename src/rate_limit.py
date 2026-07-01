from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from tenant_config import TenantProfile


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TenantRateLimiter:
    """Redis token bucket with a bounded-process fallback."""

    def __init__(
        self,
        redis_cache=None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.redis_cache = redis_cache
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(
        self,
        profile: TenantProfile,
        credential_digest: str,
    ) -> tuple[bool, int]:
        scope = f"{profile.company_id}:{credential_digest}"
        policy = profile.rate_limit
        if self.redis_cache is not None and hasattr(
            self.redis_cache,
            "allow_rate_limit",
        ):
            result = self.redis_cache.allow_rate_limit(
                scope,
                policy.requests_per_minute,
                policy.burst,
            )
            if result is not None:
                return result
        return self._allow_memory(
            scope,
            policy.requests_per_minute,
            policy.burst,
        )

    def _allow_memory(
        self,
        scope: str,
        requests_per_minute: int,
        burst: int,
    ) -> tuple[bool, int]:
        now = self.clock()
        refill_per_second = requests_per_minute / 60
        with self._lock:
            bucket = self._buckets.get(scope)
            if bucket is None:
                bucket = _Bucket(tokens=float(burst), updated_at=now)
                self._buckets[scope] = bucket
            elapsed = max(now - bucket.updated_at, 0)
            bucket.tokens = min(
                float(burst),
                bucket.tokens + elapsed * refill_per_second,
            )
            bucket.updated_at = now
            if bucket.tokens < 1:
                return False, max(int(bucket.tokens), 0)
            bucket.tokens -= 1
            return True, max(int(bucket.tokens), 0)
