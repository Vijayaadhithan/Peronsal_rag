#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tenant_config import discover_tenant_profiles


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a company API key. The secret is printed once and is "
            "not written to disk."
        )
    )
    parser.add_argument("--company", required=True)
    args = parser.parse_args()

    profiles = discover_tenant_profiles()
    try:
        profile = profiles[args.company]
    except KeyError:
        available = ", ".join(sorted(profiles)) or "none"
        print(
            f"Unknown company {args.company!r}; available: {available}",
            file=sys.stderr,
        )
        return 1

    secret = f"rag_{profile.company_id}_{secrets.token_urlsafe(32)}"
    fingerprint = hashlib.sha256(secret.encode("utf-8")).hexdigest()[:16]
    print(f"Company: {profile.company_id}")
    print(f"Endpoint: /api/v1/{profile.endpoint_slug}/search")
    print(f"Set one of: {', '.join(profile.api_key_envs)}")
    print(f"API key (shown once): {secret}")
    print(f"SHA-256 fingerprint: {fingerprint}")
    print("Store the key in a secret manager and send it as X-API-Key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
