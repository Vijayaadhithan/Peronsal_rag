import logging
import threading
import time
from urllib.parse import quote

import requests

from settings import (
    GEMINI_API_BASE_URL,
    GEMINI_API_KEY,
    QUERY_EXTRACT_MODELS,
    QUERY_EXTRACT_TIMEOUT_SECONDS,
)

FALLBACK_HTTP_STATUSES = {408, 429, 500, 502, 503, 504}
LOGGER = logging.getLogger("uvicorn.error")


class GeminiModelUnavailableError(RuntimeError):
    def __init__(
        self,
        model: str,
        status_code: int | None = None,
        reason: str | None = None,
    ):
        self.model = model
        self.status_code = status_code
        self.reason = reason or (
            f"http_{status_code}" if status_code is not None else "unavailable"
        )
        super().__init__(
            f"Google model '{model}' is temporarily unavailable "
            f"({self.reason})."
        )


class GeminiProvider:
    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        base_url: str = GEMINI_API_BASE_URL,
        timeout_seconds: float = QUERY_EXTRACT_TIMEOUT_SECONDS,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._state = threading.local()

    @property
    def last_chat_metrics(self) -> dict[str, object]:
        return getattr(self._state, "last_chat_metrics", {})

    @last_chat_metrics.setter
    def last_chat_metrics(self, value: dict[str, object]) -> None:
        self._state.last_chat_metrics = value

    def structured_chat(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict,
        temperature: float = 0,
    ) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Add it to the project .env file."
            )

        started = time.perf_counter()
        metrics: dict[str, float | int | str | list] = {
            "load_ms": 0.0,
            "model": model,
        }
        try:
            response = requests.post(
                (
                    f"{self.base_url}/models/"
                    f"{quote(model, safe='.-')}:generateContent"
                ),
                headers={
                    "Content-Type": "application/json",
                    "X-goog-api-key": self.api_key,
                },
                json={
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}],
                    },
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": user_prompt}],
                        }
                    ],
                    "generationConfig": {
                        "temperature": temperature,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": schema,
                    },
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usageMetadata") or {}
            metrics.update(
                {
                    "input_tokens": int(
                        usage.get("promptTokenCount", 0) or 0
                    ),
                    "output_tokens": int(
                        usage.get("candidatesTokenCount", 0) or 0
                    ),
                    "thought_tokens": int(
                        usage.get("thoughtsTokenCount", 0) or 0
                    ),
                    "total_tokens": int(
                        usage.get("totalTokenCount", 0) or 0
                    ),
                }
            )
            content = payload["candidates"][0]["content"]
            text = "".join(
                part.get("text", "")
                for part in content.get("parts", [])
            ).strip()
            if not text:
                raise ValueError("Gemini returned an empty response.")
            return strip_json_fence(text)
        except requests.HTTPError as exc:
            status_code = (
                exc.response.status_code
                if exc.response is not None
                else 0
            )
            if status_code in FALLBACK_HTTP_STATUSES:
                raise GeminiModelUnavailableError(
                    model,
                    status_code=status_code,
                ) from exc
            raise RuntimeError(
                f"Cannot extract a structured query with Google model "
                f"'{model}' (HTTP {status_code})."
            ) from exc
        except requests.Timeout as exc:
            raise GeminiModelUnavailableError(
                model,
                reason="timeout",
            ) from exc
        except requests.ConnectionError as exc:
            raise GeminiModelUnavailableError(
                model,
                reason="connection_error",
            ) from exc
        except (
            requests.RequestException,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                f"Cannot extract a structured query with Google model "
                f"'{model}'."
            ) from exc
        finally:
            metrics["total_ms"] = (time.perf_counter() - started) * 1000
            self.last_chat_metrics = metrics


DEFAULT_GEMINI_PROVIDER = GeminiProvider()


def strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1 :]
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def structured_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    temperature: float = 0,
) -> str:
    models = (
        QUERY_EXTRACT_MODELS
        if model == QUERY_EXTRACT_MODELS[0]
        else (model,)
    )
    attempted_models = []
    attempts = []
    started = time.perf_counter()
    last_error = None
    for position, candidate_model in enumerate(models, start=1):
        attempted_models.append(candidate_model)
        LOGGER.info(
            "step=query_model status=attempt model=%s position=%d/%d",
            candidate_model,
            position,
            len(models),
        )
        try:
            content = DEFAULT_GEMINI_PROVIDER.structured_chat(
                candidate_model,
                system_prompt,
                user_prompt,
                schema,
                temperature,
            )
            DEFAULT_GEMINI_PROVIDER.last_chat_metrics.update(
                {
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "attempted_models": attempted_models,
                    "attempts": attempts
                    + [
                        {
                            **DEFAULT_GEMINI_PROVIDER.last_chat_metrics,
                            "status": "success",
                        }
                    ],
                }
            )
            LOGGER.info(
                "step=query_model status=success model=%s duration_ms=%.0f",
                candidate_model,
                DEFAULT_GEMINI_PROVIDER.last_chat_metrics["total_ms"],
            )
            return content
        except GeminiModelUnavailableError as exc:
            last_error = exc
            attempts.append(
                {
                    **DEFAULT_GEMINI_PROVIDER.last_chat_metrics,
                    "status": "fallback",
                    "reason": exc.reason,
                }
            )
            DEFAULT_GEMINI_PROVIDER.last_chat_metrics.update(
                {
                    "total_ms": (time.perf_counter() - started) * 1000,
                    "attempted_models": list(attempted_models),
                    "attempts": list(attempts),
                    "failure_reason": exc.reason,
                }
            )
            LOGGER.warning(
                "step=query_model status=fallback model=%s reason=%s "
                "next_model=%s",
                candidate_model,
                exc.reason,
                (
                    models[position]
                    if position < len(models)
                    else "none"
                ),
            )
    LOGGER.error(
        "step=query_model status=failed attempted_models=%s reason=%s",
        ",".join(attempted_models),
        last_error.reason if last_error is not None else "unknown",
    )
    raise RuntimeError(
        "All configured Google query models are unavailable "
        f"(last_reason={last_error.reason if last_error else 'unknown'})."
    ) from last_error


def last_gemini_metrics() -> dict[str, object]:
    return dict(DEFAULT_GEMINI_PROVIDER.last_chat_metrics)
