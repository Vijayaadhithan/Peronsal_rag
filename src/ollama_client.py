import requests

from settings import EMBED_MODEL, OLLAMA_BASE_URL


def embed_text(text: str) -> list[float]:
    return embed_texts([text], timeout=120)[0]


def embed_texts(texts: list[str], timeout: int = 300) -> list[list[float]]:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": texts,
                "keep_alive": "30m",
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Cannot get embeddings from Ollama at {OLLAMA_BASE_URL}. "
            f"Start Ollama and confirm '{EMBED_MODEL}' is installed."
        ) from exc

    embeddings = response.json().get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an invalid embedding response.")
    return embeddings


def structured_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    temperature: float = 0,
) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "format": schema,
                "stream": False,
                "think": False,
                "keep_alive": "30m",
                "options": {"temperature": temperature},
            },
            timeout=300,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except (requests.RequestException, KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Cannot extract a structured query with '{model}' at "
            f"{OLLAMA_BASE_URL}."
        ) from exc
