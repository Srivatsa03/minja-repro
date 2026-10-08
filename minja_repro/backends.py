"""Real model backends.

Plain ``urllib`` rather than a vendor SDK, so the package keeps zero
dependencies and the credential stays in the caller's environment. Nothing here
logs a key, and no key is accepted as a function argument from the CLI.

A note on what a Claude subscription does and does not cover: Pro, Max and Teams
are seats on claude.ai and the desktop and CLI apps. They do not include API
access. Programmatic use goes through api.anthropic.com with a key from the
console, billed separately. Driving a subscription CLI in a loop to avoid that is
not what a seat is for, so it is deliberately not implemented here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .models import CallableModel

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"


class BackendError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict, *, timeout: float, retries: int) -> dict:
    """POST JSON with bounded retries on transient failures.

    Retries only on 429 and 5xx. A 400 means the request is wrong and repeating
    it just burns tokens, so it is raised immediately.
    """
    body = json.dumps(payload).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, data=body, headers={**headers, "content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode(errors="replace")
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise BackendError(f"HTTP {exc.code}: {detail}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise BackendError(f"transport failure: {exc}") from exc
            last = exc
        time.sleep(min(2 ** attempt, 8))
    raise BackendError(f"exhausted retries: {last}")


def anthropic_backend(
    model: str = "claude-haiku-4-5-20251001",
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: float = 60.0,
    retries: int = 3,
) -> CallableModel:
    """Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment.

    Temperature defaults to 0 so a replication is as close to repeatable as the
    service allows. Any reported rate still has to name the model and the date,
    because a hosted model is not a fixed artifact.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise BackendError(
            "ANTHROPIC_API_KEY is not set. A Claude Pro, Max or Teams subscription does "
            "not include API access; create a key at console.anthropic.com."
        )

    def call(prompt: str) -> str:
        data = _post(
            ANTHROPIC_URL,
            {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            },
            {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION},
            timeout=timeout,
            retries=retries,
        )
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        if not text:
            raise BackendError(f"no text in response: {json.dumps(data)[:300]}")
        return text

    return CallableModel(call, f"anthropic:{model}")


def ollama_backend(
    model: str = "llama3.2:3b",
    *,
    host: str | None = None,
    timeout: float = 180.0,
    retries: int = 1,
) -> CallableModel:
    """A local ollama server. Free and keyless, with a caveat worth stating.

    On an 8 GB machine this means a small quantised model, which is a much weaker
    substitution than the GPT-4-class backbones the paper used. A result from it
    answers "does this transfer to a small local model", which is a different and
    narrower question than the paper's. Label it that way.
    """
    base = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST).rstrip("/")

    def call(prompt: str) -> str:
        data = _post(
            f"{base}/api/generate",
            {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0}},
            {},
            timeout=timeout,
            retries=retries,
        )
        text = data.get("response", "")
        if not text:
            raise BackendError(f"empty response from ollama: {json.dumps(data)[:200]}")
        return text

    return CallableModel(call, f"ollama:{model}")


def resolve(spec: str) -> CallableModel:
    """Build a backend from a ``kind:model`` string, e.g. ``anthropic:claude-sonnet-5``."""
    kind, _, name = spec.partition(":")
    if kind == "anthropic":
        return anthropic_backend(name or "claude-haiku-4-5-20251001")
    if kind == "ollama":
        return ollama_backend(name or "llama3.2:3b")
    raise BackendError(f"unknown backend {kind!r}; use anthropic:<model> or ollama:<model>")
