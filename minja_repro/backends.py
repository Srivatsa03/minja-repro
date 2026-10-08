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
import re
import time
import urllib.error
import urllib.request

from .models import CallableModel

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

#: Some providers sit behind a WAF that rejects urllib's default agent outright.
#: Groq returns Cloudflare error 1010 to "Python-urllib/3.x" while accepting the
#: identical request from curl, so a real agent string is required, not cosmetic.
USER_AGENT = "minja-repro/0.1 (+https://github.com/Srivatsa03/minja-repro)"
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

#: OpenAI-compatible providers, by backend name. The chat-completions shape is identical
#: across these, so one implementation covers them and anything else that speaks it.
OPENAI_COMPATIBLE: dict[str, tuple[str, str, str]] = {
    # name: (base url, env var holding the key, default model)
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY", "grok-4"),
    # Groq is the inference provider at groq.com, not xAI's Grok model. Keys begin
    # "gsk_" where xAI's begin "xai-", and the two are easy to confuse by name.
    "groq": ("https://api.groq.com/openai/v1", "XAI_API_KEY", "openai/gpt-oss-120b"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4.1"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openai/gpt-4.1"),
}


class BackendError(RuntimeError):
    pass


def _retry_after_seconds(exc: urllib.error.HTTPError, body: str) -> float | None:
    """How long the server asked us to wait, from the header or the message body.

    A token-per-minute limit recovers on its own in well under a second, and the
    provider says exactly when: a Retry-After header, or a "try again in 165ms"
    phrase in the body. Honoring that beats a blind exponential backoff, which
    would sleep seconds for a sub-second limit.
    """
    header = exc.headers.get("retry-after") if exc.headers else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    match = re.search(r"try again in ([\d.]+)\s*(ms|s)", body)
    if match:
        value = float(match.group(1))
        return value / 1000.0 if match.group(2) == "ms" else value
    return None


def _post(url: str, payload: dict, headers: dict, *, timeout: float, retries: int) -> dict:
    """POST JSON, retrying transient failures and honoring the server's retry hint.

    A 400 means the request is wrong and repeating it just burns tokens, so it is
    raised immediately. A 429 is a rate limit that recovers, so it does not count
    against the retry budget: a long run on a tight free-tier quota is paced by
    the server rather than failed by it.
    """
    body = json.dumps(payload).encode()
    last: Exception | None = None
    attempt = 0
    rate_limit_waits = 0
    while True:
        request = urllib.request.Request(
            url,
            data=body,
            headers={**headers, "content-type": "application/json", "user-agent": USER_AGENT},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode(errors="replace")
            if exc.code == 429:
                # Rate limits recover; pace against them rather than spend a retry.
                # Cap the number of waits so a genuinely exhausted quota still ends.
                if rate_limit_waits >= 120:
                    raise BackendError(f"HTTP 429 after {rate_limit_waits} waits: {detail}") from exc
                rate_limit_waits += 1
                time.sleep((_retry_after_seconds(exc, detail) or 1.0) + 0.1)
                continue
            if exc.code not in (500, 502, 503, 504) or attempt >= retries:
                raise BackendError(f"HTTP {exc.code}: {detail}") from exc
            last = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt >= retries:
                raise BackendError(f"transport failure: {exc}") from exc
            last = exc
        attempt += 1
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


def openai_compatible_backend(
    provider: str,
    model: str | None = None,
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float = 90.0,
    retries: int = 3,
    reasoning_effort: str | None = "low",
) -> CallableModel:
    """Any provider speaking the OpenAI chat-completions shape, including xAI.

    One implementation rather than one per vendor, because the request and
    response bodies are the same and the only differences are the base URL, the
    environment variable, and the default model.
    """
    try:
        base, env_var, default_model = OPENAI_COMPATIBLE[provider]
    except KeyError:
        raise BackendError(
            f"unknown provider {provider!r}; known: {', '.join(sorted(OPENAI_COMPATIBLE))}"
        ) from None

    key = os.environ.get(env_var)
    if not key:
        raise BackendError(f"{env_var} is not set; export it and re-run")
    chosen = model or default_model

    unsupported: set[str] = set()
    token_cap: list[int | None] = [None]

    def call(prompt: str) -> str:
        payload: dict[str, object] = {
            "model": chosen,
            "max_tokens": token_cap[0] or max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        # A reasoning model spends the completion budget thinking before it emits
        # anything, so a budget sized for the visible answer alone returns empty
        # content with finish_reason "length". Capping the effort keeps the
        # answer in budget; gpt-oss-120b used 115 completion tokens at low effort
        # against 202 at default on the same prompt.
        if reasoning_effort and "reasoning_effort" not in unsupported:
            payload["reasoning_effort"] = reasoning_effort

        headers = {"authorization": f"Bearer {key}"}
        # Parameters that are valid for one model on an OpenAI-compatible endpoint
        # are rejected by another: a classifier refuses reasoning_effort, and a
        # small one caps max_tokens well below a chat model's. Rather than make the
        # caller carry a table of per-model quirks, adapt once from the error and
        # remember, so the rest of the run is one request per prompt.
        for _ in range(3):
            try:
                data = _post(f"{base}/chat/completions", payload, headers,
                             timeout=timeout, retries=retries)
                break
            except BackendError as exc:
                message = str(exc)
                if "reasoning_effort" in message and "reasoning_effort" in payload:
                    payload.pop("reasoning_effort", None)
                    unsupported.add("reasoning_effort")
                    continue
                cap = re.search(r"max_tokens` must be less than or equal to `(\d+)", message)
                if cap:
                    limit = int(cap.group(1))
                    payload["max_tokens"] = limit
                    token_cap[0] = limit
                    continue
                raise
        else:
            raise BackendError(f"could not satisfy {chosen} after adapting parameters")
        choices = data.get("choices") or []
        if not choices:
            raise BackendError(f"no choices in response: {json.dumps(data)[:300]}")
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        if not text:
            # Say why, rather than printing a truncated blob. Empty content on a
            # reasoning model almost always means the budget went on reasoning.
            raise BackendError(
                f"empty content from {chosen}: finish_reason={choice.get('finish_reason')!r}, "
                f"reasoning chars={len(message.get('reasoning') or '')}, "
                f"completion_tokens={(data.get('usage') or {}).get('completion_tokens')}. "
                "Raise max_tokens or lower reasoning_effort."
            )
        return text

    return CallableModel(call, f"{provider}:{chosen}")


def list_models(provider: str) -> list[str]:
    """Ask the provider what it serves, so a model name is checked not guessed."""
    base, env_var, _ = OPENAI_COMPATIBLE[provider]
    key = os.environ.get(env_var)
    if not key:
        raise BackendError(f"{env_var} is not set")
    request = urllib.request.Request(
        f"{base}/models",
        headers={"authorization": f"Bearer {key}", "user-agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise BackendError(f"HTTP {exc.code}: {exc.read()[:200].decode(errors='replace')}") from exc
    return sorted(m.get("id", "") for m in (payload.get("data") or []))


def resolve(spec: str) -> CallableModel:
    """Build a backend from a ``kind:model`` string, e.g. ``anthropic:claude-sonnet-5``."""
    kind, _, name = spec.partition(":")
    if kind == "anthropic":
        return anthropic_backend(name or "claude-haiku-4-5-20251001")
    if kind == "ollama":
        return ollama_backend(name or "llama3.2:3b")
    if kind in OPENAI_COMPATIBLE:
        return openai_compatible_backend(kind, name or None)
    known = ", ".join(["anthropic", "ollama", *sorted(OPENAI_COMPATIBLE)])
    raise BackendError(f"unknown backend {kind!r}; use one of: {known}")
