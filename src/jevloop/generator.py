"""Explicit mock and OpenAI-compatible chat-completions generators.

The real adapter uses only credentials and endpoint configuration supplied to
this application. It does not infer API access from a chat-product
subscription.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .contracts import EffectivePrompt, GeneratorOutput
from .deadline import within_deadline

MAX_RESPONSE_BYTES = 1_000_000


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        # Never forward the application credential to a redirected endpoint.
        return None


class GeneratorError(RuntimeError):
    """Raised when a configured generator cannot return a valid response."""


class OpenAICompatibleGenerator:
    """Call a configured OpenAI-compatible ``/chat/completions`` endpoint.

    ``base_url`` is the provider's API base URL (for example, an endpoint
    ending in ``/v1``). The adapter sends the effective system prompt, a
    bounded recent history, and the current user message as a separate
    unmodified user turn.
    """

    mode = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_history_messages: int = 20,
        max_history_chars: int = 12_000,
        max_tokens: int = 512,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("base_url and model must be non-empty")
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive")
        if (
            isinstance(max_history_messages, bool)
            or not isinstance(max_history_messages, int)
            or isinstance(max_history_chars, bool)
            or not isinstance(max_history_chars, int)
            or max_history_messages < 0
            or max_history_chars < 0
        ):
            raise ValueError("history limits must be non-negative")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.base_url = self._validate_base_url(base_url)
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars
        self.max_tokens = max_tokens
        self._opener = opener or build_opener(_NoRedirects()).open

    @staticmethod
    def _validate_base_url(base_url: str) -> str:
        parsed = urlsplit(base_url)
        if parsed.scheme not in ("https", "http") or not parsed.hostname:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, query, or fragment")
        local_hosts = {"localhost", "127.0.0.1", "::1"}
        if parsed.scheme != "https" and parsed.hostname.lower() not in local_hosts:
            raise ValueError("HTTP is allowed only for a loopback generator endpoint")
        return base_url.rstrip("/")

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _bounded_history(self, prompt: EffectivePrompt) -> list[dict[str, str]]:
        if not self.max_history_messages or not self.max_history_chars:
            return []
        messages: list[dict[str, str]] = []
        remaining = self.max_history_chars
        for item in reversed(prompt.conversation[-self.max_history_messages :]):
            if item.role not in ("user", "assistant") or remaining <= 0:
                continue
            text = item.text[-remaining:]
            messages.append({"role": item.role, "content": text})
            remaining -= len(text)
        messages.reverse()
        return messages

    def __call__(self, prompt: EffectivePrompt, _feedback: dict[str, Any]) -> GeneratorOutput:
        messages = [{"role": "system", "content": prompt.system_prompt}]
        messages.extend(self._bounded_history(prompt))
        # Keep the current request separate and verbatim, even if it is longer
        # than the history budget.
        messages.append({"role": "user", "content": prompt.user_message})
        body = json.dumps(
            {"model": self.model, "messages": messages, "max_tokens": self.max_tokens},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.endpoint, data=body, headers=headers, method="POST")

        def request_response() -> bytes:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                return bytes(response.read(MAX_RESPONSE_BYTES + 1))

        try:
            response_bytes = within_deadline(request_response, self.timeout_seconds)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise GeneratorError(f"generator request failed with HTTP {status}") from None
        except (URLError, TimeoutError, OSError) as exc:
            raise GeneratorError(f"generator request failed ({type(exc).__name__})") from None

        if len(response_bytes) > MAX_RESPONSE_BYTES:
            raise GeneratorError("generator response exceeded the 1 MB limit")
        try:
            payload = json.loads(response_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise GeneratorError("generator returned invalid JSON") from None
        if not isinstance(payload, dict):
            raise GeneratorError("generator response must be a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise GeneratorError("generator response has no first choice")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise GeneratorError("generator response content must be a non-empty string")

        request_id = payload.get("id")
        return GeneratorOutput(
            text=content,
            metadata={
                "generator": self.mode,
                "model": self.model,
                "request_id": request_id if isinstance(request_id, str) else None,
                "history_messages": len(messages) - 2,
            },
        )


def build_generator(
    *,
    mode: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> Callable[[EffectivePrompt, dict[str, Any]], GeneratorOutput]:
    """Build the explicitly selected real or deterministic mock generator."""
    selected_mode = mode or os.environ.get("JEVLOOP_GENERATOR_MODE", "mock")
    if selected_mode == "mock":
        from .loop import echo_generator

        return echo_generator
    if selected_mode != "openai-compatible":
        raise ValueError(f"Unknown generator mode: {selected_mode!r}")

    configured_base_url = base_url or os.environ.get("JEVLOOP_GENERATOR_BASE_URL")
    configured_model = model or os.environ.get("JEVLOOP_GENERATOR_MODEL")
    configured_api_key = (
        api_key if api_key is not None else os.environ.get("JEVLOOP_GENERATOR_API_KEY")
    )
    if not configured_base_url or not configured_model:
        raise GeneratorError(
            "openai-compatible mode requires JEVLOOP_GENERATOR_BASE_URL and "
            "JEVLOOP_GENERATOR_MODEL (or explicit arguments)"
        )
    return OpenAICompatibleGenerator(
        base_url=configured_base_url,
        model=configured_model,
        api_key=configured_api_key,
        **kwargs,
    )


__all__ = ["GeneratorError", "OpenAICompatibleGenerator", "build_generator"]
