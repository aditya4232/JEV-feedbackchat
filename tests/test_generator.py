"""Tests for the real configurable generator adapter without network calls."""

from __future__ import annotations

import json
from typing import Any

import pytest

from jevloop import (
    EffectivePrompt,
    GeneratorError,
    Message,
    OpenAICompatibleGenerator,
    build_generator,
)


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return self.body


def test_generator_sends_exact_prompt_and_bounded_history() -> None:
    captured: dict[str, Any] = {}

    def opener(request: Any, *, timeout: float) -> _Response:
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return _Response(
            json.dumps(
                {
                    "id": "req-test",
                    "choices": [{"message": {"role": "assistant", "content": "answer"}}],
                }
            ).encode()
        )

    generator = OpenAICompatibleGenerator(
        base_url="https://api.example.test/v1",
        model="configured-model",
        api_key="fake-test-key",
        timeout_seconds=7,
        max_history_messages=2,
        max_history_chars=8,
        opener=opener,
    )
    user_request = "  keep this request verbatim  "
    prompt = EffectivePrompt(
        system_prompt="adapted system prompt",
        user_message=user_request,
        conversation=[
            Message(id="u1", role="user", text="oldest history", turn=0),
            Message(id="a1", role="assistant", text="older answer", turn=0),
            Message(id="u2", role="user", text="recent context", turn=1),
        ],
    )

    result = generator(prompt, {"feedback": {"flagged": {"deception": 0.9}}})

    messages = captured["body"]["messages"]
    assert captured["url"] == "https://api.example.test/v1/chat/completions"
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer fake-test-key"
    assert messages[0] == {"role": "system", "content": "adapted system prompt"}
    assert messages[-1] == {"role": "user", "content": user_request}
    history = messages[1:-1]
    assert len(history) <= 2
    assert sum(len(message["content"]) for message in history) <= 8
    assert result.text == "answer"
    assert result.metadata["request_id"] == "req-test"
    assert result.metadata["history_messages"] == len(history)


@pytest.mark.parametrize(
    "response_body",
    [b"not-json", b"{}", b'{"choices":[{"message":{"content":[]}}]}'],
)
def test_generator_rejects_malformed_responses(response_body: bytes) -> None:
    generator = OpenAICompatibleGenerator(
        base_url="http://127.0.0.1:9999/v1",
        model="local-model",
        opener=lambda *_args, **_kwargs: _Response(response_body),
    )
    with pytest.raises(GeneratorError):
        generator(EffectivePrompt("system", "user", []), {})


def test_generator_factory_requires_explicit_real_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JEVLOOP_GENERATOR_BASE_URL", raising=False)
    monkeypatch.delenv("JEVLOOP_GENERATOR_MODEL", raising=False)
    with pytest.raises(GeneratorError, match="JEVLOOP_GENERATOR_BASE_URL"):
        build_generator(mode="openai-compatible")


def test_generator_factory_defaults_to_labelled_echo_mock() -> None:
    generator = build_generator(mode="mock")
    result = generator(EffectivePrompt("system", "hello", []), {})
    assert result.metadata["generator"] == "echo"


def test_generator_rejects_remote_plain_http() -> None:
    with pytest.raises(ValueError, match="loopback"):
        OpenAICompatibleGenerator(
            base_url="http://api.example.test/v1",
            model="configured-model",
        )
