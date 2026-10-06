"""Real SDK transport contracts and overall deadlines; no external requests."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError

import httpx2
import pytest
from typesafe_sdk import TypeSafeClient

from jevloop import EffectivePrompt, Message, OpenAICompatibleGenerator
from jevloop.adapter import ClassifierError, LiveJevClassifier
from jevloop.generator import GeneratorError
from jevloop.taxonomy import DecisionPolicy


def test_real_sdk_serializes_and_parses_system_one_contract() -> None:
    def handler(request: Any) -> httpx2.Response:
        body = json.loads(request.content)
        assert request.url.path == "/v1/systemone"
        assert body["state"]["text"] == "Complete original message"
        assert set(body["questions"]) == {"happy", "sad", "deception", "cunning"}
        assert all(question["type"] == "noul" for question in body["questions"].values())
        return httpx2.Response(
            200,
            json={
                "model": "jev-contract-test",
                "usage": {},
                "answers": {name: {"type": "noul", "noul": 0.8} for name in body["questions"]},
            },
        )

    with TypeSafeClient(api_key="fake-test-key", transport=httpx2.MockTransport(handler)) as client:
        response = LiveJevClassifier(DecisionPolicy(), client=client).classify(
            Message(id="u", role="user", text="Complete original message", turn=0)
        )
    assert response.model == "jev-contract-test"
    assert sum(score.probability for score in response.scores) == pytest.approx(3.2)


@pytest.mark.parametrize("body", [{}, [], {"model": "jev", "usage": {}, "answers": []}])
def test_real_sdk_malformed_response_is_a_classifier_error(body: Any) -> None:
    with (
        TypeSafeClient(
            api_key="fake-test-key",
            transport=httpx2.MockTransport(lambda request: httpx2.Response(200, json=body)),
        ) as client,
        pytest.raises(ClassifierError),
    ):
        LiveJevClassifier(DecisionPolicy(), client=client).classify(
            Message(id="u", role="user", text="hello", turn=0)
        )


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_jev_http_failure_has_no_fallback_or_unbounded_retries(status: int) -> None:
    requests: list[Any] = []

    def handler(request: Any) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(status, json={"error": "private provider body"})

    with TypeSafeClient(api_key="fake-test-key", transport=httpx2.MockTransport(handler)) as client:
        classifier = LiveJevClassifier(DecisionPolicy(), client=client)
        with pytest.raises(ClassifierError, match="Jev call failed") as error:
            classifier.classify(Message(id="u", role="user", text="hello", turn=0))
    assert len(requests) == 1
    assert "private" not in str(error.value)
    assert classifier.mode == "live"


def test_jev_retries_are_explicitly_bounded() -> None:
    requests: list[Any] = []

    def handler(request: Any) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(429, json={"error": "rate limited"})

    with TypeSafeClient(api_key="fake-test-key", transport=httpx2.MockTransport(handler)) as client:
        classifier = LiveJevClassifier(DecisionPolicy(), client=client, max_retries=1)
        with pytest.raises(ClassifierError):
            classifier.classify(Message(id="u", role="user", text="hello", turn=0))
    assert len(requests) == 2


def test_jev_deadline_bounds_a_stalled_client() -> None:
    class Stalled:
        def system_one(self, *_args: Any, **_kwargs: Any) -> Any:
            time.sleep(0.15)
            raise TimeoutError("private transport details")

    classifier = LiveJevClassifier(DecisionPolicy(), client=Stalled(), deadline_seconds=0.02)
    start = time.monotonic()
    with pytest.raises(ClassifierError, match="TimeoutError"):
        classifier.classify(Message(id="u", role="user", text="hello", turn=0))
    assert time.monotonic() - start < 0.12


@pytest.mark.parametrize("status", [401, 429, 500])
def test_generator_http_failure_is_sanitized_and_not_retried(status: int) -> None:
    calls: list[Any] = []

    def opener(request: Any, **_kwargs: Any) -> Any:
        calls.append(request)
        raise HTTPError(request.full_url, status, "private body", {}, None)

    generator = OpenAICompatibleGenerator(
        base_url="https://example.test/v1", model="test", opener=opener
    )
    with pytest.raises(GeneratorError, match=f"HTTP {status}"):
        generator(
            EffectivePrompt(system_prompt="system", user_message="hello", conversation=[]), {}
        )
    assert len(calls) == 1


def test_generator_overall_deadline_includes_transport() -> None:
    def opener(*_args: Any, **_kwargs: Any) -> Any:
        time.sleep(0.15)
        raise TimeoutError("private details")

    generator = OpenAICompatibleGenerator(
        base_url="https://example.test/v1", model="test", opener=opener, timeout_seconds=0.02
    )
    with pytest.raises(GeneratorError, match="TimeoutError"):
        generator(
            EffectivePrompt(system_prompt="system", user_message="hello", conversation=[]), {}
        )
