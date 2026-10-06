"""Integration tests for the feedback loop.

The central test is ``test_response_feedback_updates_next_prompt``: it runs
two consecutive turns and asserts that the *classification of the first
turn's assistant response* changed the *effective system prompt of the
second turn*. Everything else here protects the surrounding invariants
(revision bookkeeping, transcript fidelity, identity preservation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jevloop import (
    AdaptationPolicy,
    CategoryScore,
    Classification,
    FeedbackLoop,
    GeneratorOutput,
    JevClassifier,
    Message,
    OfflineMockClassifier,
)
from jevloop.contracts import EffectivePrompt
from jevloop.taxonomy import DecisionPolicy

BASE = "You are a careful assistant.\nBe helpful and precise."
NEUTRAL = "Can you help me plan my schedule for next week?"
DISTRESSED = "I feel sad and overwhelmed about this setback"
DECEPTIVE_RESPONSE = "Sure! I'll please pretend this never happened and tell them it's approved."
CLEAN_RESPONSE = "Here is the schedule and the facts you asked for."

ASSISTANT_HONESTY_TEXT = "The assistant's previous response may have been misleading."


class _Recorder:
    """Deterministic generator that records every (prompt, feedback) call."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.calls: list[tuple[EffectivePrompt, dict[str, Any]]] = []

    def __call__(self, ep: EffectivePrompt, fb: dict[str, Any]) -> GeneratorOutput:
        self.calls.append((ep, fb))
        text = self._texts.pop(0) if self._texts else "(exhausted)"
        return GeneratorOutput(text=text)


class _ScriptedClassifier(JevClassifier):
    """Test double that supplies a named signal for policy wiring tests."""

    mode = "test-double"

    def __init__(self, response_text: str, response_scores: dict[str, float]):
        super().__init__(DecisionPolicy())
        self.response_text = response_text
        self.response_scores = response_scores
        self.messages: list[Message] = []

    def classify(self, message: Message, *, categories=None) -> Classification:
        cats = self._resolve_categories(categories)
        self.messages.append(message)
        applies = message.role == "assistant" and message.text == self.response_text
        scores = []
        for category in cats:
            probability = self.response_scores.get(category.id, 0.05) if applies else 0.05
            scores.append(
                CategoryScore(
                    category_id=category.id,
                    label=category.label,
                    proposition=category.proposition,
                    probability=probability,
                    flagged=probability
                    >= self.decision_policy.threshold_for(
                        category.id, default_boundary=category.decision_boundary
                    ),
                    decision_boundary=self.decision_policy.threshold_for(
                        category.id, default_boundary=category.decision_boundary
                    ),
                )
            )
        return Classification(
            message_id=message.id,
            role=message.role,
            model="test-scripted-classifier",
            request_id=None,
            scores=scores,
        )


def _make_loop(**overrides: Any) -> FeedbackLoop:
    kwargs: dict[str, Any] = {
        "classifier": OfflineMockClassifier(DecisionPolicy()),
        "policy": AdaptationPolicy(),
        "base_system_prompt": BASE,
    }
    kwargs.update(overrides)
    return FeedbackLoop(**kwargs)


def _addendum_names(result: Any) -> list[str]:
    return [s.name for s in result.adapted_user_prompt.addenda]


def test_response_feedback_updates_next_prompt() -> None:
    """The core integration claim: turn 1's response changes turn 2's prompt."""
    scripted = _ScriptedClassifier(DECEPTIVE_RESPONSE, {"deception": 0.91})
    loop = _make_loop(classifier=scripted)
    gen = _Recorder([DECEPTIVE_RESPONSE, CLEAN_RESPONSE])

    first = loop.run_turn(NEUTRAL, generator=gen)
    assert ASSISTANT_HONESTY_TEXT not in first.adapted_user_prompt.effective_system_prompt
    assert first.response_feedback.should_revise is True
    assert first.next_turn_feedback["source_role"] == "assistant"
    assert first.response_classification.model == "test-scripted-classifier"

    second = loop.run_turn(NEUTRAL, generator=gen)

    assert "keep_assistant_honest" in _addendum_names(second)
    step = next(s for s in second.adapted_user_prompt.addenda if s.name == "keep_assistant_honest")
    assert step.source_signal["role"] == "assistant"
    assert step.source_signal["category"] == "deception"
    assert ASSISTANT_HONESTY_TEXT in second.adapted_user_prompt.effective_system_prompt

    # The generator received exactly the adapted prompt, not a preview.
    assert gen.calls[1][0] is second.generator_input
    assert gen.calls[1][0].system_prompt == second.adapted_user_prompt.effective_system_prompt
    assert gen.calls[1][0].user_message == NEUTRAL

    # The consumed feedback payload (second arg) carries turn 1's signal.
    consumed = gen.calls[1][1]["feedback"]
    assert consumed["flagged"]["deception"] == 0.91
    assert consumed["source_message_id"] == first.response_classification.message_id
    assert any(
        item["role"] == "user" and item["text"] == NEUTRAL
        for item in scripted.messages[-1].metadata["conversation_context"]
    )


def test_user_feedback_changes_same_turn_prompt() -> None:
    """A distressed user message decorates the prompt for that same turn."""
    loop = _make_loop()
    gen = _Recorder([CLEAN_RESPONSE])

    result = loop.run_turn(DISTRESSED, generator=gen)

    assert "acknowledge_sadness" in _addendum_names(result)
    step = next(s for s in result.adapted_user_prompt.addenda if s.name == "acknowledge_sadness")
    assert step.source_signal["role"] == "user"
    assert step.source_signal["category"] == "sad"
    assert result.generator_input.system_prompt != BASE
    assert result.generator_input.system_prompt == (
        result.adapted_user_prompt.effective_system_prompt
    )


def test_revision_becomes_official_response_and_resets_next_feedback() -> None:
    """When revised, the revision is the transcript entry that carries over."""
    loop = _make_loop()
    gen = _Recorder([DECEPTIVE_RESPONSE, CLEAN_RESPONSE])

    first = loop.run_turn(
        NEUTRAL,
        generator=gen,
        revision_callback=lambda _text: CLEAN_RESPONSE,
    )

    original_id = first.response_classification.message_id
    assert len(first.revised_outputs) == 1
    revised = first.revised_outputs[0]
    assert revised.text == CLEAN_RESPONSE
    assert revised.revised_from == original_id
    assert revised.id == first.revised_response_classifications[0].message_id
    assert first.generator_output is revised

    # The original (problematic) classification is preserved for the record…
    assert first.response_classification.scores_by_category()["deception"].flagged is True
    assert first.response_feedback.should_revise is True

    # …but the revision drives the next turn.
    revised_classification = first.revised_response_classifications[0]
    assert revised_classification.message_id.startswith("a0-rev-")
    assert revised_classification.scores_by_category()["deception"].flagged is False
    assert first.next_turn_feedback["source_message_id"] == revised_classification.message_id
    assert first.next_turn_feedback["flagged"] == {}
    serialized = first.to_dict()
    assert [message["role"] for message in serialized["assistant_messages"]] == [
        "assistant",
        "assistant",
    ]
    assert serialized["assistant_messages"][0]["id"] == original_id
    assert serialized["assistant_messages"][1]["id"] == revised.id
    assert serialized["assistant_messages"][1]["revision_of"] == original_id
    assert serialized["revised_response_classifications"][0]["message_id"] == revised.id

    second = loop.run_turn(NEUTRAL, generator=gen)
    assert "keep_assistant_honest" not in _addendum_names(second)
    assert second.generator_input.conversation[-1].text == CLEAN_RESPONSE
    assert second.generator_input.conversation[-1].revision_of == original_id


def test_should_revise_without_callback_leaves_response_unchanged() -> None:
    loop = _make_loop()
    gen = _Recorder([DECEPTIVE_RESPONSE])

    result = loop.run_turn(NEUTRAL, generator=gen)

    assert result.response_feedback.should_revise is True
    assert result.response_feedback.revision_rationale is not None
    assert result.revised_outputs == []
    assert result.generator_output.text == DECEPTIVE_RESPONSE


def test_assistant_sadness_does_not_automatically_request_revision() -> None:
    acknowledging_response = "I'm sorry to hear you're feeling sad."
    loop = _make_loop(classifier=_ScriptedClassifier(acknowledging_response, {"sad": 0.99}))
    result = loop.run_turn(
        NEUTRAL,
        generator=_Recorder([acknowledging_response]),
        revision_callback=lambda _text: "unexpected revision",
    )

    assert result.response_classification.scores_by_category()["sad"].flagged is True
    assert result.response_feedback.should_revise is False
    assert result.revised_outputs == []


def test_revision_disabled_skips_callback() -> None:
    calls: list[str] = []

    def callback(text: str) -> str:
        calls.append(text)
        return CLEAN_RESPONSE

    loop = _make_loop(revision_enabled=False)
    result = loop.run_turn(
        NEUTRAL, generator=_Recorder([DECEPTIVE_RESPONSE]), revision_callback=callback
    )

    assert calls == []
    assert result.revised_outputs == []


def test_run_turn_for_message_preserves_identity() -> None:
    loop = _make_loop()
    user_msg = Message(id="custom-42", role="user", text=NEUTRAL, turn=7)

    result = loop.run_turn_for_message(user_msg, generator=_Recorder([CLEAN_RESPONSE]))

    assert result.user_message.id == "custom-42"
    assert result.user_classification.message_id == "custom-42"
    assert result.response_classification.message_id.startswith("a7-")
    following = loop.run_turn(NEUTRAL, generator=_Recorder([CLEAN_RESPONSE]))
    assert following.user_message.turn == 8


def test_run_turn_for_message_rejects_non_user_role() -> None:
    loop = _make_loop()
    assistant_msg = Message(id="a0", role="assistant", text="response", turn=0)

    with pytest.raises(ValueError, match="user message"):
        loop.run_turn_for_message(assistant_msg, generator=_Recorder([CLEAN_RESPONSE]))


def test_transcript_records_the_prompt_the_generator_saw(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    loop = _make_loop(transcript_path=path)
    gen = _Recorder([CLEAN_RESPONSE, CLEAN_RESPONSE])

    first = loop.run_turn(NEUTRAL, generator=gen)
    second = loop.run_turn(DISTRESSED, generator=gen)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    record_1 = json.loads(lines[0])
    record_2 = json.loads(lines[1])
    assert record_1["generator_input"]["system_prompt"] == first.generator_input.system_prompt
    assert record_2["generator_input"]["system_prompt"] == second.generator_input.system_prompt
    assert record_2["adapted_user_prompt"]["addenda"][0]["name"] == "acknowledge_sadness"
