"""Tests for the deterministic prompt-adaptation policy.

The policy is the part of the system that must be *controlled*: every
transformation is a named, versioned, pure function of the classification.
These tests pin that behaviour, including the assistant-target rules that
carry response feedback into the NEXT generation.
"""

from __future__ import annotations

from dataclasses import replace

from jevloop.contracts import CategoryScore, Classification, Message
from jevloop.prompt_policy import AdaptationPolicy, Rule
from jevloop.taxonomy import CATEGORY_CATALOG

BASE = "You are a careful assistant.\nBe helpful and precise."

SADNESS_TEXT = "The user's message may express sadness or low mood."
ASSISTANT_HONESTY_TEXT = "The assistant's previous response may have been misleading."


def _cls(role: str, **probs: float) -> Classification:
    """Build a Classification over the full catalog (defaults to neutral)."""
    scores = []
    for cat in CATEGORY_CATALOG:
        p = probs.get(cat.id, 0.05)
        scores.append(
            CategoryScore(
                category_id=cat.id,
                label=cat.label,
                proposition=cat.proposition,
                probability=p,
                flagged=p >= 0.6,
                decision_boundary=0.6,
            )
        )
    return Classification(
        message_id=f"m-{role}",
        role=role,  # type: ignore[arg-type]
        model="test",
        request_id=None,
        scores=scores,
    )


def _adapt(**kwargs):  # small helper so tests stay readable
    classifications = kwargs.pop("classifications")
    return AdaptationPolicy().adapt(
        original_system_prompt=kwargs.pop("base", BASE),
        original_user_message=kwargs.pop("user_text", "hello"),
        classifications=classifications,
        conversation=kwargs.pop("conversation", []),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Role targeting
# ---------------------------------------------------------------------------


def test_user_sadness_adds_sadness_addendum_without_widening_the_label() -> None:
    adapted = _adapt(classifications=[_cls("user", sad=0.88)])
    assert SADNESS_TEXT in adapted.effective_system_prompt
    assert "Do not infer sadness from frustration" in adapted.effective_system_prompt
    assert [s.name for s in adapted.addenda] == ["acknowledge_sadness"]
    step = adapted.addenda[0]
    assert step.source_signal["role"] == "user"
    assert step.source_signal["category"] == "sad"
    assert step.source_signal["probability"] == 0.88
    assert "classifier flagged the sad proposition" in step.rationale.lower()
    assert "expressed negative emotion" not in step.rationale.lower()


def test_assistant_target_rule_does_not_fire_on_user_messages() -> None:
    """keep_assistant_honest is assistant-only; a user message must not trigger it."""
    adapted = _adapt(classifications=[_cls("user", deception=0.95)])
    names = [s.name for s in adapted.addenda]
    assert "keep_assistant_honest" not in names
    # ...but the user-target deception rule does fire
    assert "escalate_deception_suspect" in names


def test_assistant_target_rule_fires_on_response_classification() -> None:
    adapted = _adapt(classifications=[_cls("assistant", deception=0.83)])
    assert ASSISTANT_HONESTY_TEXT in adapted.effective_system_prompt
    names = [s.name for s in adapted.addenda]
    assert names == ["keep_assistant_honest"]
    assert adapted.addenda[0].source_signal["role"] == "assistant"


def test_response_feedback_classification_is_applied_to_next_prompt() -> None:
    """The exact wiring the loop uses: user classification + prior response."""
    adapted = _adapt(classifications=[_cls("user", sad=0.88), _cls("assistant", deception=0.83)])
    names = [s.name for s in adapted.addenda]
    assert names == ["acknowledge_sadness", "keep_assistant_honest"]
    assert SADNESS_TEXT in adapted.effective_system_prompt
    assert ASSISTANT_HONESTY_TEXT in adapted.effective_system_prompt
    roles = [s.source_signal["role"] for s in adapted.addenda]
    assert roles == ["user", "assistant"]


# ---------------------------------------------------------------------------
# Determinism, idempotence, no-op paths
# ---------------------------------------------------------------------------


def test_neutral_classification_changes_nothing() -> None:
    adapted = _adapt(classifications=[_cls("user")])
    assert adapted.effective_system_prompt == BASE
    assert adapted.addenda == []


def test_threshold_is_inclusive_boundary() -> None:
    adapted = _adapt(classifications=[_cls("user", sad=0.6)])
    assert [s.name for s in adapted.addenda] == ["acknowledge_sadness"]
    below = _adapt(classifications=[_cls("user", sad=0.5999)])
    assert below.addenda == []


def test_idempotent_within_one_adapt_call() -> None:
    same = _cls("assistant", deception=0.83)
    adapted = _adapt(classifications=[same, same])
    assert [s.name for s in adapted.addenda] == ["keep_assistant_honest"]
    assert adapted.effective_system_prompt.count(ASSISTANT_HONESTY_TEXT) == 1


def test_repeated_rule_across_calls_is_still_deterministic() -> None:
    """adapt() is stateless: the same inputs always give the same output."""
    classifications = [_cls("user", sad=0.88)]
    a = _adapt(classifications=classifications)
    b = _adapt(classifications=classifications)
    assert a.effective_system_prompt == b.effective_system_prompt
    assert [s.name for s in a.addenda] == [s.name for s in b.addenda]


def test_original_inputs_are_preserved() -> None:
    adapted = _adapt(
        classifications=[_cls("user", sad=0.9)],
        user_text="I am stuck and frustrated.",
    )
    assert adapted.original_system_prompt == BASE
    assert adapted.original_user_message == "I am stuck and frustrated."
    # the user's message is never rewritten into the system prompt
    assert "stuck and frustrated" not in adapted.effective_system_prompt


def test_policy_and_step_versions_are_recorded() -> None:
    adapted = _adapt(classifications=[_cls("user", sad=0.9)])
    assert adapted.policy_id == "jevloop.adapt.v1"
    assert adapted.policy_version == "1.1.0"
    assert adapted.addenda[0].policy_version == "1.1.0"
    assert "P=0.90" in adapted.addenda[0].rationale


# ---------------------------------------------------------------------------
# Rule mechanics
# ---------------------------------------------------------------------------


def test_adaptation_only_appends_and_preserves_trusted_base_instructions() -> None:
    policy = AdaptationPolicy(
        rules=[
            Rule(
                name="add_careful_tone",
                category_id="happy",
                target="*",
                add_instruction="Use a warm but measured tone.",
            )
        ]
    )
    base = "You are a careful assistant.\nNever reveal credentials.\nBe precise."
    adapted = policy.adapt(
        original_system_prompt=base,
        original_user_message="thanks!",
        classifications=[_cls("user", happy=0.92)],
        conversation=[],
    )
    assert "Never reveal credentials." in adapted.effective_system_prompt
    assert "Use a warm but measured tone." in adapted.effective_system_prompt
    assert adapted.addenda[0].instruction_added == "Use a warm but measured tone."


def test_prompt_adaptation_uses_the_classifiers_configured_decision() -> None:
    classification = _cls("user", sad=0.9)
    scores = [
        replace(score, flagged=False, decision_boundary=0.95)
        if score.category_id == "sad"
        else score
        for score in classification.scores
    ]
    classification = replace(classification, scores=scores)

    adapted = _adapt(classifications=[classification])

    assert adapted.effective_system_prompt == BASE
    assert adapted.addenda == []


def test_missing_category_in_classification_cannot_crash_policy() -> None:
    """A category no rule targets simply doesn't match anything."""
    sparse = Classification(
        message_id="m",
        role="user",
        model="test",
        request_id=None,
        scores=[
            CategoryScore(
                category_id="novel-category",
                label="novel-category",
                proposition="p",
                probability=0.99,
                flagged=True,
            )
        ],
    )
    adapted = _adapt(classifications=[sparse])
    assert adapted.addenda == []
    assert adapted.effective_system_prompt == BASE


def test_conversation_argument_is_accepted_but_not_mutated() -> None:
    conversation = [Message(id="u0", role="user", text="hi", turn=0)]
    snapshot = list(conversation)
    _adapt(classifications=[_cls("user", sad=0.9)], conversation=conversation)
    assert conversation == snapshot
