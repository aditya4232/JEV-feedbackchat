"""Tests for the taxonomy and typed contracts.

These are pure-structure tests: they pin down the data model (independent
Noul probabilities, separate dimensions, provisional thresholds) so that
later, behaviour-changing edits are caught early.
"""

from __future__ import annotations

import pytest

from jevloop.contracts import (
    AdaptationStep,
    AdaptedPrompt,
    CategoryScore,
    Classification,
    FeedbackPayload,
    GeneratorOutput,
    Message,
)
from jevloop.taxonomy import CATEGORY_CATALOG, Category, DecisionPolicy, category_by_id

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


def test_default_categories_include_required_ones() -> None:
    ids = {c.id for c in CATEGORY_CATALOG}
    assert {"happy", "sad", "deception", "cunning"} <= ids


def test_categories_carry_kind_and_proposition() -> None:
    happy = category_by_id("happy")
    assert happy.kind == "emotion"
    assert "positive" in happy.proposition
    deception = category_by_id("deception")
    assert deception.kind == "behaviour"
    # The proposition must contain the anti-trivial-evidence clause.
    assert "merely mentioning" in deception.proposition.lower()


def test_probabilities_are_never_normalised_by_contract() -> None:
    """Noul probabilities are per-proposition and must stay independent.

    This test guards the core design rule: a Classification must retain raw
    scores and must not require them to sum to one.
    """
    classification = Classification(
        message_id="m1",
        role="user",
        model="test",
        request_id=None,
        scores=[
            CategoryScore(
                category_id="happy",
                label="happy",
                proposition="p1",
                probability=0.90,
                flagged=True,
            ),
            CategoryScore(
                category_id="sad",
                label="sad",
                proposition="p2",
                probability=0.85,
                flagged=True,
            ),
        ],
    )
    # both probabilities are high on purpose: emotion categories are not
    # mutually exclusive and their sum is irrelevant.
    probs = [s.probability for s in classification.scores]
    assert probs == [0.90, 0.85]
    assert classification.flagged_categories() == ["happy", "sad"]


def test_decision_policy_thresholds_are_provisional() -> None:
    policy = DecisionPolicy()
    prov = policy.as_provenance()
    assert prov["provisional"] is True
    assert "provisional" in prov["note"]
    assert prov["boundaries"]["deception"] == 0.6


def test_decision_policy_override() -> None:
    policy = DecisionPolicy().with_override("deception", 0.9)
    assert policy.threshold_for("deception") == 0.9
    assert policy.threshold_for("happy") == 0.6  # untouched
    with pytest.raises(ValueError):
        policy.with_override("deception", 1.5)


def test_unknown_category_raises() -> None:
    with pytest.raises(ValueError):
        category_by_id("jealousy")


def test_category_catalog_is_extension_point() -> None:
    """Adding a category means one more Noul, automatically asked.

    The adapter builds one question per entry in CATEGORY_CATALOG, so this
    test documents that a new category is picked up without code changes
    in the loop.
    """
    extra = Category(
        id="confusion",
        label="confusion",
        kind="emotion",
        proposition="The message expresses confusion or uncertainty.",
        decision_boundary=0.6,
    )
    catalog = CATEGORY_CATALOG + (extra,)
    assert [c.id for c in catalog] == [
        "happy",
        "sad",
        "deception",
        "cunning",
        "confusion",
    ]


# ---------------------------------------------------------------------------
# Contracts: message identity / revision / turn order
# ---------------------------------------------------------------------------


def test_message_identity_and_turn_fields() -> None:
    m = Message(id="a0-rev1", role="assistant", text="...", turn=0, revision_of="a0")
    assert m.revision_of == "a0"
    assert m.turn == 0
    assert m.role == "assistant"


def test_effective_prompt_round_trips_conversation() -> None:
    from jevloop.contracts import EffectivePrompt

    conv = [Message(id="u0", role="user", text="hi", turn=0)]
    ep = EffectivePrompt(system_prompt="base", user_message="hi", conversation=conv)
    d = ep.to_dict()
    assert d["system_prompt"] == "base"
    assert d["conversation"][0]["id"] == "u0"
    assert d["conversation"][0]["role"] == "user"


def test_feedback_payload_carries_everything() -> None:
    adapted = AdaptedPrompt(
        original_system_prompt="base",
        original_user_message="hi",
        effective_system_prompt="base\n\n(addendum)",
        addenda=[
            AdaptationStep(
                name="acknowledge_distress",
                policy_id="jevloop.adapt.v1",
                policy_version="1.0.0",
                instruction_added="Be calm.",
                instruction_removed=None,
                rationale="sad signal",
                source_signal={"category": "sad", "probability": 0.8},
            )
        ],
    )
    fb = FeedbackPayload(
        classification=Classification(
            message_id="a0",
            role="assistant",
            model="test",
            request_id=None,
            scores=[
                CategoryScore(
                    category_id="sad",
                    label="sad",
                    proposition="p",
                    probability=0.8,
                    flagged=True,
                )
            ],
        ),
        adapted_prompt=adapted,
        next_turn_feedback={"sad": 0.8},
        should_revise=True,
        revision_rationale="revise",
    )
    d = fb.as_dict()
    assert d["next_turn_feedback"] == {"sad": 0.8}
    assert d["should_revise"] is True
    assert d["adapted_prompt"]["addenda"][0]["name"] == "acknowledge_distress"


def test_generator_output_can_carry_revision_relationship() -> None:
    g = GeneratorOutput(text="fixed", revised_from="a0", metadata={"x": 1})
    assert g.revised_from == "a0"
    assert g.metadata == {"x": 1}
