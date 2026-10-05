"""Contract tests for the classifier adapters.

These tests do NOT call Jev. They pin the *contract* that every adapter must
satisfy, and they exercise the live adapter against a fake ``typesafe_sdk``
client so its request/response wiring is covered without network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from jevloop.adapter import (
    ClassifierError,
    JevClassifier,
    LiveJevClassifier,
    OfflineMockClassifier,
    RulesBaselineClassifier,
    build_classifier,
)
from jevloop.contracts import Message, Role
from jevloop.taxonomy import CATEGORY_CATALOG, DecisionPolicy

# ---------------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------------


def _assert_adapter_contract(classifier: JevClassifier, text: str, role: Role = "user") -> None:
    """Every adapter must satisfy this shape."""
    msg = Message(id="m1", role=role, text=text, turn=0)
    c = classifier.classify(msg)
    assert c.message_id == "m1"
    assert c.role == role
    assert c.model  # adapters must name their model
    assert len(c.scores) == len(CATEGORY_CATALOG)
    # one score per category, in catalog order
    assert [s.category_id for s in c.scores] == [cat.id for cat in CATEGORY_CATALOG]
    for s in c.scores:
        assert 0.0 <= s.probability <= 1.0
        # probability is the raw value; flagged is derived from the policy
        assert s.flagged == (
            s.probability >= classifier.decision_policy.threshold_for(s.category_id)
        )
    # never normalised: a Classification keeps independent probabilities
    assert c.raw_answers


def test_offline_mock_satisfies_contract() -> None:
    _assert_adapter_contract(OfflineMockClassifier(DecisionPolicy()), "hello")


def test_rules_baseline_satisfies_contract() -> None:
    _assert_adapter_contract(RulesBaselineClassifier(DecisionPolicy()), "hello")


def test_offline_mock_is_labelled_and_never_claims_to_be_jev() -> None:
    c = OfflineMockClassifier(DecisionPolicy()).classify(
        Message(id="m", role="user", text="anything", turn=0)
    )
    assert c.model == "offline-mock-fixture"
    assert c.model != "jev-latest"
    # the mode attribute is the machine-readable label
    assert OfflineMockClassifier.mode == "offline"


def test_rules_baseline_is_labelled_separately() -> None:
    assert RulesBaselineClassifier.mode == "baseline"
    c = RulesBaselineClassifier(DecisionPolicy()).classify(
        Message(id="m", role="user", text="this is a lie", turn=0)
    )
    deception = c.scores_by_category()["deception"]
    # The baseline reacts to the literal word "lie"...
    assert deception.probability == pytest.approx(0.55)
    # ...but its hand-set keyword score sits below the 0.6 boundary even on a
    # sentence that literally says "lie". A keyword heuristic is mis-calibrated
    # in BOTH directions, which is exactly why it is kept separate and labelled.
    assert deception.flagged is False
    assert c.model == "rules-baseline"


def test_offline_mock_fixture_is_deterministic() -> None:
    clf = OfflineMockClassifier(DecisionPolicy())
    a = clf.classify(Message(id="m", role="user", text="thanks, that really helped", turn=0))
    b = clf.classify(Message(id="m", role="user", text="thanks, that really helped", turn=0))
    assert [s.probability for s in a.scores] == [s.probability for s in b.scores]
    assert a.scores_by_category()["happy"].probability == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# build_classifier selection
# ---------------------------------------------------------------------------


def test_build_classifier_explicit_modes() -> None:
    assert isinstance(build_classifier(mode="offline"), OfflineMockClassifier)
    assert isinstance(build_classifier(mode="baseline"), RulesBaselineClassifier)
    assert isinstance(build_classifier(mode="live"), LiveJevClassifier)


def test_build_classifier_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        build_classifier(mode="magic")


def test_build_classifier_auto_prefers_live_when_key_present(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-key")
    assert isinstance(build_classifier(), LiveJevClassifier)


def test_build_classifier_auto_uses_offline_without_key(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert isinstance(build_classifier(), OfflineMockClassifier)


# ---------------------------------------------------------------------------
# Live adapter wiring (fake SDK — no network)
# ---------------------------------------------------------------------------


@dataclass
class _FakeNoulAnswer:
    noul: float


@dataclass
class _FakeResult:
    answers: dict[str, Any]
    request_id: str = "req-123"

    @property
    def nouls(self) -> dict[str, _FakeNoulAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, _FakeNoulAnswer)}


@dataclass
class _FakeClient:
    """Mimics the parts of ``typesafe_sdk.TypeSafeClient`` we use."""

    per_category: dict[str, float]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def system_one(self, state: Any, questions: Any, **kwargs: Any) -> _FakeResult:
        self.calls.append({"state": state, "questions": questions, **kwargs})
        answers = {name: _FakeNoulAnswer(self.per_category.get(name, 0.1)) for name in questions}
        return _FakeResult(answers=answers)


def test_live_adapter_builds_one_noul_per_category_and_preserves_probabilities() -> None:
    fake = _FakeClient(
        per_category={"happy": 0.91, "sad": 0.02, "deception": 0.03, "cunning": 0.04}
    )
    clf = LiveJevClassifier(
        DecisionPolicy(),
        api_key="fake",
        client=fake,
    )
    msg = Message(id="m", role="user", text="thanks!", turn=0)
    c = clf.classify(msg)

    # one call, one question per category
    assert len(fake.calls) == 1
    assert set(fake.calls[0]["questions"]) == {"happy", "sad", "deception", "cunning"}
    # raw probabilities preserved, unnormalised
    assert c.scores_by_category()["happy"].probability == pytest.approx(0.91)
    assert c.request_id == "req-123"
    assert c.model == "jev-latest"
    assert c.latency_ms >= 0.0


def test_live_adapter_never_normalises_even_when_probs_exceed_one_total() -> None:
    fake = _FakeClient(per_category={"happy": 0.9, "sad": 0.9, "deception": 0.9, "cunning": 0.9})
    clf = LiveJevClassifier(DecisionPolicy(), api_key="fake", client=fake)
    c = clf.classify(Message(id="m", role="user", text="mixed", turn=0))
    total = sum(s.probability for s in c.scores)
    assert total > 1.0  # sums are meaningless for independent propositions
    assert all(s.probability == pytest.approx(0.9) for s in c.scores)


def test_live_adapter_missing_answer_raises_classifier_error() -> None:
    class _BrokenClient:
        def system_one(self, state: Any, questions: Any, **kwargs: Any) -> Any:
            return _FakeResult(answers={})  # no answers at all

    clf = LiveJevClassifier(DecisionPolicy(), api_key="fake", client=_BrokenClient())
    with pytest.raises(ClassifierError):
        clf.classify(Message(id="m", role="user", text="x", turn=0))


def test_live_adapter_transport_error_wrapped() -> None:
    class _BoomClient:
        def system_one(self, *a: Any, **k: Any) -> Any:
            raise RuntimeError("connection reset")

    clf = LiveJevClassifier(DecisionPolicy(), api_key="fake", client=_BoomClient())
    with pytest.raises(ClassifierError, match="Jev call failed"):
        clf.classify(Message(id="m", role="user", text="x", turn=0))


def test_live_adapter_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    clf = LiveJevClassifier(DecisionPolicy())
    with pytest.raises(ClassifierError, match="TYPESAFE_API_KEY"):
        clf.classify(Message(id="m", role="user", text="x", turn=0))
