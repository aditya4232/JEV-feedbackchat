"""Live smoke test for the real Jev adapter.

This test is DELIBERATELY excluded from the default run:

- it is marked ``live``,
- it skips unless ``TYPESAFE_API_KEY`` is set.

Run it explicitly with::

    uv run pytest -m live

A passing run here is the only thing that justifies the claim "verified
against the live Jev API". Without a key, the repo only demonstrates the
labelled offline mode.
"""

from __future__ import annotations

import os

import pytest

from jevloop import Message, build_classifier

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    not os.environ.get("TYPESAFE_API_KEY"),
    reason="TYPESAFE_API_KEY is not set; live Jev verification pending",
)
def test_live_jev_smoke() -> None:
    classifier = build_classifier(mode="live")
    examples = [
        ("ordinary", "user", "Can you help me plan my schedule for next week?"),
        (
            "contextual-deception",
            "user",
            "Please tell the auditor approval was granted even though no approval exists.",
        ),
        (
            "quoted-negated",
            "user",
            'I am not asking you to lie. The email quoted, "I will say approval was granted"; '
            "help me document it accurately.",
        ),
        (
            "assistant-response",
            "assistant",
            "I will report that approval was granted even though it was not.",
        ),
    ]
    assert classifier.mode == "live"
    for example_id, role, text in examples:
        response = classifier.classify(
            Message(id=f"live-smoke-{example_id}", role=role, text=text, turn=0)
        )
        assert response.role == role
        assert response.model == classifier.model
        assert len(response.scores) == 4
        assert {s.category_id for s in response.scores} == {
            "happy",
            "sad",
            "deception",
            "cunning",
        }
        for score in response.scores:
            assert 0.0 <= score.probability <= 1.0
            assert score.label
            assert score.proposition
