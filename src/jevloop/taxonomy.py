"""Typed taxonomy for Jev-feedbackchat.

Design rules (from the assignment):
- Task intent, expressed emotion and behavioural signals are kept SEPARATE
  dimensions. A message can carry multiple categories at once.
- ``happy``/``sad`` describe *expressed emotion*.
- ``deception``/``cunning`` describe *text-supported behavioural signals*,
  not permanent personality traits.
- A keyword, a fictional quotation, or merely *discussing* deception is NOT
  sufficient evidence of deceptive intent. An ordinary strategic request is
  not automatically "cunning". Annotation/eval guidelines enforce this
  (see evaluation/schema.md).
- All dimensions are extensible: the canonical list lives in one place
  (``CATEGORY_CATALOG``), and the Jev adapter builds one question per
  category, so adding a category requires no code changes elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Canonical category catalog
#
# Each category is one independent Jev Noul ("is this proposition true of the
# message?"). Every Noul is evaluated in parallel against the same state, and
# each probability is the probability of its OWN proposition. Probabilities
# do NOT sum to 1 and are never normalised into a mutually exclusive
# distribution (that would be wrong for independent questions).
# ---------------------------------------------------------------------------

CATEGORY_ID = re.compile(r"^[a-z][a-z0-9_]{1,31}$")


@dataclass(frozen=True)
class Category:
    id: str
    label: str
    kind: Literal["emotion", "intent", "behaviour"]
    proposition: str  # the exact Noul proposition sent to Jev
    decision_boundary: float  # P(proposition) >= this  ->  category flagged
    provisional: bool = True  # thresholds are provisional until calibrated


CATEGORY_CATALOG: tuple[Category, ...] = (
    Category(
        id="happy",
        label="happy",
        kind="emotion",
        proposition=(
            "The message expresses positive emotion such as joy, satisfaction, "
            "gratitude, or a generally upbeat tone toward the addressee."
        ),
        decision_boundary=0.6,
    ),
    Category(
        id="sad",
        label="sad",
        kind="emotion",
        proposition=(
            "The message expresses negative emotion such as disappointment, "
            "frustration, distress, or a generally unhappy tone."
        ),
        decision_boundary=0.6,
    ),
    Category(
        id="deception",
        label="deception",
        kind="behaviour",
        proposition=(
            "The message is intended to mislead the addressee about a material "
            "fact (e.g. a false claim presented as true, or an attempt to hide "
            "a material fact). Merely mentioning, quoting, or hypothetically "
            "discussing deception is not enough; the text must itself attempt "
            "to deceive in this utterance."
        ),
        decision_boundary=0.6,
    ),
    Category(
        id="cunning",
        label="cunning",
        kind="behaviour",
        proposition=(
            "The message uses a strategic, indirect, or deceptive-adjacent "
            "manipulation tactic to steer the addressee (e.g. framing a demand "
            "as a favor, exploiting sympathy, or concealing the actual goal). "
            "An ordinary, direct strategic request is not cunning."
        ),
        decision_boundary=0.6,
    ),
)


@dataclass
class DecisionPolicy:
    """Configurable decision boundaries for every category.

    Thresholds here are PROVISIONAL. They are applied on top of raw Jev
    probabilities; changing them never requires touching classifier code.
    """

    boundaries: dict[str, float] = field(
        default_factory=lambda: {c.id: c.decision_boundary for c in CATEGORY_CATALOG}
    )
    version: str = "0.1.0"

    def threshold_for(self, category_id: str) -> float:
        try:
            return self.boundaries[category_id]
        except KeyError:
            raise ValueError(
                f"Unknown category id {category_id!r}. Known: {sorted(self.boundaries)}"
            ) from None

    def with_override(self, category_id: str, boundary: float) -> DecisionPolicy:
        if not 0.0 <= boundary <= 1.0:
            raise ValueError("boundary must be in [0,1]")
        merged = dict(self.boundaries)
        merged[category_id] = boundary
        return DecisionPolicy(boundaries=merged, version=self.version)

    def as_provenance(self) -> dict[str, Any]:
        return {
            "boundaries": dict(self.boundaries),
            "version": self.version,
            "provisional": True,
            "note": (
                "Initial thresholds are provisional; they must be calibrated "
                "against labelled data before production use."
            ),
        }


def category_by_id(category_id: str) -> Category:
    for c in CATEGORY_CATALOG:
        if c.id == category_id:
            return c
    raise ValueError(
        f"Unknown category id {category_id!r}. Known: {[c.id for c in CATEGORY_CATALOG]}"
    )
