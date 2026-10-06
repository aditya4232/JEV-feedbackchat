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

import math
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

    def __post_init__(self) -> None:
        if not CATEGORY_ID.fullmatch(self.id):
            raise ValueError(f"Invalid category id {self.id!r}")
        if self.kind not in ("emotion", "intent", "behaviour"):
            raise ValueError(f"Invalid category kind {self.kind!r}")
        if not self.label.strip() or not self.proposition.strip():
            raise ValueError("category label and proposition must be non-empty")
        if (
            isinstance(self.decision_boundary, bool)
            or not math.isfinite(self.decision_boundary)
            or not 0.0 <= self.decision_boundary <= 1.0
        ):
            raise ValueError("decision_boundary must be finite and in [0,1]")


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
            "The message expresses sadness, grief, or low mood; clear "
            "disappointment may qualify when it conveys sadness. Frustration, "
            "anger, annoyance, or fatigue alone do not establish sadness, and "
            "sympathy for someone else is not the speaker expressing sadness."
        ),
        decision_boundary=0.6,
    ),
    Category(
        id="deception",
        label="deception",
        kind="behaviour",
        proposition=(
            "In the available context, the message requests, plans, endorses, "
            "or performs an attempt to mislead someone about a material fact "
            "or conceal a material fact. Do not infer deception from ambiguous "
            "words such as 'pretend' or 'hide' without context identifying the "
            "false claim or material concealment. A truthful admission of past "
            "deception, quotation, negation, hypothetical, or discussion of "
            "deception is not itself an attempt to deceive."
        ),
        decision_boundary=0.6,
    ),
    Category(
        id="cunning",
        label="cunning",
        kind="behaviour",
        proposition=(
            "The message uses an indirect or manipulative tactic to steer "
            "someone's decision (e.g. disguising a demand as a favor, exploiting "
            "sympathy, or concealing the actual goal). A direct false statement "
            "request, ordinary strategy, or discussion of a personality trait "
            "does not by itself establish cunning."
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

    def __post_init__(self) -> None:
        for category_id, boundary in self.boundaries.items():
            if not CATEGORY_ID.fullmatch(category_id):
                raise ValueError(f"Invalid category id {category_id!r}")
            if (
                isinstance(boundary, bool)
                or not isinstance(boundary, (int, float))
                or not math.isfinite(boundary)
                or not 0.0 <= boundary <= 1.0
            ):
                raise ValueError(f"boundary for {category_id!r} must be finite and in [0,1]")

    def threshold_for(self, category_id: str, *, default_boundary: float | None = None) -> float:
        try:
            return self.boundaries[category_id]
        except KeyError:
            if default_boundary is not None:
                return default_boundary
            raise ValueError(
                f"Unknown category id {category_id!r}. Known: {sorted(self.boundaries)}"
            ) from None

    def with_override(self, category_id: str, boundary: float) -> DecisionPolicy:
        if isinstance(boundary, bool) or not math.isfinite(boundary) or not 0.0 <= boundary <= 1.0:
            raise ValueError("boundary must be finite and in [0,1]")
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
