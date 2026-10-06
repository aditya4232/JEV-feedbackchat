"""Typed contracts for the feedback loop.

These are the agent-facing and host-facing schemas. Everything that crosses a
boundary (CLI input, HTTP body, classifier call, generator call, persisted
session) is one of the dataclasses defined here. This is what makes the
core middleware actually middleware-shaped, rather than a script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    """One complete message in the conversation.

    We classify COMPLETE messages, not streaming fragments. Every message
    carries its turn order and any revision relationships explicitly.
    """

    id: str
    role: Role
    text: str
    turn: int  # 0-indexed
    revision_of: str | None = None  # message.id this one revises
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("message id must be a non-empty string")
        if self.role not in ("user", "assistant", "system"):
            raise ValueError(f"unsupported message role: {self.role!r}")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("message text must be a non-empty string")
        if isinstance(self.turn, bool) or not isinstance(self.turn, int) or self.turn < 0:
            raise ValueError("message turn must be a non-negative integer")
        if self.revision_of is not None and (
            not isinstance(self.revision_of, str) or not self.revision_of.strip()
        ):
            raise ValueError("revision_of must be a non-empty message id when provided")
        if not isinstance(self.metadata, dict):
            raise ValueError("message metadata must be a dictionary")


@dataclass(frozen=True)
class CategoryScore:
    category_id: str
    label: str
    proposition: str
    probability: float  # P(this proposition holds for this message)
    flagged: bool  # probability >= DecisionPolicy.threshold_for(category_id)
    abstained: bool = False  # classifier chose not to commit (e.g. ambiguous)
    decision_boundary: float | None = None
    decision_provisional: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.category_id, str)
            or not self.category_id.strip()
            or not isinstance(self.label, str)
            or not self.label.strip()
            or not isinstance(self.proposition, str)
            or not self.proposition.strip()
        ):
            raise ValueError("category score id, label and proposition must be non-empty")
        if (
            isinstance(self.probability, bool)
            or not isinstance(self.probability, (int, float))
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
        ):
            raise ValueError("category probability must be finite and in [0,1]")
        if not isinstance(self.flagged, bool) or not isinstance(self.abstained, bool):
            raise ValueError("flagged and abstained must be booleans")
        if self.flagged and self.abstained:
            raise ValueError("an abstained category cannot be flagged")
        if self.decision_boundary is not None and (
            isinstance(self.decision_boundary, bool)
            or not isinstance(self.decision_boundary, (int, float))
            or not math.isfinite(self.decision_boundary)
            or not 0.0 <= self.decision_boundary <= 1.0
        ):
            raise ValueError("decision_boundary must be finite and in [0,1]")
        if not isinstance(self.decision_provisional, bool):
            raise ValueError("decision_provisional must be a boolean")
        if self.decision_boundary is not None:
            expected_flag = not self.abstained and self.probability >= self.decision_boundary
            if self.flagged != expected_flag:
                raise ValueError(
                    "flagged must match probability, abstention, and decision_boundary"
                )

    def as_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "label": self.label,
            "proposition": self.proposition,
            "probability": round(self.probability, 4),
            "flagged": self.flagged,
            "abstained": self.abstained,
            "decision_boundary": self.decision_boundary,
            "decision_provisional": self.decision_provisional,
        }


@dataclass
class Classification:
    """Result of classifying one message with Jev.

    Probabilities are NOT normalised. Each value is the Noul probability of
    its own proposition; multiple categories may be true at once.
    """

    message_id: str
    role: Role
    model: str
    request_id: str | None
    scores: list[CategoryScore]
    raw_answers: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def scores_by_category(self) -> dict[str, CategoryScore]:
        return {s.category_id: s for s in self.scores}

    def flagged_categories(self) -> list[str]:
        return [s.category_id for s in self.scores if s.flagged and not s.abstained]

    def as_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "model": self.model,
            "request_id": self.request_id,
            "latency_ms": round(self.latency_ms, 2),
            "scores": [s.as_dict() for s in self.scores],
            "raw_answers": self.raw_answers,
            "flagged": self.flagged_categories(),
        }


@dataclass
class AdaptationStep:
    """One named transformation applied to the effective prompt.

    The policy is a small, named, versioned transformation. Each step records
    WHAT changed, WHY it changed, and the SOURCE signal that triggered it.
    """

    name: str  # e.g. "acknowledge_sadness"
    policy_id: str  # e.g. "jevloop.adapt.v1"
    policy_version: str  # e.g. "1.0.0"
    instruction_added: str | None
    rationale: str  # human-readable: which Jev signal triggered this
    source_signal: dict[str, Any]  # machine-readable category id, score, and message role


@dataclass
class AdaptedPrompt:
    """Result of running the policy over the agent's effective prompt.

    ``effective_system_prompt`` is what the generator will actually see. The
    ORIGINAL user message, task, and trusted base instructions are preserved
    (we only add or remove named addenda; we never rewrite the body of the
    user message itself).
    """

    original_system_prompt: str
    original_user_message: str
    effective_system_prompt: str
    addenda: list[AdaptationStep] = field(default_factory=list)
    policy_id: str = "jevloop.adapt.v1"
    policy_version: str = "1.0.0"

    def summary(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "effective_system_prompt": self.effective_system_prompt,
            "addenda": [
                {
                    "name": s.name,
                    "added": s.instruction_added,
                    "rationale": s.rationale,
                    "source_signal": s.source_signal,
                }
                for s in self.addenda
            ],
        }


@dataclass
class FeedbackPayload:
    """The structured feedback that the middleware returns to the agent.

    This is what the host orchestrator consumes to (a) drive prompt
    rephrasing on the next turn and (b) optionally decide to ask for
    revision on the current turn.
    """

    classification: Classification
    adapted_prompt: AdaptedPrompt
    next_turn_feedback: dict[str, Any]
    should_revise: bool
    revision_rationale: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification.as_dict(),
            "adapted_prompt": self.adapted_prompt.summary(),
            "next_turn_feedback": self.next_turn_feedback,
            "should_revise": self.should_revise,
            "revision_rationale": self.revision_rationale,
        }


@dataclass
class EffectivePrompt:
    """The full effective prompt sent to the generator.

    The generator must see this exact value. This is what we record in the
    session so a reviewer can confirm that the adapted prompt really was
    the one used.
    """

    system_prompt: str
    user_message: str
    conversation: list[Message]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompt": self.system_prompt,
            "user_message": self.user_message,
            "conversation": [
                {
                    "id": m.id,
                    "role": m.role,
                    "turn": m.turn,
                    "text": m.text,
                    "revision_of": m.revision_of,
                }
                for m in self.conversation
            ],
        }


@dataclass
class GeneratorOutput:
    """What the generator returns.

    The generator may return a single ``text``, or a list of candidate
    revisions (when the loop asks for ``should_revise=True``). The
    middleware records revision relationships explicitly.
    """

    text: str
    id: str | None = None
    revised_from: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
