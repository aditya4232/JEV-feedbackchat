"""Typed contracts for the feedback loop.

These are the agent-facing and host-facing schemas. Everything that crosses a
boundary (CLI input, HTTP body, classifier call, generator call, persisted
session) is one of the dataclasses defined here. This is what makes the
core middleware actually middleware-shaped, rather than a script.
"""

from __future__ import annotations

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


@dataclass
class CategoryScore:
    category_id: str
    label: str
    proposition: str
    probability: float  # P(this proposition holds for this message)
    flagged: bool  # probability >= DecisionPolicy.threshold_for(category_id)
    abstained: bool = False  # classifier chose not to commit (e.g. ambiguous)

    def as_dict(self) -> dict[str, Any]:
        return {
            "category_id": self.category_id,
            "label": self.label,
            "proposition": self.proposition,
            "probability": round(self.probability, 4),
            "flagged": self.flagged,
            "abstained": self.abstained,
        }


@dataclass
class Classification:
    """Result of classifying one message with Jev.

    Probabilities are NOT normalised. They are independent propositions and
    can sum to anything.  ``confidence`` (when available) is Jev's own
    per-question confidence, kept distinct from ``probability`` because the
    two are not the same thing.
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
            "flagged": self.flagged_categories(),
        }


@dataclass
class AdaptationStep:
    """One named transformation applied to the effective prompt.

    The policy is a small, named, versioned transformation. Each step records
    WHAT changed, WHY it changed, and the SOURCE signal that triggered it.
    """

    name: str  # e.g. "acknowledge_distress"
    policy_id: str  # e.g. "jevloop.adapt.v1"
    policy_version: str  # e.g. "1.0.0"
    instruction_added: str | None
    instruction_removed: str | None
    rationale: str  # human-readable: which Jev signal triggered this
    source_signal: dict[
        str, Any
    ]  # machine-readable: e.g. {"category": "sad", "p": 0.81}


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
                    "removed": s.instruction_removed,
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
