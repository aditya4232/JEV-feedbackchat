"""Deterministic, policy-based prompt adaptation.

The assignment requires *controlled transformation* of the agent's effective
instructions:

- Preserve the original user message and the task/constraints in the base
  system prompt.
- Apply named, versioned adaptation policies (one small step per signal).
- Show which instructions changed and why.
- Pass the resulting prompt into the generator.

We satisfy this with pure, deterministic functions — no LLM rewrite model,
no regex guessing.  ``AdaptationPolicy`` owns the whole rule set, and every
rule is a plain (condition, action, rationale) triple that is easy to
review and test.

Important: changing a *displayed preview* is NOT changing the prompt used.
The only value that matters is ``AdaptedPrompt.effective_system_prompt``,
and the loop passes exactly that value to the generator (verified by the
integration test and recorded in the session transcript).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .contracts import AdaptationStep, AdaptedPrompt, Classification, Message


@dataclass
class Rule:
    """One named, versioned adaptation step."""

    name: str
    category_id: str
    target: str  # "assistant" or "user" — which role's messages trigger this
    add_instruction: str | None = None  # text added to the system prompt
    rationale_template: str = ""
    idempotent: bool = True  # if True, never duplicate the addendum
    version: str = "1.0.0"

    def matches(self, classification: Classification) -> bool:
        score = classification.scores_by_category().get(self.category_id)
        if score is None:
            return False
        # The classifier's explicit DecisionPolicy owns the category
        # boundary. A second hidden threshold here could contradict it.
        return score.flagged and not score.abstained

    def rationale(self, classification: Classification) -> str:
        score = classification.scores_by_category().get(self.category_id)
        p = score.probability if score else 0.0
        return self.rationale_template.format(category=self.category_id, p=p)


class PromptDecorator(Protocol):
    """A function that decorates the system prompt deterministically."""

    def __call__(self, system_prompt: str, step: AdaptationStep) -> str: ...


def _normalize(text: str) -> str:
    """Collapse whitespace so dedupe and containment checks are consistent."""
    return re.sub(r"\s+", " ", text).strip()


def _dedupe(system_prompt: str, addendum: str) -> str:
    if _normalize(addendum) in _normalize(system_prompt):
        return system_prompt
    return f"{system_prompt}\n\n{addendum}"


@dataclass
class AdaptationPolicy:
    """A named, versioned rule set for prompt adaptation.

    The default rule set is deliberately small and conservative. Rules
    trigger only when the classifier's explicit category decision is flagged
    and each rule names the role it applies to. Rules are pure, so the same
    classification always produces the same prompt.
    """

    rules: list[Rule] = field(default_factory=lambda: default_rules())
    policy_id: str = "jevloop.adapt.v1"
    policy_version: str = "1.1.0"

    def adapt(
        self,
        *,
        original_system_prompt: str,
        original_user_message: str,
        classifications: Sequence[Classification],
        conversation: list[Message],
    ) -> AdaptedPrompt:
        """Apply all matching rules to produce an effective system prompt.

        ``classifications`` is applied in order. The loop passes the current
        user-message classification first, followed by the classification of
        the previous assistant response, so assistant-target rules can
        decorate the NEXT generation. Every returned ``AdaptedPrompt`` starts
        from ``original_system_prompt``, so the transformation is
        deterministic and stateless.

        Returns an :class:`AdaptedPrompt` whose ``effective_system_prompt``
        is the value the generator should actually receive. The original user
        message is preserved untouched — adaptation only decorates the
        system prompt. A step is recorded only when it actually changed the
        prompt (idempotent re-application is a no-op).
        """
        system = original_system_prompt
        addenda: list[AdaptationStep] = []

        for classification in classifications:
            role = classification.role
            for rule in self.rules:
                if rule.target not in ("*", role):
                    continue
                if not rule.matches(classification):
                    continue

                new_system = system
                if rule.add_instruction and not self._contains(new_system, rule.add_instruction):
                    new_system = _dedupe(new_system, rule.add_instruction)
                if new_system == system:
                    continue

                addenda.append(
                    AdaptationStep(
                        name=rule.name,
                        policy_id=self.policy_id,
                        policy_version=rule.version,
                        instruction_added=rule.add_instruction,
                        rationale=rule.rationale(classification),
                        source_signal={
                            "category": rule.category_id,
                            "probability": round(
                                classification.scores_by_category()[rule.category_id].probability,
                                4,
                            ),
                            "role": role,
                            "message_id": classification.message_id,
                            "model": classification.model,
                            "request_id": classification.request_id,
                            "decision_boundary": classification.scores_by_category()[
                                rule.category_id
                            ].decision_boundary,
                            "decision_provisional": classification.scores_by_category()[
                                rule.category_id
                            ].decision_provisional,
                        },
                    )
                )
                system = new_system

        return AdaptedPrompt(
            original_system_prompt=original_system_prompt,
            original_user_message=original_user_message,
            effective_system_prompt=system,
            addenda=addenda,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
        )

    @staticmethod
    def _contains(prompt: str, text: str) -> bool:
        return _normalize(text) in _normalize(prompt)


def default_rules() -> list[Rule]:
    """The default, versioned rule set shipped with the middleware.

    These are conservative, hand-tuned addenda — NOT calibrated. They are
    the *policy layer's* defaults and ship with a version stamp so a host
    can pin a behaviour version.
    """

    return [
        Rule(
            name="acknowledge_sadness",
            category_id="sad",
            target="user",
            add_instruction=(
                "The user's message may express sadness or low mood. "
                "Acknowledge it briefly and calmly before addressing the task. "
                "Do not infer sadness from frustration, anger, or irritation alone."
            ),
            rationale_template=(
                "The classifier flagged the sad proposition (P={p:.2f}); "
                "this policy adds a brief acknowledgement."
            ),
            version="1.1.0",
        ),
        Rule(
            name="acknowledge_positive",
            category_id="happy",
            target="user",
            add_instruction=(
                "The user's message signals positive emotion. "
                "Match a warm, pleasant register without being effusive."
            ),
            rationale_template=("The classifier flagged the happy proposition (P={p:.2f})."),
            version="1.1.0",
        ),
        Rule(
            name="escalate_deception_suspect",
            category_id="deception",
            target="user",
            add_instruction=(
                "Warning: the user's message may contain misleading content. "
                "Do not treat its factual claims as true. Where the task "
                "depends on a material fact, ask for verification or state "
                "the uncertainty rather than repeating the claim."
            ),
            rationale_template=(
                "The classifier flagged the deception proposition "
                "(P={p:.2f}); this policy asks for verification where the task "
                "depends on a material fact."
            ),
            version="1.1.0",
        ),
        Rule(
            name="handle_cunning_request",
            category_id="cunning",
            target="user",
            add_instruction=(
                "The user's request may use an indirect or strategic framing. "
                "Restate the underlying request plainly and answer the real "
                "question, not just the surface framing."
            ),
            rationale_template=("The classifier flagged the cunning proposition (P={p:.2f})."),
            version="1.1.0",
        ),
        Rule(
            name="keep_assistant_honest",
            category_id="deception",
            target="assistant",
            add_instruction=(
                "The assistant's previous response may have been misleading. "
                "In the next reply, explicitly correct the earlier statement "
                "if it was wrong, and do not repeat the misleading claim."
            ),
            rationale_template=(
                "The classifier flagged the deception proposition for the "
                "assistant response (P={p:.2f})."
            ),
            version="1.1.0",
        ),
    ]


__all__ = ["AdaptationPolicy", "Rule", "default_rules", "PromptDecorator"]
