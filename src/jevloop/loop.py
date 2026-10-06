"""The agent feedback loop.

``run_turn`` is one complete iteration of the loop:

    classify user -> build adapted prompt -> generate -> classify response
        -> build response feedback -> decide if revision is warranted

Every complete message (the user message AND the assistant response) is
classified. The structured feedback from the response affects the NEXT
generation. The integration test proves this by running two consecutive
turns and asserting that the effective prompt of the second turn was
modified by the response feedback of the first turn.

The generator is a callable
``(EffectivePrompt, feedback) -> GeneratorOutput``. The labelled echo mock
supports offline runs; ``OpenAICompatibleGenerator`` is the configurable
real adapter for chat-completions endpoints.
"""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .adapter import JevClassifier
from .contracts import (
    AdaptedPrompt,
    Classification,
    EffectivePrompt,
    FeedbackPayload,
    GeneratorOutput,
    Message,
)
from .prompt_policy import AdaptationPolicy

# ---------------------------------------------------------------------------
# Generator interface
# ---------------------------------------------------------------------------


Generator = Callable[[EffectivePrompt, dict[str, Any]], GeneratorOutput]


@dataclass
class LoopResult:
    """One turn of the loop: the user message and its full bookkeeping."""

    user_message: Message
    user_classification: Classification
    adapted_user_prompt: AdaptedPrompt
    generator_input: EffectivePrompt
    generator_output: GeneratorOutput
    response_classification: Classification
    assistant_messages: list[Message]
    response_feedback: FeedbackPayload
    next_turn_feedback: dict[str, Any]
    revised_outputs: list[GeneratorOutput] = field(default_factory=list)
    revised_response_classifications: list[Classification] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_message": {
                "id": self.user_message.id,
                "role": self.user_message.role,
                "turn": self.user_message.turn,
                "text": self.user_message.text,
            },
            "user_classification": self.user_classification.as_dict(),
            "adapted_user_prompt": self.adapted_user_prompt.summary(),
            "generator_input": self.generator_input.to_dict(),
            "generator_output": {
                "id": self.generator_output.id,
                "text": self.generator_output.text,
                "revised_from": self.generator_output.revised_from,
                "metadata": self.generator_output.metadata,
            },
            "assistant_messages": [
                {
                    "id": message.id,
                    "role": message.role,
                    "turn": message.turn,
                    "text": message.text,
                    "revision_of": message.revision_of,
                    "metadata": message.metadata,
                }
                for message in self.assistant_messages
            ],
            "response_classification": self.response_classification.as_dict(),
            "response_feedback": self.response_feedback.as_dict(),
            "next_turn_feedback": self.next_turn_feedback,
            "revised_outputs": [
                {
                    "id": output.id,
                    "text": output.text,
                    "revised_from": output.revised_from,
                    "metadata": output.metadata,
                }
                for output in self.revised_outputs
            ],
            "revised_response_classifications": [
                classification.as_dict() for classification in self.revised_response_classifications
            ],
        }


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


@dataclass
class FeedbackLoop:
    """The middleware. Holds the classifier, the policy, and the transcript.

    The transcript is recorded in memory and (optionally) persisted to a
    JSONL file. Reviewers can replay it to verify that the prompt the
    generator actually saw matches the prompt the policy claims it built.
    """

    classifier: JevClassifier
    policy: AdaptationPolicy
    base_system_prompt: str
    transcript_path: Path | None = None
    revision_enabled: bool = True
    revision_threshold: float = 0.7  # assistant deception probability -> revise
    max_context_messages: int = 20
    max_context_chars: int = 12_000
    _turn: int = 0
    _conversation: list[Message] = field(default_factory=list)
    _last_response_feedback: dict[str, Any] = field(default_factory=dict)
    _last_response_classification: Classification | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_context_messages, bool)
            or not isinstance(self.max_context_messages, int)
            or isinstance(self.max_context_chars, bool)
            or not isinstance(self.max_context_chars, int)
            or self.max_context_messages < 0
            or self.max_context_chars < 0
        ):
            raise ValueError("conversation context limits must be non-negative")
        if (
            isinstance(self.revision_threshold, bool)
            or not math.isfinite(self.revision_threshold)
            or not 0.0 <= self.revision_threshold <= 1.0
        ):
            raise ValueError("revision_threshold must be finite and in [0,1]")
        if self.transcript_path is not None:
            self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
            self.transcript_path.write_text("", encoding="utf-8")

    # -- public API -------------------------------------------------------

    def run_turn(
        self,
        user_text: str,
        *,
        generator: Generator,
        revision_callback: Callable[[str], str] | None = None,
    ) -> LoopResult:
        """Run one full turn: user in, response out, response classified."""
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")
        self._turn += 1
        turn = self._turn

        user_msg = Message(
            id=f"u{turn}-{uuid.uuid4().hex[:6]}",
            role="user",
            text=user_text,
            turn=turn - 1,
        )
        return self._run_turn_for(
            user_msg, generator=generator, revision_callback=revision_callback
        )

    def run_turn_for_message(
        self,
        user_message: Message,
        *,
        generator: Generator,
        revision_callback: Callable[[str], str] | None = None,
    ) -> LoopResult:
        """Run a turn with a pre-built :class:`Message` (useful for tests)."""
        if user_message.role != "user":
            raise ValueError("run_turn_for_message requires a user message")
        if user_message.turn < self._turn:
            raise ValueError(
                f"user message turn {user_message.turn} is earlier than the next turn {self._turn}"
            )
        self._turn = max(self._turn, user_message.turn + 1)
        return self._run_turn_for(
            user_message, generator=generator, revision_callback=revision_callback
        )

    # -- internals --------------------------------------------------------

    def _run_turn_for(
        self,
        user_msg: Message,
        *,
        generator: Generator,
        revision_callback: Callable[[str], str] | None = None,
    ) -> LoopResult:
        history = self._bounded_context(self._conversation)

        # 1. classify the current user message with bounded prior context.
        user_classification = self.classifier.classify(self._with_context(user_msg, history))

        # 2. adapt the effective prompt for THIS generation. The current
        #    user-message classification drives the primary adaptations; the
        #    classification of the PREVIOUS assistant response is applied
        #    after it, so response feedback can change the next prompt
        #    (assistant-target rules such as keep_assistant_honest).
        classifications: list[Classification] = [user_classification]
        if self._last_response_classification is not None:
            classifications.append(self._last_response_classification)
        adapted = self.policy.adapt(
            original_system_prompt=self.base_system_prompt,
            original_user_message=user_msg.text,
            classifications=classifications,
            conversation=history,
        )

        # 3. assemble the full effective prompt the generator will see
        gen_input = EffectivePrompt(
            system_prompt=adapted.effective_system_prompt,
            user_message=user_msg.text,
            conversation=history,
        )

        # 4. ask the generator to produce the response
        response = generator(
            gen_input,
            {
                "user_classification": user_classification.as_dict(),
                "feedback": dict(self._last_response_feedback),
                "applied_policies": adapted.summary(),
            },
        )
        response.id = response.id or f"a{user_msg.turn}-{uuid.uuid4().hex[:6]}"

        # 5. classify the response — this is the LOOP part that completes
        #    the feedback cycle
        assistant_msg = Message(
            id=response.id,
            role="assistant",
            text=response.text,
            turn=user_msg.turn,
            metadata=response.metadata,
        )
        assistant_messages = [assistant_msg]
        assistant_context = self._bounded_context([*history, user_msg])
        response_classification = self.classifier.classify(
            self._with_context(assistant_msg, assistant_context)
        )

        # 6. build response feedback for the NEXT turn
        response_feedback = self._build_response_feedback(
            assistant_msg=assistant_msg,
            classification=response_classification,
            adapted_prompt=adapted,
        )

        # 7. decide on optional revision of THIS response
        revised: list[GeneratorOutput] = []
        revised_classifications: list[Classification] = []
        if (
            self.revision_enabled
            and revision_callback is not None
            and response_feedback.should_revise
        ):
            revised_text = revision_callback(response.text)
            revised_id = f"a{user_msg.turn}-rev-{uuid.uuid4().hex[:6]}"
            revised_output = GeneratorOutput(
                text=revised_text,
                id=revised_id,
                revised_from=response.id,
                metadata={"triggered_by": "response_feedback"},
            )
            revised_msg = Message(
                id=revised_id,
                role="assistant",
                text=revised_text,
                turn=user_msg.turn,
                revision_of=response.id,
            )
            assistant_messages.append(revised_msg)
            revised_context = self._bounded_context([*history, user_msg, assistant_msg])
            revised_classifications.append(
                self.classifier.classify(self._with_context(revised_msg, revised_context))
            )
            revised.append(revised_output)
            # The revised response becomes the official transcript entry
            assistant_msg = revised_msg
            response = revised_output

        # 8. update transcript state for the NEXT turn. The final transcript
        #    entry is what carries into the next generation, so the feedback
        #    for the next turn is computed from the revised message whenever
        #    a revision happened.
        final_classification = (
            revised_classifications[-1] if revised_classifications else response_classification
        )
        self._conversation = self._bounded_context([*history, user_msg, assistant_msg])
        self._last_response_classification = final_classification
        self._last_response_feedback = self._next_turn_feedback_for(
            assistant_msg, final_classification
        )

        # 9. persist this turn
        result = LoopResult(
            user_message=user_msg,
            user_classification=user_classification,
            adapted_user_prompt=adapted,
            generator_input=gen_input,
            generator_output=response,
            response_classification=response_classification,
            assistant_messages=assistant_messages,
            response_feedback=response_feedback,
            next_turn_feedback=self._last_response_feedback,
            revised_outputs=revised,
            revised_response_classifications=revised_classifications,
        )
        if self.transcript_path is not None:
            with self.transcript_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False))
                f.write("\n")
        return result

    def _bounded_context(self, messages: list[Message]) -> list[Message]:
        """Keep the most recent history within both count and character limits."""
        if not self.max_context_messages or not self.max_context_chars:
            return []
        selected: list[Message] = []
        remaining = self.max_context_chars
        for message in reversed(messages[-self.max_context_messages :]):
            if remaining <= 0:
                break
            text = message.text[-remaining:]
            selected.append(replace(message, text=text))
            remaining -= len(text)
        selected.reverse()
        return selected

    @staticmethod
    def _with_context(message: Message, context: list[Message]) -> Message:
        metadata = dict(message.metadata)
        metadata["conversation_context"] = [
            {"role": item.role, "text": item.text} for item in context
        ]
        return replace(message, metadata=metadata)

    # -- feedback construction -------------------------------------------

    def _build_response_feedback(
        self,
        *,
        assistant_msg: Message,
        classification: Classification,
        adapted_prompt: AdaptedPrompt,
    ) -> FeedbackPayload:
        """Build the structured feedback that wraps the assistant message.

        The next turn consumes ``next_turn_feedback`` (a small dict that the
        generator can use as additional context). We keep the policy-driven
        adapted_prompt here too so a host orchestrator can replay the loop
        from a single FeedbackPayload.
        """
        should_revise, rationale = self._should_revise_response(classification)

        return FeedbackPayload(
            classification=classification,
            adapted_prompt=adapted_prompt,
            next_turn_feedback=self._next_turn_feedback_for(assistant_msg, classification),
            should_revise=should_revise,
            revision_rationale=rationale,
        )

    def _next_turn_feedback_for(
        self,
        assistant_msg: Message,
        classification: Classification,
    ) -> dict[str, Any]:
        """The compact, machine-readable feedback the next generation sees.

        This is the *consumed* signal — the loop's ``generator`` callable
        receives it as part of its second argument. The integration test
        asserts that the next prompt changed because of this signal.
        """
        flagged = {
            s.category_id: round(s.probability, 4)
            for s in classification.scores
            if s.flagged and not s.abstained
        }
        return {
            "source_message_id": assistant_msg.id,
            "source_role": assistant_msg.role,
            "flagged": flagged,
            "scores": {s.category_id: round(s.probability, 4) for s in classification.scores},
            "model": classification.model,
            "request_id": classification.request_id,
        }

    def _should_revise_response(self, classification: Classification) -> tuple[bool, str | None]:
        """Decide whether the response itself should be revised.

        Conservative: only a high deception score can request revision. Sadness
        or an empathetic acknowledgement is not a reason to revise a response.
        The classifier supplies a score, not evidence about which claim is false.
        """
        by_cat = classification.scores_by_category()
        deception_p = by_cat.get("deception")
        if (
            deception_p
            and deception_p.flagged
            and not deception_p.abstained
            and deception_p.probability >= self.revision_threshold
        ):
            return True, (
                f"Classifier deception score P={deception_p.probability:.2f} "
                f"meets the revision threshold {self.revision_threshold:.2f}; "
                "request a fact-grounded review without assuming which claim is false."
            )
        return False, None


# ---------------------------------------------------------------------------
# Default generators
# ---------------------------------------------------------------------------


def echo_generator(ep: EffectivePrompt, _feedback: dict[str, Any]) -> GeneratorOutput:
    """Trivial generator for testing.

    Returns a one-line acknowledgement plus a copy of the user message so the
    transcript is readable. Not a real model.
    """
    return GeneratorOutput(
        text=(f"Acknowledged: {ep.user_message[:200]}" if ep.user_message else "(empty)"),
        metadata={"generator": "echo"},
    )


def recorded_generator(
    fixture: Iterable[str],
) -> Generator:
    """Generator that returns successive lines from a fixture list.

    Cycles when exhausted. Useful for deterministic tests.
    """
    iterator = iter(fixture)

    def _gen(_ep: EffectivePrompt, _fb: dict[str, Any]) -> GeneratorOutput:
        try:
            text = next(iterator)
        except StopIteration:
            text = "(fixture exhausted)"
        return GeneratorOutput(text=text, metadata={"generator": "recorded"})

    return _gen


# Public re-export of run_turn as a convenience function on top of a loop.
def run_turn(
    loop: FeedbackLoop,
    user_text: str,
    *,
    generator: Generator = echo_generator,
) -> LoopResult:
    """One full turn through the loop. Provided as a module-level shortcut."""
    return loop.run_turn(user_text, generator=generator)


__all__ = [
    "FeedbackLoop",
    "LoopResult",
    "Generator",
    "echo_generator",
    "recorded_generator",
    "run_turn",
]
