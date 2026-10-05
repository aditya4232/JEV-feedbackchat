"""The agent feedback loop.

``run_turn`` is one complete iteration of the loop:

    classify user -> build adapted prompt -> generate -> classify response
        -> build response feedback -> decide if revision is warranted

Every complete message (the user message AND the assistant response) is
classified. The structured feedback from the response affects the NEXT
generation. The integration test proves this by running two consecutive
turns and asserting that the effective prompt of the second turn was
modified by the response feedback of the first turn.

The generator is just a callable
``(EffectivePrompt) -> GeneratorOutput``. We ship a trivial
``EchoGenerator`` for testing and a ``RecordedGenerator`` for replaying
fixtures. Real agents plug in here without modifying the loop.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
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
            "generator_output": {"text": self.generator_output.text},
            "response_classification": self.response_classification.as_dict(),
            "response_feedback": self.response_feedback.as_dict(),
            "next_turn_feedback": self.next_turn_feedback,
            "revised_outputs": [{"text": r.text} for r in self.revised_outputs],
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
    revision_threshold: float = 0.7  # response P(any flagged cat) -> revise
    _turn: int = 0
    _conversation: list[Message] = field(default_factory=list)
    _last_response_feedback: dict[str, Any] = field(default_factory=dict)
    _last_response_classification: Classification | None = None

    def __post_init__(self) -> None:
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
        # 1. classify the user message
        user_classification = self.classifier.classify(user_msg)

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
            conversation=list(self._conversation),
        )

        # 3. assemble the full effective prompt the generator will see
        gen_input = EffectivePrompt(
            system_prompt=adapted.effective_system_prompt,
            user_message=user_msg.text,
            conversation=list(self._conversation),
        )

        # 4. ask the generator to produce the response
        response = generator(gen_input, {"feedback": dict(self._last_response_feedback)})
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
        response_classification = self.classifier.classify(assistant_msg)

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
            revised_output = GeneratorOutput(
                text=revised_text,
                revised_from=response.id,
                metadata={"triggered_by": "response_feedback"},
            )
            revised_msg = Message(
                id=f"a{user_msg.turn}-rev-{uuid.uuid4().hex[:6]}",
                role="assistant",
                text=revised_text,
                turn=user_msg.turn,
                revision_of=response.id,
            )
            revised_classifications.append(self.classifier.classify(revised_msg))
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
        self._conversation.append(user_msg)
        self._conversation.append(assistant_msg)
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
        flagged = {s.label: round(s.probability, 4) for s in classification.scores if s.flagged}
        return {
            "source_message_id": assistant_msg.id,
            "source_role": assistant_msg.role,
            "flagged": flagged,
            "scores": {s.label: round(s.probability, 4) for s in classification.scores},
            "model": classification.model,
            "request_id": classification.request_id,
        }

    def _should_revise_response(self, classification: Classification) -> tuple[bool, str | None]:
        """Decide whether the response itself should be revised.

        Conservative: only revise on clearly-flagged high-probability signals
        for deception or distress — i.e. the response is itself problematic.
        """
        by_cat = classification.scores_by_category()
        deception_p = by_cat.get("deception")
        sad_p = by_cat.get("sad")
        if (
            deception_p
            and not deception_p.abstained
            and deception_p.probability >= self.revision_threshold
        ):
            return True, (
                f"Response likely deceptive (P={deception_p.probability:.2f}); "
                "revising to correct the misleading claim."
            )
        if sad_p and not sad_p.abstained and sad_p.probability >= self.revision_threshold:
            return True, (
                f"Response expressed distress (P={sad_p.probability:.2f}); "
                "revising for a steadier register."
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
