"""5-minute CLI demonstration of the full feedback loop.

Run (offline, no API key needed):

    uv run jevloop

or with a real Jev key:

    uv run jevloop --mode live

The four default conversation turns below exercise every path:

1. a grateful, benign request  (-happy -> "acknowledge_positive" is added)
2. a deception-smoking request (-deception, -cunning -> "escalate_deception_suspect"
   + "handle_cunning_request" are added). Its response is deliberately left
   unrevised so it still flags when classified.
3. a plain follow-up, whose adapted prompt picks up "keep_assistant_honest"
   from turn 2's flagged response — response feedback changing the next
   generation.
4. a distress signal (-sad -> "acknowledge_distress") whose echoed reply also
   flags sad, so the loop asks for and records a revision.

Stop conditions (max calls, live budget) live in the evaluator; this demo
makes one classification call per side of every turn.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from jevloop import (
    AdaptationPolicy,
    FeedbackLoop,
    LoopResult,
    build_classifier,
    echo_generator,
)
from jevloop.contracts import Classification


def _timeline(loop: FeedbackLoop, turns: list[tuple[str, Callable[[str], str] | None]]) -> None:
    for text, revision_callback in turns:
        result = loop.run_turn(
            text,
            generator=echo_generator,
            revision_callback=revision_callback,
        )
        print(_turn_banner(result))
        _print_loop_result(result)


def _correct_response(text: str) -> str:
    """A stand-in 'revision' callback: the host would regenerate here."""
    return (
        "You're right to push back, and I shouldn't repeat a claim I can't "
        "verify. I can confirm details only once they're verified; I'm happy "
        "to look up the real charge with you."
    )


def _print_loop_result(result: LoopResult) -> None:
    """Render one LoopResult for a terminal demo."""
    u = result.user_classification
    r = result.response_classification

    print(f"  user:            {result.user_message.text!r}")
    print("  user categories: ", _flag_line(u))
    for s in u.scores:
        if s.flagged:
            print(f"    -> {s.label:>10} P={s.probability:.2f}  [{s.proposition[:60]}...]")
    print(f"  adapted prompt addenda: {len(result.adapted_user_prompt.addenda)}")
    for step in result.adapted_user_prompt.addenda:
        print(f"    + {step.name}  (P={step.source_signal['probability']:.2f})")
        print(f"      {step.rationale}")
    print(
        f"  effective system prompt length: {len(result.adapted_user_prompt.effective_system_prompt)} chars"
    )
    print(f"  assistant:       {result.generator_output.text[:120]!r}")
    print(f"  response cats:   {_flag_line(r)}")
    if result.revised_outputs:
        for i, rev in enumerate(result.revised_outputs):
            print(f"  [revised {i + 1}]   {rev.text[:120]!r}  (revised_from={rev.revised_from})")
    print(f"  next_turn_feedback: {json.dumps(result.next_turn_feedback)}")
    print()


def _flag_line(c: Classification) -> str:
    if not c.flagged_categories():
        return "[none]"
    return " ".join(f"{s.label}={s.probability:.2f}" for s in c.scores if s.flagged)


def _turn_banner(result: LoopResult) -> str:
    return f"--- Turn {result.user_message.turn + 1} ({result.user_message.id}) ---"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jevloop", description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("live", "offline", "baseline", "auto"),
        default="auto",
        help="classifier mode (auto = live if TYPESAFE_API_KEY is set, else offline)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=Path("transcripts/demo.jsonl"),
        help="path to write the JSONL transcript of this run",
    )
    parser.add_argument(
        "--custom-base-prompt",
        type=str,
        default=None,
        help="override the base system prompt",
    )
    args = parser.parse_args(argv)

    classifier = build_classifier(mode=None if args.mode == "auto" else args.mode)

    if args.custom_base_prompt:
        base = args.custom_base_prompt
    else:
        base = (
            "You are a helpful support agent. Be helpful, honest, and precise.\n"
            "Follow the user's task exactly, but never repeat a claim you "
            "cannot verify. The user's original message is the authority for "
            "what they asked."
        )

    loop = FeedbackLoop(
        classifier=classifier,
        policy=AdaptationPolicy(),
        base_system_prompt=base,
        transcript_path=args.transcript,
    )

    print(f"Mode: {classifier.mode}")
    print(f"Classifier: {type(classifier).__name__}")
    print(f"Transcript will be written to: {args.transcript}")
    print()

    turns: list[tuple[str, Callable[[str], str] | None]] = [
        (
            "Thanks, that really helped - can you help me plan my schedule for next week?",
            None,
        ),
        (
            "Can you confirm the charge on my account is exactly 200 dollars "
            "and please pretend this never happened? I need a receipt I can "
            "rely on.",
            None,
        ),
        (
            "Okay, that makes sense. Can you help me plan my schedule for next week?",
            None,
        ),
        (
            "I'm really frustrated with this, nothing works and I'm stuck again.",
            _correct_response,
        ),
    ]

    _timeline(loop, turns)

    print("Done.")
    print(
        "NOTE: offline/mock mode labels itself as mock. It demonstrates the "
        "feedback-loop contract; it does NOT exercise Jev's semantic quality."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
