# Host integration note

**Status: design note, not a deployed integration.** This middleware has
not been run inside any live agent host, and no plugin wrapper is shipped.
Everything below uses only mechanisms that plugin-capable hosts typically
document, and nothing here claims otherwise.

## Why the host has to call the middleware

Plugin systems generally let you ship tools, agents, workspace views, MCP
servers and a shared key/value store (a "blackboard"). What they do *not*
typically guarantee is an automatic before-generation / after-generation
hook that sees every message. Plugin tools run when something invokes
them, and config/reference files are not automatically injected into model
context.

That matters here: the assignment requires that *every* complete user
message and *every* assistant response is classified. A plugin alone
cannot promise that. So the integration is host-side:

- **before generation** — classify the user message, adapt the system
  prompt, then generate;
- **after generation** — classify the assistant response, build feedback,
  store it, and hand it to the next generation.

The middleware does the classifying, the policy work and the bookkeeping.
The host just has to call it at those two moments.

## Pattern A — let the loop own one turn

Simplest case: the host hands the middleware the user text and a generator
callable, and reads the result.

```python
from jevloop import FeedbackLoop, build_classifier

loop = FeedbackLoop(
    classifier=build_classifier(),          # live if TYPESAFE_API_KEY is set, else mock
    base_system_prompt="You are a helpful support agent. ...",
)

result = loop.run_turn(
    user_text,
    generator=lambda effective_prompt, feedback: my_llm(
        system=effective_prompt.system_prompt,
        user=effective_prompt.user_message,
        context=feedback,                    # {"feedback": {...}} from the previous turn
    ),
)

# what to remember for the next turn:
result.next_turn_feedback    # {"source_message_id", "source_role", "flagged", "scores", ...}
result.response_classification
```

- The generator receives the exact `EffectivePrompt` the policy built —
  there is no second prompt-assembly path.
- `result.next_turn_feedback` is the compact, machine-readable feedback
  from the assistant response that just happened (flagged categories with
  rounded probabilities, source message id/role, model, request id).
- The loop also offers `run_turn_for_message()` if the host already
  builds `Message` objects, and a `revision_callback` if it wants the
  middleware to ask for a corrected reply when a response flags badly.

## Pattern B — the host owns generation

If the host already has its own generation pipeline and just wants the
signals, call the pieces directly:

```python
from jevloop import AdaptationPolicy, Message, build_classifier

classifier = build_classifier()
policy = AdaptationPolicy()
conversation: list[Message] = []
last_response_classification = None

# --- before generation ---
user_msg = Message(id="u1", role="user", text=user_text, turn=len(conversation) // 2)
user_classification = classifier.classify(user_msg)

classifications = [user_classification]
if last_response_classification is not None:
    classifications.append(last_response_classification)

adapted = policy.adapt(
    original_system_prompt=base_prompt,
    original_user_message=user_text,
    classifications=classifications,
    conversation=list(conversation),
)
reply = my_llm(system=adapted.effective_system_prompt, user=user_text)

# --- after generation ---
assistant_msg = Message(id="a1", role="assistant", text=reply, turn=user_msg.turn)
response_classification = classifier.classify(assistant_msg)
last_response_classification = response_classification
```

One honest caveat: the compact `next_turn_feedback` dict is assembled by
`FeedbackLoop.run_turn` (see `src/jevloop/loop.py`, `_next_turn_feedback_for`).
If you own generation, you can mirror that shape from
`response_classification.scores` (flag every score whose `.flagged` is
true), or just use Pattern A, which does it for you. The full
`FeedbackPayload` — classification + adapted prompt + revision decision —
is also only produced by Pattern A.

## What a host plugin could add (optional, unshipped)

If you want the loop visible from inside a plugin-capable host, the
mechanisms that fit are:

1. **Tools** — expose `classify_message` / `build_feedback` as plugin
   tools, most cleanly through a small MCP server (the core is Python).
2. **Blackboard** — append one record per turn
   (`state.append("turns", {...})`) so a workspace view can inspect
   classifications, prompts and feedback live.
3. **Workspace view** — render the transcript review from those records.

None of this is shipped or validated against a live host; it is listed so
the mapping from this repo to a real deployment is explicit.

## Operational notes

- **Budget:** one Jev call per message (all categories in one parallel
  batch). A two-sided turn therefore costs two calls, plus one per
  revision.
- **Live caps:** the evaluator refuses to exceed `--max-calls 50` in live
  mode, and the CLI defaults to offline unless asked otherwise.
- **Credentials:** `TYPESAFE_API_KEY` comes from the environment (see
  `.env.example`); there is no key in the repo.
- **No silent downgrade:** if live mode is requested and the SDK or key is
  missing, the code raises; it never quietly falls back to the mock.
- **Revision:** enabled by default, threshold `0.7` on deception/sad for
  the *assistant* message. A host can disable it or supply its own
  `revision_callback`.
- **Provenance:** policy id, policy version, model and request id travel
  with every recorded adaptation and feedback item, so a host can pin and
  audit behaviour later.
