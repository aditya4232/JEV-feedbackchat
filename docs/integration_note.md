# Bahulam integration proposal

**Status: proposal only.** The core middleware works in its CLI and tests;
no Bahulam plugin wrapper has been implemented or run in Bahulam.

## Verified extension points and the message boundary

The published [plugin overview](https://docs.bahulam.ai/plugins) describes
client-side callable tools, agents, workspace views, and plugin state. The
[plugin creation guide](https://docs.bahulam.ai/plugins/create) documents
`plugin.yaml`, ES-module tool handlers, agent configuration, and workspace
views. The [MCP guide](https://docs.bahulam.ai/plugins/mcp) documents Python
and other-language MCP tool servers. The
[shared-blackboard guide](https://docs.bahulam.ai/plugins/shared-blackboard)
documents persistent state through `options.state` and view-side access.

Those pages describe tools that run when called, agents that can be invoked,
and state shared by a handler and view. The reviewed pages do not document an
automatic before-generation/after-generation hook that observes every chat
message. A Bahulam host orchestrator must therefore invoke this middleware
at both boundaries if it needs the assignment's every-message behavior:

1. **Before generation:** pass the complete user message and bounded history
   to the loop. It classifies the user message, appends applicable named
   instructions to the trusted base system prompt, and gives that exact
   effective prompt plus the unchanged user request to the generator.
2. **After generation:** classify the complete assistant response and retain
   the structured feedback. On the next generation, include the prior
   assistant classification so the policy can adapt that prompt.

The plugin contract does not automatically intercept either boundary.
Calling a tool from an agent only classifies the message explicitly passed to
that tool.

## Core agent-facing contract

The host can delegate one full turn to the reusable Python loop:

```python
from jevloop import FeedbackLoop, build_classifier, build_generator

loop = FeedbackLoop(
    classifier=build_classifier(mode="live"),
    base_system_prompt="You are a helpful support agent. ...",
)
generator = build_generator(mode="openai-compatible")  # explicit app API config

result = loop.run_turn(
    user_text,
    generator=generator,
)

# The next loop turn consumes the response classification and this payload.
result.next_turn_feedback
result.response_classification
```

- The generator receives the exact `EffectivePrompt` the policy built; the
  integration test checks the prompt passed on the next generation.
- `OpenAICompatibleGenerator` sends the current request verbatim and includes
  only bounded recent history. Configure a real provider endpoint and API
  credentials explicitly; chat-product subscriptions are not assumed to
  provide application API access.
- The feedback records the source message id and role, per-category
  probabilities, decisions, classifier model and request id. Revision links
  are explicit, and revised assistant messages are classified too.
- The loop also offers `run_turn_for_message()` if the host already
  builds `Message` objects, and a `revision_callback` if it wants the
  middleware to ask for a corrected reply when a response flags badly.

## Pattern B — adapt an existing host generation path

If the host keeps its own model client, invoke the classifier and policy
before calling that client, then classify its complete response:

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

For a Bahulam plugin wrapper, a proposal consistent with the published
mechanisms is a Python MCP server exposing the middleware as a callable tool,
plus a JavaScript handler using `options.state.append("turns", record)` for
persistent review data and an optional workspace view. The manifest would
declare the MCP server and tool allowlist as documented. This wrapper would
make the capability callable and inspectable. It would still **not** create
automatic message interception; the host orchestrator must call it before
and after generation. No plugin manifest, handler, MCP server, or Bahulam
backend change is included here.

## Operational notes

- **Budget:** one Jev call per message (all categories in one parallel
  batch). A two-sided turn therefore costs two calls, plus one per
  revision.
- **Live caps:** the evaluator refuses to exceed `--max-calls 50` in live
  mode, and the CLI defaults to offline unless asked otherwise.
- **Credentials:** `TYPESAFE_API_KEY` comes from the environment (see
  `.env.example`); there is no key in the repo.
- **No silent downgrade:** if live mode is requested and the SDK or key is
  missing, classification raises; it never quietly falls back to the mock.
- **Revision:** the decision check is enabled by default at `0.7` and also
  requires the configured deception category decision to be flagged. No
  revision happens unless the host supplies a `revision_callback`. Assistant
  sadness or empathy does not automatically request revision.
- **Provenance:** each score records its boundary and provisional status;
  adaptations record policy version, signal, source message, model and
  request id.
