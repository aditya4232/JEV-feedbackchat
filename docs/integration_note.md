# Agent-facing contract and host integration proposal

The core is a Python library; no native plugin wrapper has been verified.

```python
from jevloop import AdaptationPolicy, FeedbackLoop, build_classifier, build_generator

loop = FeedbackLoop(
    classifier=build_classifier(mode="live"),
    policy=AdaptationPolicy(),
    base_system_prompt="You are a helpful, honest assistant.",
)
generator = build_generator(mode="openai-compatible")  # separately configured API
first = loop.run_turn("Help me plan tomorrow", generator=generator)
second = loop.run_turn("Continue that plan", generator=generator)
```

Keep one loop instance per conversation. It classifies each complete user
message with bounded prior context, adapts the prompt and invokes the generator.
It then classifies the complete assistant response and consumes that feedback
on the next call. The library is synchronous; a host must serialize turns for
one conversation. State is in memory unless the host persists/restores it.

Generator signature: `(EffectivePrompt, dict) -> GeneratorOutput`.
The prompt contains exact effective system instructions, the unchanged user
request, and bounded role-tagged history. The dictionary contains current
`user_classification`, previous assistant `feedback`, and `applied_policies`.
`LoopResult.to_dict()` returns IDs, roles, turn order, all classifications,
policy provenance, actual generator input, feedback and revision relationships.

`run_turn_for_message()` accepts a host-owned `Message` ID. An optional
`revision_callback(text) -> text` can correct a flagged deceptive response;
at most one revision runs per turn. Its full response is also classified.
The final response classification determines the following turn's feedback.
No paid classification occurs for individual streaming fragments.

## Proposed extension boundary

Reviewed extension documentation describes callable tools, agents with tool
allowlists, workspace views and persistent state (`options.state`). A thin
wrapper could expose this library through an MCP tool server and use a handler
and view to retain/inspect records. That proposal has no implemented manifest,
handler or host runtime verification.

Those extension mechanisms do not establish automatic message interception.
The host orchestrator must explicitly invoke the core at the before-generation
and after-generation boundaries, or delegate the complete turn to `FeedbackLoop`.
Calling a tool only classifies the message passed to it. No private backend
or undocumented endpoint is assumed. Company-specific names and manifest
examples are intentionally excluded from this submission.
