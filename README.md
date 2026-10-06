# JEV feedback middleware

**Built by Aditya Shenvi.** This repository implements a small, reusable
middleware loop that classifies complete chat messages, turns those decisions
into explicit feedback, and adapts the effective instructions passed to an
agent generator.

## Interview requirements and implementation choices

The reported interview requirements are:

1. Use Jev to classify every user input and AI response.
2. Support extensible categories, including happy, sad, deception and cunning.
3. Send classifications back to the agent system so its effective prompt can
   be rephrased.
4. Submit the implementation through a Git repository.

The core path for these requirements is the live `LiveJevClassifier`, the
configurable category taxonomy, and `FeedbackLoop`. Live Jev use requires
`TYPESAFE_API_KEY`; without it, the CLI must be run in explicit offline mode,
which is a deterministic mock and does not verify Jev semantics.

The Python library, CLI, small single-annotator evaluation set, and optional
provider adapters are implementation choices. FastAPI, SQLite, a review UI,
and a native Bahulam plugin wrapper are not interview requirements and are
not part of this core implementation. No deployment is required.

## Five-minute local demonstration

On Windows PowerShell, run:

```powershell
./examples/five_minute_demo.ps1
```

The script installs locked dependencies, runs a four-turn conversation with
the explicitly labelled offline classifier and mock generator, runs tests,
and prints separate mock and keyword-baseline evaluations. The demo includes
an assistant response whose classification adds `keep_assistant_honest` to
the next turn's effective system prompt. The generator reports that it
received the adapted prompt. These fixed fixtures demonstrate orchestration,
not semantic classification quality.

On a Unix shell, the equivalent is `bash examples/five_minute_demo.sh`.
For individual commands:

```powershell
uv sync --frozen --extra live
uv run jevloop --mode offline --generator-mode mock
uv run pytest -q                 # excludes live-marked tests, even with a key set
uv run ruff check src evaluation tests
uv run mypy src evaluation
uv run jevloop-eval --mode offline
uv run jevloop-eval --mode baseline
```

The CLI writes its transcript to ignored `transcripts/run.jsonl` by default.
The checked-in `transcripts/demo.jsonl` is a sample from the offline/mock
demo, not a Jev run. The active classifier and generator modes are printed;
`offline` is never described as Jev.

## How the loop works

```text
complete user message + bounded history
  -> Jev classifies each configured proposition independently
  -> structured score / decision / abstention
  -> named, versioned deterministic prompt policy
  -> generator receives exact effective system prompt + original request
  -> complete assistant response is classified
  -> assistant classification changes the next generation's prompt
```

`FeedbackLoop.run_turn` owns one turn. `tests/test_loop.py::
test_response_feedback_updates_next_prompt` uses a scripted assistant
classification and a recording generator to assert that the next call
receives the exact `keep_assistant_honest` effective prompt and the matching
feedback payload. This verifies orchestration independently of Jev's
semantic quality.

Prompt adaptation is a controlled append-only transformation. It preserves
the original user message and trusted base instructions, records the policy
name/version, changed instruction, source message, classifier model and
request id, and passes that effective prompt to the generator. Jev does not
write or rewrite prompts. The real generator adapter keeps the current user
request verbatim and sends only bounded prior context.

Message IDs, roles, turn numbers, and revision links are recorded. Every
complete user and assistant message is classified; if a host requests a
revision through a callback, the revised assistant response is classified as
well. Streaming fragments are not individually classified.

## Source map

| Requirement | Code |
|---|---|
| Categories, independent scores, provisional boundaries | `src/jevloop/taxonomy.py`, `src/jevloop/contracts.py` |
| Real Jev request and validated response | `src/jevloop/adapter.py:LiveJevClassifier` |
| Explicit deterministic mock | `src/jevloop/adapter.py:OfflineMockClassifier` |
| Prompt changes and rationale | `src/jevloop/prompt_policy.py:AdaptationPolicy` |
| User/assistant classification and next-turn feedback | `src/jevloop/loop.py:FeedbackLoop` |
| Configurable real and mock generators | `src/jevloop/generator.py` |
| Independent labels and per-category evaluation | `evaluation/cases.jsonl`, `evaluation/run_eval.py` |
| Orchestration and adapter contracts | `tests/test_loop.py`, `tests/test_adapter.py`, `tests/test_generator.py` |

The classifier creates one Noul question per category. Each score is the
probability of that proposition; scores are independent and are not
normalized to sum to one. `DecisionPolicy` applies explicit category
boundaries. Initial boundaries are provisional, and each returned score
retains its decision, abstention, boundary, and provisional status.

Emotion (`happy`, `sad`) is separate from behavioral signals (`deception`,
`cunning`). Frustration alone does not imply sadness. An assistant expressing
sympathy does not automatically need revision. Ambiguous "pretend" wording,
quotations, negation, hypothetical requests, and trait discussion require
context; direct lies or ordinary strategy do not automatically imply
cunning. The policy records classifier scores and decisions; it does not
invent Jev evidence or explanations.

## Live Jev and generator configuration

Set `TYPESAFE_API_KEY` in the process environment, then smoke-test Jev before
spending budget on evaluation:

```powershell
uv run pytest -m live -x
uv run jevloop --mode live --generator-mode mock
uv run jevloop-eval --mode live --max-calls 49
```

The live smoke test sends a small ordinary/adversarial set and validates
response parsing and probability ranges. Evaluation is separately capped at
50 message calls. Live results, when produced, are written separately from
offline and baseline reports. The present verification status is reported at
the end of this README.

The default generator is the labelled mock echo. A real OpenAI-compatible
chat-completions generator requires application API configuration:

```powershell
$env:JEVLOOP_GENERATOR_MODE = "openai-compatible"
$env:JEVLOOP_GENERATOR_BASE_URL = "https://provider.example/v1"
$env:JEVLOOP_GENERATOR_MODEL = "your-api-model"
$env:JEVLOOP_GENERATOR_API_KEY = "your-application-api-key"
uv run jevloop --mode offline --generator-mode openai-compatible
```

Use a provider endpoint and API credential explicitly issued for application
requests. The adapter does not assume that an OpenCode or Kimi chat
subscription includes application API access. Jev mode and generator mode
are independent.

## Evaluation

The current set has 49 manually labelled English examples and one annotator.
Its expected labels are independent of the classifier output. It covers both
roles, sadness versus frustration, corrections, quotation, negation,
ambiguity, trait discussion, and benign versus manipulative strategy.
`evaluation/schema.md` records the annotation rules and known limits.

The `offline` classifier is a deterministic fixed-phrase fixture mock. Its
metrics validate evaluation plumbing only. `baseline` is a separate literal
keyword heuristic, not a regex model and not the main classifier. It has no
true positives on this set: the only keyword score above the default 0.6
boundary is `sad` at 0.65; none of the positive sadness examples contains the
literal token `sad`. The explicit negation example `neg-not-sad` is its one
false positive. Report precision/recall/F1 by category with confusion counts;
F1 is not accuracy. Do not interpret mock or baseline results as Jev quality.
In the offline report, the only two true positives (`happy-helped` and
`dec-pretend`) are exact fixture overlaps. They are not evidence of semantic
classification quality; evaluation text was not changed to improve mock
scores.

## Proposed Bahulam use

Bahulam documents plugin tools, agents, workspace views, and persistent
plugin state in its [plugin overview](https://docs.bahulam.ai/plugins),
[creation guide](https://docs.bahulam.ai/plugins/create), and
[shared-blackboard guide](https://docs.bahulam.ai/plugins/shared-blackboard).
The [MCP guide](https://docs.bahulam.ai/plugins/mcp) documents Python and
other-language tool servers declared by a plugin.
The reviewed pages do not document an automatic before/after-generation hook
that sees every message. A host orchestrator would need to call this
middleware before generation and after each complete assistant response.
A Python MCP server could expose the core as a callable tool; the shared
blackboard and workspace view could optionally store and inspect records.
That server and its plugin manifest would still need to be built and tested.
These mechanisms do not create automatic message interception. No Bahulam
wrapper is shipped or verified in a live host. See
[`docs/integration_note.md`](docs/integration_note.md).

## Verification status

Verified locally on 2026-10-06: `uv sync --frozen --extra live` checked 29
locked packages; the PowerShell demonstration exited 0; default tests report
69 passed and 1 live test deselected; the explicit live-test command selects
that test and skips it because `TYPESAFE_API_KEY` is absent. Ruff lint and
format checks pass, and mypy succeeds on 10 source files. The 49-case offline
mock evaluation and keyword baseline both ran and are reported separately.

Live Jev verification is pending because no `TYPESAFE_API_KEY` is configured.
No application generator endpoint, model, or key is configured, so the real
generator has not been exercised; its request/response contract is covered by
fake-transport tests. Offline and keyword-baseline results are not Jev quality
measurements.
