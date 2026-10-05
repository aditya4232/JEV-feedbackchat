# jev-feedbackchat

**Built by Aditya Shenvi.**

A small middleware layer for an agent chat loop. Every user message *and*
every assistant response gets classified with [Jev](https://typesafe.ai),
the classifications turn into structured feedback, and a deterministic
policy turns that feedback into concrete additions to the *next* system
prompt. Nothing is rewritten by a model, every change is named and
versioned, and the exact prompt the generator saw is recorded.

This repo is my submission for the assignment: classify every user input
and AI response with Jev, keep the categories extensible (happy, sad,
deception, cunning, …), feed the classification back into the agent system
so prompts are rephrased, and submit through a Git repository. No
deployment is involved.

**Jump to:** [Quick start](#quick-start-5-minutes) ·
[How the loop works](#how-the-loop-works) ·
[Repo map](#repo-map) ·
[Evaluation](#e-evaluation) ·
[Host integration](#f-using-this-with-a-plugin-based-agent-host) ·
[Status](#g-reproducibility-and-status)

---

## Quick start (5 minutes)

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/). No API key is
required for any of this — the live SDK is installed, but the demo uses
the labelled offline mock unless you ask for `--mode live`.

```bash
uv sync --extra live      # core has no runtime deps; dev tools + Jev SDK
uv run jevloop            # four-turn demo conversation, offline mock
uv run pytest -q          # 45 passed, 1 skipped (live smoke test, no key)
```

Or let the script do all of it, including both evaluation modes:

```bash
bash examples/five_minute_demo.sh
```

The demo runs a four-turn support conversation and writes
`transcripts/demo.jsonl`, so you can replay exactly what happened. Between
them, the turns exercise every default rule and both feedback directions:

1. a grateful request → `acknowledge_positive` added;
2. a request that tries to get the assistant to lie →
   `escalate_deception_suspect` **and** `handle_cunning_request` added; the
   echoed reply itself flags deception and is deliberately left unrevised, so
   it stays flagged;
3. a plain follow-up → the prompt picks up `keep_assistant_honest` from
   turn 2's *response* classification (response feedback changing the next
   generation);
4. a distress signal → `acknowledge_distress` added, the flagged reply is
   classified, and a revision is requested and recorded.

### Live mode

```bash
cp .env.example .env             # then put TYPESAFE_API_KEY in it
set -a; source .env; set +a      # or: export TYPESAFE_API_KEY=...
uv run jevloop --mode live       # real Jev classification
```

Then, within a bounded budget:

```bash
uv run pytest -m live -x                              # small live smoke test
uv run jevloop-eval --mode live --max-calls 42        # scored live evaluation
```

---

## How the loop works

```
                        ┌──────────────────────────────────────────┐
                        │            YOUR AGENT HOST               │
                        │    (any chat loop / orchestrator)        │
                        └──────────────────────────────────────────┘
                                        ▲                 │
        user message in                 │                 ▼
   ─────────────────────────────────────┤         assistant out
                                        │                 │
                        ┌───────────────┴─────────────────▼───────┐
                        │          jev-feedbackchat core          │
                        │   classify(user) -> adapt(prompt)       │
                        │   classify(assistant) -> response       │
                        │   feedback -> next generation           │
                        └─────────────────────────────────────────┘
```

`FeedbackLoop.run_turn(user_text, generator)` does eight things:

1. classify the user message;
2. adapt the system prompt using the user-message signals;
3. build the exact `EffectivePrompt` the generator will receive;
4. call the generator;
5. classify the assistant response;
6. build the response feedback (`FeedbackPayload`);
7. optionally request a revision if the response itself flagged badly;
8. persist the turn and hand `next_turn_feedback` to the **next**
   generation.

Because step 8 feeds step 2 of the following turn, an assistant response
genuinely changes a later prompt — that's what
`tests/test_loop.py::test_response_feedback_updates_next_prompt` asserts.
It runs two turns and checks that the second turn's effective prompt
contains the addendum triggered by the first response's classification.

---

## Repo map

```
src/jevloop/
├── taxonomy.py       A. category catalog + DecisionPolicy (thresholds)
├── contracts.py      A. every dataclass that crosses a boundary:
│                        Message, Classification, CategoryScore,
│                        AdaptedPrompt, AdaptationStep, FeedbackPayload,
│                        EffectivePrompt, GeneratorOutput
├── adapter.py        B. JevClassifier interface, LiveJevClassifier
│                        (typesafe-sdk), OfflineMockClassifier (labelled),
│                        RulesBaselineClassifier (labelled, comparison only)
├── prompt_policy.py  C. deterministic, named, versioned adaptation rules
├── loop.py           D. FeedbackLoop + run_turn + generators
└── cli.py            the four-turn demo (`uv run jevloop`)

evaluation/
├── schema.md         annotation rules for the eval set
├── cases.jsonl       42 hand-annotated, deliberately varied cases
└── run_eval.py       bounded evaluator (offline / baseline / live)

tests/                unit + integration tests (orchestration, not accuracy)
docs/integration_note.md   how a plugin-based host calls the middleware
examples/five_minute_demo.sh   end-to-end offline walkthrough
transcripts/demo.jsonl         replayable transcript of a real demo run
```

The letter tags A–G match the deliverables in the assignment brief.

---

## A. The taxonomy

Four default categories, each tagged with its *kind*:

| id | kind | what it actually asks |
|---|---|---|
| `happy` | emotion | does the message express positive emotion? |
| `sad` | emotion | does the message express negative emotion? |
| `deception` | behaviour | is the message *itself* trying to mislead — not merely talking about deception? |
| `cunning` | behaviour | does the message use a strategic/indirect manipulative framing (not every strategic request)? |

Two design rules I kept from the brief:

- **Dimensions stay separate.** Task intent, expressed emotion and
  behavioural signals never collapse into one label; a message can flag
  several categories at once.
- **Extensible by construction.** Categories live in `CATEGORY_CATALOG`,
  and the adapter builds one Jev question per category, so adding one is a
  small, local change.

Each Jev question is the probability of *its own* proposition. The values
are independent, they don't sum to 1, and they are **never normalised**.
`DecisionPolicy` applies the configurable boundaries (defaults are 0.6 and
provisional, versioned, overridable).

## B. The Jev adapter

- `LiveJevClassifier` — the real integration via `typesafe-sdk`, one
  parallel question batch per message. Needs `TYPESAFE_API_KEY`.
- `OfflineMockClassifier` — no network, deterministic fixture scores,
  labelled as mock in every output. It proves contracts and orchestration,
  not semantic quality.
- `RulesBaselineClassifier` — a keyword heuristic that exists only as a
  comparison; labelled with its limitations.
- `build_classifier()` — never silently downgrades live → offline. If you
  ask for live and the key/SDK is missing, it raises; the mode is explicit
  in every transcript entry.

## C. Prompt adaptation

Adaptation is a controlled transformation of the **effective system
prompt**, not a rewrite:

- the user's message is preserved verbatim;
- the task, constraints and trusted base instructions are preserved;
- named, versioned **addenda** are appended, each recording what changed,
  which Jev signal triggered it, and why;
- the resulting `effective_system_prompt` is exactly what the generator
  receives.

Each `AdaptationStep` looks like this:

```json
{
  "name": "acknowledge_distress",
  "instruction_added": "The user's message signals distress…",
  "rationale": "User message expressed negative emotion (sad category, P=0.88).",
  "source_signal": {"category": "sad", "probability": 0.88, "role": "user"}
}
```

The rationale is written *by the policy code* from Jev's number — I never
fabricate an explanation or a quotation that Jev didn't produce. The same
classification always produces the same prompt.

## D. The feedback loop

The public surface is intentionally tiny:

```python
from jevloop import FeedbackLoop, build_classifier

loop = FeedbackLoop(
    classifier=build_classifier(),
    base_system_prompt="You are a helpful support agent. ...",
)
result = loop.run_turn(user_text, generator=my_generator)
result.next_turn_feedback      # compact dict for the next generation
```

Every complete user message and every assistant message is classified —
including a revised response, if revision is enabled. Message ids, roles,
turn order and revision relationships are explicit (`Message.id`, `.role`,
`.turn`, `.revision_of`).

Revision is conservative and opt-in at the call site: a response is only
re-requested when its own deception/sad probability crosses 0.7 *and* the
host supplied a `revision_callback`.

## E. Evaluation

`evaluation/schema.md` defines the annotation rules; the expected labels
were hand-written with those rules and were **not** produced by the
classifier being evaluated. `evaluation/cases.jsonl` has 42 varied cases:
both roles, contextual corrections, negation, sarcasm, quotations,
ambiguity, and benign-versus-manipulative strategy.

```bash
uv run jevloop-eval                              # offline harness check (mock)
uv run jevloop-eval --mode baseline              # labelled regex baseline
uv run jevloop-eval --mode live --max-calls 42   # bounded live evaluation
```

Reports land in `evaluation/results/` (gitignored — they're regenerable).
Captured runs from this machine:

**Offline mock** — this checks that the *harness* is wired correctly. A
few cases intentionally overlap the mock's fixture table, so non-zero true
positives mean predictions reached the right labels. It says nothing about
Jev's semantic quality.

| category | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| happy | 1 | 0 | 7 | 34 | 1.000 | 0.125 | 0.222 |
| sad | 1 | 0 | 6 | 35 | 1.000 | 0.143 | 0.250 |
| deception | 2 | 0 | 10 | 30 | 1.000 | 0.167 | 0.286 |
| cunning | 2 | 0 | 10 | 30 | 1.000 | 0.167 | 0.286 |
| macro average | — | — | — | — | 1.000 | 0.150 | 0.261 |

**Rules baseline** — deliberately the weakest thing I could write
(keyword/regex, max probability 0.55). It never crosses the 0.6 boundary,
so it predicts no positives at all: a reminder that a trivial classifier
can't carry this problem.

| category | TP | FP | FN | TN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| happy | 0 | 0 | 8 | 34 | — | 0.000 | — |
| sad | 0 | 0 | 7 | 35 | — | 0.000 | — |
| deception | 0 | 0 | 12 | 30 | — | 0.000 | — |
| cunning | 0 | 0 | 12 | 30 | — | 0.000 | — |
| macro average | — | — | — | — | — | 0.000 | — |

**Live Jev** — pending, because I had no `TYPESAFE_API_KEY` in this
environment. I'm not claiming any live precision/recall numbers.

Orchestration is tested *separately* from accuracy: `tests/test_loop.py`
covers the loop mechanics, while `evaluation/run_eval.py` scores semantics.

## F. Using this with a plugin-based agent host

A plugin-based host can't guarantee that every message gets classified —
plugin tools run when something invokes them, and there is no documented
automatic before/after-generation hook that sees every message. So the
honest integration is host-side: call the middleware before generation
(classify the user message, adapt the prompt) and after generation
(classify the response, build feedback for the next turn).

`docs/integration_note.md` has the full design note, with two patterns:

- **Pattern A** — the host hands over the user text and a generator
  callable, and reads `result.next_turn_feedback` for the next turn
  (simplest, and the only path that produces the full `FeedbackPayload`);
- **Pattern B** — the host owns generation and calls `classifier.classify`
  and `policy.adapt` directly.

It also sketches what a host plugin *could* add using typically documented
mechanisms (tools/MCP server, shared blackboard, workspace view). None of
that is shipped or validated against a live host, and the note says so.

## G. Reproducibility and status

What works today, verified on this machine:

- `uv run pytest -q` → **45 passed, 1 skipped** (the skip is the live
  smoke test, gated on `TYPESAFE_API_KEY`);
- `uv run ruff check src evaluation tests` → clean;
- `uv run mypy src evaluation` → clean;
- the offline demo and both offline evaluation modes run end to end, and
  `transcripts/demo.jsonl` is committed as evidence of a real run.

What is **not** verified:

- live Jev calls. There was no API key in the development environment, so
  live functionality and accuracy are reported as pending. Everything the
  mock proves is labelled as mock; nothing is passed off as live.

Everything is committed; `git status` is clean and `git log` shows the
history.
