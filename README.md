# JEV-feedbackchat

**Aditya Shenvi** — reusable Python middleware for classifying complete user
and assistant messages with Jev, returning structured feedback, and adapting
the instructions actually consumed by an agent generator.

The interview requirements are Jev classification for both roles, extensible
categories (happy, sad, deception, cunning), feedback-driven prompt rephrasing,
and a Git submission. Python, CLI, deterministic policies and the evaluation
set are implementation choices. No deployment is required.

```text
user + bounded history -> Jev -> scores/decisions/abstentions
  -> named prompt policy -> effective prompt + original request -> generator
  -> complete assistant message -> Jev -> feedback -> NEXT generation's prompt
```

## First five minutes (Windows PowerShell)

Prerequisites: Git, Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
From the cloned repository root:

```powershell
uv sync --frozen --extra live
uv run jevloop --mode offline --generator-mode mock
```

One command runs setup, the demonstration, offline tests and both evaluations:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\examples\five_minute_demo.ps1
```

Walkthrough (all scores here are **deterministic mock fixtures**):

1. Gratitude adds `acknowledge_positive` to the actual system prompt.
2. The fixture request adds `escalate_deception_suspect`. The echo response
   also matches that fixture; this is plumbing, not a semantic judgment.
3. A neutral follow-up receives `keep_assistant_honest`, sourced from the
   previous assistant classification. Inspect the printed effective prompt.
4. The sadness fixture adds `acknowledge_sadness`; assistant sadness alone
   never triggers revision.

The CLI prints both roles, policies, source signals, full effective system
prompt and bounded prior conversation. JSONL output defaults to ignored
`transcripts/run.jsonl`. The tracked `transcripts/demo.jsonl` is synthetic
mock output. Avoid saving real conversations in submission files.

Try custom messages in one session; repeat `--message` to prove the next turn:

```powershell
uv run jevloop --mode offline --generator-mode mock --message "My custom request" --message "A follow-up"
```

Unseen mock input produces fixed default scores and explicit abstentions.
Fixture substring matches can misclassify quotation or negation; this mock
checks orchestration only. It is never a semantic classifier. Classifier and
generator modes are printed separately. The CLI defaults to offline/mock.
The optional `.sh` script targets Unix or explicitly selected Git Bash and
has not been verified in this audit. Plain `bash` on Windows can resolve to
a WSL launcher; the PowerShell entry point above requires neither WSL nor Bash.

## What to inspect

| Requirement | Implementation | Meaningful test |
|---|---|---|
| Classify user and assistant | `src/jevloop/loop.py`: `FeedbackLoop._run_turn_for` | `tests/test_loop.py`: `test_response_feedback_updates_next_prompt` |
| Configurable categories, separate dimensions | `taxonomy.py`: `Category`, `DecisionPolicy`; `adapter.py`: `JevClassifier` (under `src/jevloop`) | `tests/test_adapter.py`: `test_custom_categories_flow_through_classifier` |
| Real Jev request and validated answer | `src/jevloop/adapter.py`: `LiveJevClassifier.classify` | `tests/test_adapter.py`; `tests/test_provider_failures.py` |
| Controlled prompt rephrasing | `src/jevloop/prompt_policy.py`: `AdaptationPolicy.adapt` | `tests/test_prompt_policy.py` |
| Generator consumes exact prompt/request/context | `src/jevloop/generator.py`: `OpenAICompatibleGenerator.__call__` | `tests/test_generator.py`: `test_generator_sends_exact_prompt_and_bounded_history` |
| Assistant feedback changes NEXT prompt | `src/jevloop/loop.py`: `_run_turn_for` | `test_response_feedback_updates_next_prompt` in `tests/test_loop.py` |
| Classify bounded revisions | `src/jevloop/loop.py`: `_run_turn_for` | `test_deceptive_revision_is_classified_once_and_loop_stops` in `tests/test_loop.py` |

`Category(kind="intent")` allows task intent independently of emotion and
behaviour; no intent label is enabled by default. Supply categories and a
`DecisionPolicy` to the classifier, and optional `Rule` objects to the policy.
Scores are independent Noul probabilities, never normalized or renamed
confidence. Boundaries default to 0.6 and are provisional. Abstentions are
retained and never activate a policy.

Policies append named, versioned instructions while preserving the trusted
base and verbatim user request. Their rationale records a score-based policy
decision, not invented Jev reasoning or evidence. Frustration alone is not
sadness; ambiguous pretend wording, quotations and negation need context;
ordinary strategy is not cunning. Risk signals concern message content,
not permanent character. Revision is optional, limited to one callback per
turn, and every revised response is classified before subsequent feedback.

## Configure live providers separately

The Jev contract follows the [official Python SDK](https://docs.typesafe.ai/sdk/python/)
and [usage reference](https://docs.typesafe.ai/sdk/python/usage): `system_one`
with a Noul per category, then `response.nouls[id].noul`. `uv.lock` pins the
installed SDK to 0.7.2. No live-provider failure silently changes to mock mode.

Provide `TYPESAFE_API_KEY` securely in the process environment. For a local
PowerShell session, hidden keyboard input avoids writing the key in a command:

```powershell
$jevSecret = Read-Host "Jev API key" -AsSecureString
$env:TYPESAFE_API_KEY = [System.Management.Automation.PSCredential]::new("unused", $jevSecret).GetNetworkCredential().Password
```

Use the same pattern with `JEVLOOP_GENERATOR_API_KEY` for the generator.
The conversion was checked in PowerShell 5.1 using a synthetic placeholder.
`.env.example`
is a template; this project **does not load .env automatically**. Optional
`TYPESAFE_DEFAULT_MODEL` defaults to `jev-latest`; `TYPESAFE_BASE_URL` overrides
the API root. First smoke-test, then run one complete Jev-classified turn:

```powershell
uv run pytest -m live -x
uv run jevloop --mode live --generator-mode mock --message "Help me plan tomorrow"
uv run jevloop-eval --mode live --max-calls 49
```

Smoke: four complete messages. A single CLI turn: two classification requests.
Evaluation: at most 49 requests here, hard cap 50. SDK retries default to zero
for a predictable paid budget; library callers may explicitly set
`max_retries` to 0–2. Jev's overall caller deadline is 30 seconds, including
SDK retry waiting. Generator timeout/deadline is 30 seconds with zero retries
(to avoid duplicate generation charges). Both are configurable constructor
arguments. A timed-out synchronous request can remain in flight until its
socket timeout; caller waiting is bounded. No automatic fallback follows.

The real generator needs an application API endpoint/model and, if required,
its separately issued credential. A chat subscription does not imply API access.
After securely setting `JEVLOOP_GENERATOR_API_KEY`:

```powershell
$env:JEVLOOP_GENERATOR_BASE_URL = "https://your-provider.example/v1"
$env:JEVLOOP_GENERATOR_MODEL = "your-application-model"
uv run jevloop --mode live --generator-mode openai-compatible --message "Help me plan tomorrow"
```

The endpoint must implement OpenAI-compatible chat completions. HTTPS is
required except loopback HTTP; redirects are rejected. History is bounded to
20 messages / 12,000 characters; the current request is separate and unchanged.
`JEVLOOP_GENERATOR_MODE` optionally sets the default generator selection.

## Tests and evaluation

```powershell
uv run pytest -q
uv run ruff check src evaluation tests
uv run ruff format --check src evaluation tests
uv run mypy src evaluation
uv run jevloop-eval --mode offline
uv run jevloop-eval --mode baseline
```

Default tests exclude paid live tests even if a key is present. Evaluation
uses 49 independently audited **provisional agent-authored annotations**;
human review is pending. See [annotation rules](evaluation/schema.md).
Reports contain per-label confusion counts, precision, recall, F1 and failure
cases. F1 is not accuracy. Mock reports measure harness behaviour only;
live reports use a separate mode/model and filenames.

The so-called regex baseline is actually literal keyword matching. Most
fixed scores are at most 0.55, below the 0.6 boundary. Only `sad` has 0.65;
positive sadness cases contain no literal `sad`, while `neg-not-sad` does.
Thus it has no true positives and one false positive. These limitations are
reported rather than adjusting examples or boundaries to manufacture success.

## Host invocation and verified status

The host must invoke this middleware before generation and after each complete
response. Plugin tools, agents, workspace views and persistent state can
expose/store its outputs, but a callable tool or MCP server does not intercept
every chat message. No automatic interception hook was established in the
reviewed extension documentation. No native wrapper or UI is shipped.
See the [agent contract](docs/integration_note.md).

[VERIFICATION.md](VERIFICATION.md) records executed checks, fresh-copy setup,
measured mock/baseline failures, Git investigation and remaining limitations.
Implementation and offline verification complete; live verification pending.
