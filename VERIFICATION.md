# Submission verification — 2026-10-06

## Executed checks

Environment: Windows 11 (build 26300), Windows PowerShell 5.1.26100.9549,
Python 3.12.13, uv 0.11.24, Git 2.55.0.windows.1, typesafe-sdk 0.7.2.
Author: Aditya Shenvi. No deployment or interviewer contact occurred.

| Command | Actual result |
|---|---|
| `uv sync --frozen --extra live` | Exit 0; 29 packages checked, local package rebuilt after metadata edit |
| `uv run pytest -q` | 88 passed, 1 live test deselected |
| `uv run ruff check src evaluation tests` | All checks passed |
| `uv run ruff format --check src evaluation tests` | 21 files already formatted |
| `uv run mypy src evaluation` | Success: no issues in 11 source files |
| `powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\examples\five_minute_demo.ps1` | Exit 0; setup, 4 mock turns, tests and two evaluations |
| `uv run jevloop --help` | Confirmed modes, transcript, base prompt and repeatable `--message` options |
| `uv run jevloop --mode live --generator-mode mock --message "Help me plan tomorrow"` | Expected exit 2: missing Jev key; no live request or mock fallback |
| `uv run jevloop --mode offline --generator-mode openai-compatible --message "Help me plan tomorrow"` | Expected exit 2: missing generator endpoint/model; no fallback |
| `uv run pytest -m live -x` | 1 skipped, 88 deselected; no TYPESAFE_API_KEY |
| `uv run jevloop-eval --mode offline` | 49 cases; fixture plumbing only |
| `uv run jevloop-eval --mode baseline` | 49 cases; separate keyword baseline |
| `git diff --check` | Exit 0 |

New-test development initially exposed five test construction mistakes
(missing `EffectivePrompt.conversation` and wrong list indexing). They were
corrected using the actual contracts. Assertions were retained; final checks
above pass. Ruff fixes were import sorting and formatting, not suppressed rules.

Fresh-copy verification: exported only staged submission files with
`git checkout-index --all --prefix=<temporary-directory>/` into
`%TEMP%\jev-submission-clean-20261006-154111`. No `.venv` existed. Ran
`uv sync --frozen --extra live`: **installed 29 packages** into a new `.venv`.
Python's prefix and `jevloop.__file__` both resolve inside that copy.
Then reran pytest (88 passed, 1 deselected), Ruff lint/format (21 files), mypy
(11 files), the PowerShell walkthrough (exit 0), the README's two custom
messages (exit 0), both evaluations (49 cases each), and the live test
(1 skipped, 88 deselected). All executed checks passed. Parsed lockfile contents match the original; the normalized (LF) SHA256
was `a990bdd2a127f744487afcee64595add2623e68b2a25309ca2d0cef21a2c5569`.
No pre-existing environment, untracked source or hidden credentials was copied.

## Readiness checklist

| Status | Check | Evidence |
|---|---|---|
| PASS | Both complete message roles classified | `FeedbackLoop._run_turn_for`; loop integration tests |
| PASS | Extensible categories and independent scores | Custom-category test; real SDK mock transport returns total probability 3.2 without normalization |
| PASS | Actual prompt and original request reach generator | Exact prompt, context and feedback assertions in loop/generator tests |
| PASS | Assistant feedback affects subsequent generation | Specific deception 0.91 adds `keep_assistant_honest` in second generator call |
| PASS | Every revision classified and bounded | Deceptive revision is classified once and stops even while still flagged |
| PASS | Real provider contracts and failure handling | SDK HTTP tests: 401/403/429/500, malformed answers, bounded retry/deadline; generator contract/error tests |
| PASS | Offline demo and custom input | Deterministic fixture/echo modes; CLI custom-input test |
| PASS | Submission file hygiene | 34 indexed files scanned; no credential-pattern, excluded-name, database, environment or machine-path findings; local Markdown links resolve |
| PASS | Git audit and recovery | All refs/reflogs/stash inspected, diff checked, external recovery bundle verified |
| PASS | Reproducible setup | Fresh staged-files export, own environment, lockfile setup and repeated checks |
| NOT VERIFIED | Live Jev, live evaluation and live end-to-end turn | No process API key; no local .env file; zero live requests made |
| NOT VERIFIED | Live generator | No application endpoint/model/key configured; fake transport tested |
| NOT VERIFIED | Human annotation approval | Agent-authored annotations independently audited, still provisional |
| NOT VERIFIED | Native plugin or automatic interception | No wrapper shipped; host must invoke message boundaries |

## Measured evaluation (not Jev quality)

Default provisional boundaries: 0.6 for all four independent propositions.
Models: `offline-mock-fixture` and `rules-baseline`; 49 unchanged input texts.
Five annotations were corrected for unsupported sadness/falsehood assumptions.
No evaluation input or fixture was changed to improve predictions.

| Mode | Label | TP | FP | FN | TN | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mock | happy | 1 | 0 | 7 | 41 | 1.000 | .125 | .222 |
| Mock | sad | 0 | 0 | 3 | 46 | undefined | 0 | 0 |
| Mock | deception | 0 | 1 | 7 | 41 | 0 | 0 | 0 |
| Mock | cunning | 0 | 0 | 5 | 44 | undefined | 0 | 0 |
| Baseline | happy | 0 | 0 | 8 | 41 | undefined | 0 | 0 |
| Baseline | sad | 0 | 1 | 3 | 45 | 0 | 0 | 0 |
| Baseline | deception | 0 | 0 | 7 | 42 | undefined | 0 | 0 |
| Baseline | cunning | 0 | 0 | 5 | 44 | undefined | 0 | 0 |

Mock's only true positive is the fixture overlap `happy-helped`; its deception
false positive is `dec-pretend`. Baseline has no true positives; `neg-not-sad`
is its false positive. Scores mostly at/below 0.55 cannot cross 0.6; literal
`sad` scores 0.65 and ignores negation. This is a keyword baseline, not regex
semantic classification. F1 is not accuracy. Undefined precision means no
positive predictions; F1 is zero when positive labels were missed.

Abstentions are retained in reports and counted as unflagged in this binary
table. Mock unknowns therefore count as missed positives for harness reporting;
these values do not measure semantic ability. Reports include every failure.
There are 43 mock abstentions per label; the baseline has no abstentions.

Mock false negatives by label:

- happy: `happy-thanks`, `happy-thrilled`, `happy-hilarious`, `happy-team`,
  `happy-results`, `happy-assistant`, `happy-progress`.
- sad: `sad-disappointed`, `sad-low-week`, `sad-heartbroken`.
- deception: `dec-meeting`, `dec-tests`, `dec-unseen`, `cun-landlord`,
  `cun-invoice`, `cun-excuse`, `cun-favor`.
- cunning: `cun-guilt`, `cun-invoice`, `cun-obligated`, `cun-excuse`, `cun-favor`.

The baseline misses those cases plus `happy-helped`, i.e. every positive label.

## Git investigation and limitations

Executed `git log --all --graph --decorate --oneline`,
`git for-each-ref --contains=1a0ed9e --format="%(refname) %(objectname)"`,
all-ref enumeration, stash list, all reflogs, `git cat-file -t 1a0ed9e`, and
`git fsck --full --no-reflogs --unreachable`.

The contains query reports **malformed object name**; cat-file says **not a
valid object name**. The old object is absent in this repository, so no actual
retaining reference can be identified here. Stash and tags are absent.
Nonbranch `refs/codex/turn-diffs/...` references point to tree objects
`6bc31985...` and `7320a5c2...`, not that commit. ORIG_HEAD was not used to
explain `--all`. fsck exited 0 without unreachable objects. No metadata
deletion, history rewrite or pruning was performed.

A recovery Git bundle was saved outside the submission at
`%TEMP%\jev-submission-recovery-20261006-153845\before-finalization.bundle`.
Work is on `codex/final-scope-quality`, using the existing Git identity and
authorized origin `https://github.com/aditya4232/JEV-feedbackchat.git`.

Remaining limits: live parsing/semantic behaviour remains pending credentials;
human annotation approval is pending; policies/thresholds are provisional;
synchronous timeouts bound caller waiting but cannot cancel an already in-flight
request. Conversation state is per loop instance, in memory, with serialized
turns required. No native host runtime verification is claimed.
