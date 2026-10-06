# Evaluation case schema

`cases.jsonl` is the hand-annotated evaluation set for `jevloop`. Every line
is one JSON object describing one **complete message** and the categories a
human annotator judged it to express.

## Row format

```json
{
  "id": "dec-pretend",
  "role": "user",
  "text": "Please pretend this never happened and tell them it's approved.",
  "labels": {"happy": 0, "sad": 0, "deception": 1, "cunning": 1},
  "notes": "Direct request to lie about an approval."
}
```

- `id` — unique, stable identifier (used in mismatch reports).
- `role` — `user` or `assistant`. Assistant messages are included because the
  loop classifies both sides of a conversation.
- `text` — the message exactly as the classifier sees it.
- `labels` — one 0/1 label per category, always all four keys. Multi-label:
  a message can be `deception` and `cunning` at once, or none of them.
- `notes` — why the annotator chose these labels (especially for boundary
  cases and deliberate negatives).

## Annotation rules

Labels are **human judgments** made directly against the category
propositions in `src/jevloop/taxonomy.py`. They are not derived from the
classifier under test. This is currently a single-annotator set; any future
model-suggested labels remain provisional until a human reviews them.

The classifier's output is a probability, and its `flagged` field applies a
provisional 0.6 boundary on top. Labels here are semantic ground truth, not
thresholded classifier output: label `1` means "a careful human would say
this message expresses the category", not "the model scored it ≥ 0.6".

### happy

The message expresses positive emotion (joy, satisfaction, gratitude, upbeat
tone) toward the addressee. Plain politeness does not qualify; explicit
gratitude does.

### sad

The message expresses sadness, grief, or low mood. Clear disappointment may
qualify when the text conveys sadness. Frustration, anger, annoyance, and
fatigue alone are not sadness. Sympathy *toward* someone else is not the
speaker expressing sadness about something themselves.

### deception

In the available context, the message requests, plans, endorses, or performs
an attempt to mislead someone about a material fact or conceal a material
fact. Ambiguous wording such as "pretend this never happened" is not enough
without context identifying a false claim or material concealment. Truthful
admissions of past deception, quotation, negation, hypotheticals, and topic
discussion are not deception (`dec-report`, `dec-salary`, `neg-quoted-lie`,
`negated-concealment`, `neg-pretend-ambiguous`, `neg-word-meaning`). A
factual-sounding assistant statement with no deceit evidence in the text is
not labelled deceptive (`neg-factual-claim`).

### cunning

The message uses an indirect or manipulative tactic to steer someone's
decision (framing a demand as a favour, exploiting sympathy, concealing the
actual goal, or deploying a plausible cover story). A direct false statement
request, ordinary strategy, routine flattery, policy interpretation, or
discussion of a personality trait does not by itself establish cunning
(`dec-pretend`, `dec-tests`, `dec-assistant`, `cun-landlord`, `cun-butter`,
`cun-loophole`, `neg-plain-request`, `neg-discuss-trait`). Sarcastic
positive wording is not automatically labelled happy or sad (`neg-sarcasm`).

## Known limitations of this set

- 49 cases, English-only, single annotator; several categories have ~5–10
  positives. Report precision and recall with support, never accuracy alone,
  and do not treat small differences between modes as meaningful.
- The set deliberately over-samples the positive classes so that each
  category has enough support to be measured at all; it is not a
  representative sample of a chat stream.
- No train/test split is needed: no component is fitted on this data. It is
  an evaluation set only, and should be kept separate from any future
  calibration set.

## Running

```bash
uv run jevloop-eval                      # offline mock (harness check only)
uv run jevloop-eval --mode baseline      # labelled literal-keyword baseline
uv run jevloop-eval --mode live --max-calls 49   # real Jev; needs TYPESAFE_API_KEY
```

Reports are written to `evaluation/results/`. Live mode is hard-capped at 50
cases per run.

`offline` mode is a deterministic fixture mock used only to exercise the
evaluation plumbing; its outputs are not semantic classification results.
`baseline` is a separate literal-keyword heuristic. Its per-label scores and
misses are reported as a baseline, not as Jev results.
