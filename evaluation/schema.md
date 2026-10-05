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
propositions in `src/jevloop/taxonomy.py`. They were written before any
evaluation run and are never derived from the classifier under test.

The classifier's output is a probability, and its `flagged` field applies a
provisional 0.6 boundary on top. Labels here are semantic ground truth, not
thresholded classifier output: label `1` means "a careful human would say
this message expresses the category", not "the model scored it ≥ 0.6".

### happy

The message expresses positive emotion (joy, satisfaction, gratitude, upbeat
tone) toward the addressee. Plain politeness does not qualify; explicit
gratitude does.

### sad

The message expresses negative emotion (sadness, disappointment,
frustration, distress, unhappiness). Sympathy *toward* someone else is not
the speaker expressing sadness about something themselves.

### deception

The utterance itself attempts to mislead the addressee about a material fact
(false claim presented as true, or an attempt to hide a material fact).
Merely mentioning, quoting, defined as a word, or hypothetically discussing
deception is **not** deception (`neg-hypothetical-lie`, `neg-word-meaning`).
A factual-sounding assistant statement with no deceit evidence in the text is
not labelled deceptive (`neg-factual-claim`).

### cunning

The message uses a strategic, indirect, or deceptive-adjacent manipulation
tactic (framing a demand as a favour, exploiting sympathy, concealing the
actual goal). An ordinary, direct strategic request is **not** cunning
(`neg-plain-request`).

## Known limitations of this set

- 42 cases, English-only, single annotator; several categories have ~7–12
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
uv run jevloop-eval --mode baseline      # labelled regex baseline
uv run jevloop-eval --mode live --max-calls 42   # real Jev; needs TYPESAFE_API_KEY
```

Reports are written to `evaluation/results/`. Live mode is hard-capped at 50
cases per run.

`offline` mode exists to validate the harness, not to measure quality: a few
cases intentionally overlap the mock's fixture table so a working run shows
non-zero true positives. `baseline` is the separately labelled regex
classifier — weak by design, and reported as such.
