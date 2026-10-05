"""Offline/live evaluation runner for jevloop classifiers.

Runs a fixed, hand-annotated case file against one classifier mode, computes
per-category precision/recall/F1, prints a table plus mismatches, and writes a
JSON report under ``evaluation/results/``.

Design rules:

- Labels are HAND-ANNOTATED ground truth (see ``schema.md``); they are never
  produced by the classifier under test.
- ``offline`` mode is a labelled fixture mock and exists to validate the
  harness end-to-end; its numbers say nothing about Jev quality.
- Live mode costs one Jev call per case, so it must be requested explicitly
  and is hard-capped at ``LIVE_CALL_HARD_CAP`` cases per run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from jevloop import CATEGORY_CATALOG, JevClassifier, Message, Role, build_classifier
from jevloop.adapter import ClassifierError

CATEGORY_IDS: tuple[str, ...] = tuple(category.id for category in CATEGORY_CATALOG)
DEFAULT_CASES_PATH = Path(__file__).with_name("cases.jsonl")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
LIVE_CALL_HARD_CAP = 50
VALID_ROLES: tuple[str, ...] = ("user", "assistant")

REPORT_NOTE = (
    "Decision boundaries are provisional (see DecisionPolicy.as_provenance). "
    "The offline classifier is a labelled fixture mock used to validate the "
    "harness, not a measurement of Jev. This set is small and single-annotator: "
    "read precision/recall together with support, not accuracy alone."
)


class EvaluationError(RuntimeError):
    """Raised when a case file cannot be used as-is."""


@dataclass(frozen=True)
class Case:
    id: str
    role: Role
    text: str
    labels: dict[str, int]
    notes: str


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.tn + self.fp

    @property
    def predicted_positives(self) -> int:
        return self.tp + self.fp

    def precision(self) -> float | None:
        return _safe_div(self.tp, self.tp + self.fp)

    def recall(self) -> float | None:
        return _safe_div(self.tp, self.tp + self.fn)

    def f1(self) -> float | None:
        precision, recall = self.precision(), self.recall()
        if precision is None or recall is None or precision + recall == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    def accuracy(self) -> float | None:
        return _safe_div(self.tp + self.tn, self.tp + self.fp + self.fn + self.tn)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "positives": self.positives,
            "negatives": self.negatives,
            "predicted_positives": self.predicted_positives,
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "accuracy": self.accuracy(),
        }


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _parse_case(raw: Any, line_number: int, seen: set[str]) -> Case:
    if not isinstance(raw, dict):
        raise EvaluationError(f"line {line_number}: expected a JSON object")
    for key in ("id", "role", "text", "labels", "notes"):
        _require(key in raw, f"line {line_number}: missing key {key!r}")
    case_id = raw["id"]
    _require(
        isinstance(case_id, str) and bool(case_id),
        f"line {line_number}: id must be a non-empty string",
    )
    _require(case_id not in seen, f"line {line_number}: duplicate id {case_id!r}")
    role = raw["role"]
    _require(
        role in VALID_ROLES,
        f"line {line_number} ({case_id}): role must be one of {VALID_ROLES}",
    )
    text = raw["text"]
    _require(
        isinstance(text, str) and bool(text.strip()),
        f"line {line_number} ({case_id}): text must be a non-empty string",
    )
    notes = raw["notes"]
    _require(
        isinstance(notes, str) and bool(notes.strip()),
        f"line {line_number} ({case_id}): notes must be a non-empty string",
    )
    labels = raw["labels"]
    _require(isinstance(labels, dict), f"line {line_number} ({case_id}): labels must be an object")
    _require(
        set(labels) == set(CATEGORY_IDS),
        f"line {line_number} ({case_id}): labels must have exactly {sorted(CATEGORY_IDS)}",
    )
    for category_id, value in labels.items():
        _require(
            not isinstance(value, bool) and value in (0, 1),
            f"line {line_number} ({case_id}): label {category_id!r} must be 0 or 1",
        )
    seen.add(case_id)
    return Case(
        id=case_id,
        role=cast(Role, role),
        text=text,
        labels=dict(labels),
        notes=notes,
    )


def load_cases(path: Path) -> list[Case]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvaluationError(f"cannot read case file {path}: {exc}") from exc
    cases: list[Case] = []
    seen: set[str] = set()
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        cases.append(_parse_case(raw, line_number, seen))
    _require(bool(cases), f"{path}: no cases found")
    return cases


def evaluate(cases: list[Case], classifier: JevClassifier, *, mode: str) -> dict[str, Any]:
    confusion = {category_id: Confusion() for category_id in CATEGORY_IDS}
    predictions: list[dict[str, Any]] = []
    model = "unknown"
    for case in cases:
        message = Message(id=case.id, role=case.role, text=case.text, turn=0)
        classification = classifier.classify(message)
        model = classification.model or model
        scored = classification.scores_by_category()
        flagged_ids = set(classification.flagged_categories())
        predicted = {category_id: category_id in flagged_ids for category_id in CATEGORY_IDS}
        probabilities: dict[str, float | None] = {}
        for category_id in CATEGORY_IDS:
            score = scored.get(category_id)
            probabilities[category_id] = score.probability if score is not None else None
        for category_id in CATEGORY_IDS:
            cell = confusion[category_id]
            label = case.labels[category_id]
            hit = int(predicted[category_id])
            if label == 1 and hit:
                cell.tp += 1
            elif label == 0 and hit:
                cell.fp += 1
            elif label == 1:
                cell.fn += 1
            else:
                cell.tn += 1
        predictions.append(
            {
                "id": case.id,
                "role": case.role,
                "text": case.text,
                "labels": case.labels,
                "predicted": predicted,
                "probabilities": probabilities,
                "notes": case.notes,
                "model": classification.model,
                "request_id": classification.request_id,
            }
        )

    categories_report = {category_id: cell.as_dict() for category_id, cell in confusion.items()}
    macro: dict[str, float | None] = {}
    for metric in ("precision", "recall", "f1"):
        values = [
            float(entry[metric])
            for entry in categories_report.values()
            if entry[metric] is not None
        ]
        macro[metric] = sum(values) / len(values) if values else None
    return {
        "mode": mode,
        "classifier_mode": classifier.mode,
        "model": model,
        "case_count": len(cases),
        "categories": categories_report,
        "macro_average": macro,
        "note": REPORT_NOTE,
        "cases": predictions,
    }


def _format_metric(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def format_table(report: dict[str, Any]) -> str:
    lines = [
        (
            f"mode={report['mode']}  classifier={report['classifier_mode']}  "
            f"model={report['model']}  cases={report['case_count']}"
        ),
        "",
        f"{'category':<12} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}  "
        f"{'prec':>6} {'rec':>6} {'f1':>6}   support",
    ]
    for category_id, metrics in report["categories"].items():
        support = f"+{metrics['positives']}/-{metrics['negatives']}"
        lines.append(
            f"{category_id:<12} {metrics['tp']:>3} {metrics['fp']:>3} {metrics['fn']:>3} "
            f"{metrics['tn']:>3}  {_format_metric(metrics['precision']):>6} "
            f"{_format_metric(metrics['recall']):>6} {_format_metric(metrics['f1']):>6}   {support}"
        )
    macro = report["macro_average"]
    lines.append(
        f"{'macro':<12}     {'':>3} {'':>3} {'':>3}  "
        f"{_format_metric(macro['precision']):>6} {_format_metric(macro['recall']):>6} "
        f"{_format_metric(macro['f1']):>6}"
    )
    return "\n".join(lines)


def format_mismatches(report: dict[str, Any]) -> str:
    lines: list[str] = []
    for case in report["cases"]:
        for category_id, label in case["labels"].items():
            hit = int(case["predicted"][category_id])
            if hit == label:
                continue
            probability = case["probabilities"][category_id]
            p_text = f"{probability:.3f}" if probability is not None else "n/a"
            lines.append(
                f"  {case['id']:<22} {category_id:<10} label={label} predicted={hit} p={p_text}"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevloop-eval",
        description="Evaluate a jevloop classifier against hand-annotated cases.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "baseline", "live"),
        default="offline",
        help="classifier to evaluate (default: offline fixture mock)",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help=f"JSONL case file (default: {DEFAULT_CASES_PATH.name})",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help=f"live mode call budget (default/hard cap: {LIVE_CALL_HARD_CAP})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="directory for JSON reports",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cases = load_cases(args.cases)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.mode == "live":
        if not os.environ.get("TYPESAFE_API_KEY"):
            print(
                "error: TYPESAFE_API_KEY is not set; refusing to run live mode.",
                file=sys.stderr,
            )
            return 2
        max_calls = args.max_calls if args.max_calls is not None else LIVE_CALL_HARD_CAP
        if max_calls < 1 or max_calls > LIVE_CALL_HARD_CAP:
            print(
                f"error: --max-calls must be between 1 and {LIVE_CALL_HARD_CAP}.", file=sys.stderr
            )
            return 2
        if len(cases) > max_calls:
            print(
                f"error: {len(cases)} cases exceed the live call budget of {max_calls}; "
                f"trim the case file or raise --max-calls (hard cap {LIVE_CALL_HARD_CAP}).",
                file=sys.stderr,
            )
            return 2

    classifier = build_classifier(mode=args.mode)
    started = time.perf_counter()
    try:
        report = evaluate(cases, classifier, mode=args.mode)
    except ClassifierError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["case_file"] = str(args.cases)
    report["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(format_table(report))
    mismatches = format_mismatches(report)
    if mismatches:
        shown = mismatches.splitlines()[:10]
        print(f"\nmismatches (showing {len(shown)}; full list in the report):")
        print("\n".join(shown))

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out / f"{args.mode}-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
