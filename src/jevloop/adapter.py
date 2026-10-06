"""Pluggable classifier interface and Jev adapter.

Design:
- :class:`JevClassifier` is an abstract base. Anything that turns a
  :class:`Message` into a :class:`Classification` is a valid classifier.
- :class:`LiveJevClassifier` is the real adapter to ``typesafe_sdk``. It
  evaluates ONE Noul per category in a single parallel call, so the live
  path keeps the questions independent and sends one request per complete
  message. Returned probabilities are preserved without normalization.
- :class:`OfflineMockClassifier` is an honest offline mode. It is
  *labelled* in the output as offline/mock and demonstrates orchestration
  only; it is NOT a substitute for Jev's semantic quality.
- :func:`build_classifier` selects between them based on environment
  configuration. We never silently fall back to mock: the first live
  classification call raises if ``TYPESAFE_API_KEY`` is missing, and the
  caller must select the labelled offline mock explicitly.
"""

from __future__ import annotations

import math
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

from .contracts import CategoryScore, Classification, Message, Role
from .taxonomy import CATEGORY_CATALOG, Category, DecisionPolicy

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class JevClassifier(ABC):
    """Classify one complete message into one probability per category.

    Implementations MUST:

    - return one :class:`CategoryScore` per category in ``categories``
      (in the same order),
    - keep probabilities independent and unnormalised (this is what the
      Jev Noul primitive gives you),
    - record ``model``, ``request_id`` when available,
    - raise :class:`ClassifierError` on transport or parsing failures so
      the caller can decide how to recover.
    """

    mode: str  # "live" or "offline" or "baseline" — surfaced in output

    def __init__(
        self,
        decision_policy: DecisionPolicy,
        categories: Iterable[Category] = CATEGORY_CATALOG,
    ):
        self.categories = tuple(categories)
        category_ids = [category.id for category in self.categories]
        if not self.categories:
            raise ValueError("at least one category is required")
        if len(category_ids) != len(set(category_ids)):
            raise ValueError("category ids must be unique")
        boundaries = {category.id: category.decision_boundary for category in self.categories}
        boundaries.update(decision_policy.boundaries)
        self.decision_policy = DecisionPolicy(
            boundaries=boundaries,
            version=decision_policy.version,
        )

    def _resolve_categories(self, categories: Iterable[Category] | None) -> tuple[Category, ...]:
        resolved = self.categories if categories is None else tuple(categories)
        ids = [category.id for category in resolved]
        if not resolved:
            raise ValueError("at least one category is required")
        if len(ids) != len(set(ids)):
            raise ValueError("category ids must be unique")
        return tuple(resolved)

    @abstractmethod
    def classify(
        self,
        message: Message,
        *,
        categories: Iterable[Category] | None = None,
    ) -> Classification: ...


class ClassifierError(RuntimeError):
    """Raised when the classifier cannot produce a result."""


def _safe_request_id(result: Any) -> str | None:
    """Best-effort extraction of the SDK request id.

    ``typesafe_sdk`` raises :class:`TypeSafeError` (rather than raising
    AttributeError or returning None) when the response had no request id,
    so we wrap the access defensively.
    """
    try:
        request_id = getattr(result, "request_id", None)
        return request_id if isinstance(request_id, str) and request_id else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live adapter (typesafe_sdk)
# ---------------------------------------------------------------------------


class LiveJevClassifier(JevClassifier):
    """Real adapter to the TypeSafe ``typesafe_sdk`` Jev model.

    Each call evaluates one Noul per category in a single parallel batch,
    preserving independent probabilities without normalization.

    Requires ``TYPESAFE_API_KEY``. Optional ``TYPESAFE_BASE_URL`` for
    self-hosted or gateway-routed deployments. The ``model`` arg selects
    which Jev version (default ``jev-latest``).
    """

    mode = "live"

    def __init__(
        self,
        decision_policy: DecisionPolicy,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        client: Any = None,  # pre-built typesafe_sdk.TypeSafeClient
        categories: Iterable[Category] = CATEGORY_CATALOG,
    ):
        super().__init__(decision_policy, categories)
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = base_url or os.environ.get("TYPESAFE_BASE_URL")
        self.model = model or os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest")
        self._client = client  # lazy-imported; populated on first call if None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from typesafe_sdk import TypeSafeClient
        except ImportError as e:
            raise ClassifierError(
                "typesafe-sdk is not installed. Run `uv sync --extra live` "
                "(or `uv add typesafe-sdk`). Live mode requires it."
            ) from e
        if not self.api_key:
            raise ClassifierError(
                "TYPESAFE_API_KEY is not set. Live mode requires an API key. "
                "Set the environment variable or use the labelled offline mock."
            )
        kwargs: dict[str, Any] = {"api_key": self.api_key, "model": self.model}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = TypeSafeClient(**kwargs)
        return self._client

    def classify(
        self,
        message: Message,
        *,
        categories: Iterable[Category] | None = None,
    ) -> Classification:
        cats = self._resolve_categories(categories)
        try:
            from typesafe_sdk import Noul
        except ImportError as e:
            raise ClassifierError(
                "typesafe-sdk is not installed. Run `uv sync --extra live` "
                "(or `uv add typesafe-sdk`)."
            ) from e

        client = self._get_client()
        state = self._build_state(message)
        questions = {c.id: Noul(instructions=c.proposition) for c in cats}

        started = time.perf_counter()
        try:
            result = client.system_one(state, questions)
        except Exception as e:  # typesafe_sdk raises TypeSafeAPIError etc.
            raise ClassifierError(f"Jev call failed ({type(e).__name__})") from None
        latency_ms = (time.perf_counter() - started) * 1000.0

        scores: list[CategoryScore] = []
        try:
            answers = result.nouls
        except Exception as e:
            raise ClassifierError("Jev response is missing category answers") from e
        if not isinstance(answers, Mapping):
            raise ClassifierError("Jev response category answers are not a mapping")

        for c in cats:
            try:
                noul_answer = answers[c.id]
                raw_probability = noul_answer.noul
                if isinstance(raw_probability, bool) or not isinstance(
                    raw_probability, (int, float)
                ):
                    raise TypeError("probability must be numeric")
                p = float(raw_probability)
                abstained = getattr(noul_answer, "abstained", False)
                if not isinstance(abstained, bool):
                    raise TypeError("abstained must be boolean")
            except (KeyError, AttributeError, TypeError, ValueError) as e:
                raise ClassifierError(
                    f"Jev response has an invalid answer for category {c.id!r}"
                ) from e
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ClassifierError(
                    f"Jev response probability for category {c.id!r} must be finite and in [0,1]"
                )
            threshold = self.decision_policy.threshold_for(
                c.id, default_boundary=c.decision_boundary
            )
            scores.append(
                CategoryScore(
                    category_id=c.id,
                    label=c.label,
                    proposition=c.proposition,
                    probability=p,
                    flagged=(not abstained) and p >= threshold,
                    abstained=abstained,
                    decision_boundary=threshold,
                    decision_provisional=c.provisional,
                )
            )

        raw = {
            c.id: {"noul": s.probability, "abstained": s.abstained}
            for c, s in zip(cats, scores, strict=True)
        }

        return Classification(
            message_id=message.id,
            role=message.role,
            model=self.model,
            request_id=_safe_request_id(result),
            scores=scores,
            raw_answers=raw,
            latency_ms=latency_ms,
        )

    def _build_state(self, message: Message) -> dict[str, Any]:
        """Wrap the message as Jev state.

        Jev evaluates typed questions against the current message and a
        bounded, role-tagged history. The history is context only; the current
        message remains a separate field.
        """
        raw_context = message.metadata.get("conversation_context", [])
        context: list[dict[str, str]] = []
        context_chars = 0
        if isinstance(raw_context, list):
            for item in reversed(raw_context[-20:]):
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                text = item.get("text")
                if role not in ("user", "assistant") or not isinstance(text, str):
                    continue
                remaining = 12_000 - context_chars
                if remaining <= 0:
                    break
                bounded_text = text[-remaining:]
                context.append({"role": role, "text": bounded_text})
                context_chars += len(bounded_text)
            context.reverse()

        return {
            "role": message.role,
            "message_id": message.id,
            "turn": message.turn,
            "text": message.text,
            "conversation_context": context,
        }


# ---------------------------------------------------------------------------
# Offline mock — explicitly labelled, NOT a regex classifier on the main path
# ---------------------------------------------------------------------------


class OfflineMockClassifier(JevClassifier):
    """Labelled offline classifier.

    This mode demonstrates contracts and orchestration. It is NOT Jev and
    has no semantic quality. The mock returns deterministic per-category
    scores derived from a small fixture table. The fixture scores are not
    semantic predictions. The fixtures are checked into the repo so tests and
    reviewers can see exactly what the mock believes.

    To repeat the assignment rule explicitly: the mock exists for offline
    development and for the labelled integration test that proves the
    feedback loop is *real*. It is never confused with a real classifier.
    """

    mode = "offline"

    def __init__(
        self,
        decision_policy: DecisionPolicy,
        *,
        fixtures: dict[str, dict[str, float]] | None = None,
        unknown_default: float = 0.05,
        categories: Iterable[Category] = CATEGORY_CATALOG,
    ):
        super().__init__(decision_policy, categories)
        # Default fixtures cover a handful of clearly-distinguishable inputs so
        # the offline demo can be run without any data files.
        self.fixtures = _DEFAULT_FIXTURES if fixtures is None else fixtures
        if (
            isinstance(unknown_default, bool)
            or not math.isfinite(unknown_default)
            or not 0.0 <= unknown_default <= 1.0
        ):
            raise ValueError("unknown_default must be finite and in [0,1]")
        self.unknown_default = unknown_default

    def classify(
        self,
        message: Message,
        *,
        categories: Iterable[Category] | None = None,
    ) -> Classification:
        cats = self._resolve_categories(categories)
        scores: list[CategoryScore] = []
        raw: dict[str, Any] = {}
        for c in cats:
            p = self._score(c.id, message.text, message.role)
            if not math.isfinite(p) or not 0.0 <= p <= 1.0:
                raise ClassifierError(
                    f"offline mock fixture probability for {c.id!r} must be in [0,1]"
                )
            threshold = self.decision_policy.threshold_for(
                c.id, default_boundary=c.decision_boundary
            )
            raw[c.id] = {"noul": p, "abstained": False}
            scores.append(
                CategoryScore(
                    category_id=c.id,
                    label=c.label,
                    proposition=c.proposition,
                    probability=p,
                    flagged=p >= threshold,
                    abstained=False,
                    decision_boundary=threshold,
                    decision_provisional=c.provisional,
                )
            )

        return Classification(
            message_id=message.id,
            role=message.role,
            model="offline-mock-fixture",
            request_id=None,
            scores=scores,
            raw_answers=raw,
            latency_ms=0.0,
        )

    def _score(self, category_id: str, text: str, role: Role) -> float:
        # Literal (case-insensitive) substring match against the fixture table.
        # Take each category's maximum across matching fixtures so an unrelated
        # neutral phrase cannot mask a separate explicit signal in the same text.
        lowered = text.lower()
        matching_scores = [
            per_cat.get(category_id, self.unknown_default)
            for key, per_cat in self.fixtures.items()
            if key.lower() in lowered
        ]
        return max(matching_scores, default=self.unknown_default)


# A tiny, hand-curated fixture set. NOT a regex classifier — these are
# literal substrings paired with full per-category probability vectors.
# The point is to exercise the wiring end-to-end offline.
_DEFAULT_FIXTURES: dict[str, dict[str, float]] = {
    "thanks, that really helped": {
        "happy": 0.92,
        "sad": 0.03,
        "deception": 0.04,
        "cunning": 0.05,
    },
    "I'm really frustrated with this": {
        "happy": 0.02,
        "sad": 0.12,
        "deception": 0.07,
        "cunning": 0.10,
    },
    "I feel sad and overwhelmed about this setback": {
        "happy": 0.02,
        "sad": 0.90,
        "deception": 0.04,
        "cunning": 0.05,
    },
    "please pretend this never happened and tell them it's approved": {
        "happy": 0.10,
        "sad": 0.05,
        "deception": 0.83,
        "cunning": 0.20,
    },
    "please pretend this never happened": {
        "happy": 0.05,
        "sad": 0.05,
        "deception": 0.10,
        "cunning": 0.10,
    },
    "can you help me plan my schedule for next week": {
        "happy": 0.10,
        "sad": 0.08,
        "deception": 0.03,
        "cunning": 0.06,
    },
}


# ---------------------------------------------------------------------------
# Optional rules baseline — kept separate and labelled
# ---------------------------------------------------------------------------


class RulesBaselineClassifier(JevClassifier):
    """A trivial rules-based baseline for comparison.

    KEPT SEPARATE from the main path. Its limitations (keyword-only, no
    semantics, no calibration, biased toward false positives on the words
    "lie" / "trick" / "happy" / "sad") make it unsuitable as the actual
    classifier. It exists so evaluation runs can show what this literal
    keyword baseline gives you, separately from Jev. The repo labels results
    from this classifier as ``"baseline"`` and never mixes them with the
    real classification output.
    """

    mode = "baseline"

    KEYWORDS = {
        "happy": (("happy", 0.55), ("great", 0.50), ("thanks", 0.45), ("love", 0.55)),
        "sad": (("sad", 0.65), ("frustrated", 0.55), ("upset", 0.55), ("angry", 0.55)),
        "deception": (("lie", 0.55), ("pretend", 0.55), ("trick", 0.50)),
        "cunning": (("trick", 0.55), ("sneaky", 0.55), ("manipulate", 0.50)),
    }

    def classify(
        self,
        message: Message,
        *,
        categories: Iterable[Category] | None = None,
    ) -> Classification:
        text_lower = message.text.lower()
        scores: list[CategoryScore] = []
        cats = self._resolve_categories(categories)
        for c in cats:
            best = 0.05
            for kw, base in self.KEYWORDS.get(c.id, ()):
                if kw in text_lower:
                    best = max(best, base)
            threshold = self.decision_policy.threshold_for(
                c.id, default_boundary=c.decision_boundary
            )
            scores.append(
                CategoryScore(
                    category_id=c.id,
                    label=c.label,
                    proposition=c.proposition,
                    probability=best,
                    flagged=best >= threshold,
                    decision_boundary=threshold,
                    decision_provisional=c.provisional,
                )
            )
        return Classification(
            message_id=message.id,
            role=message.role,
            model="rules-baseline",
            request_id=None,
            scores=scores,
            raw_answers={"note": "rules baseline; not calibrated"},
            latency_ms=0.0,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_classifier(
    *,
    mode: str | None = None,
    decision_policy: DecisionPolicy | None = None,
    **kwargs: Any,
) -> JevClassifier:
    """Select a classifier based on explicit ``mode`` or environment.

    Modes:
    - ``"live"``     -> :class:`LiveJevClassifier`. Requires TYPESAFE_API_KEY.
    - ``"offline"``  -> :class:`OfflineMockClassifier`. Explicitly labelled.
    - ``"baseline"`` -> :class:`RulesBaselineClassifier`. Explicitly labelled.

    If ``mode`` is ``None``, ``live`` is preferred when a key is available,
    otherwise offline. We never silently downgrade live to offline.
    """

    policy = decision_policy or DecisionPolicy()
    if mode is None:
        mode = "live" if os.environ.get("TYPESAFE_API_KEY") else "offline"

    if mode == "live":
        return LiveJevClassifier(policy, **kwargs)
    if mode == "offline":
        return OfflineMockClassifier(policy, **kwargs)
    if mode == "baseline":
        return RulesBaselineClassifier(policy, **kwargs)
    raise ValueError(f"Unknown classifier mode: {mode!r}")
