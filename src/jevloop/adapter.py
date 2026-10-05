"""Pluggable classifier interface and Jev adapter.

Design:
- :class:`JevClassifier` is an abstract base. Anything that turns a
  :class:`Message` into a :class:`Classification` is a valid classifier.
- :class:`LiveJevClassifier` is the real adapter to ``typesafe_sdk``. It
  evaluates ONE Noul per category in a single parallel call, so the live
  path matches the documented Jev contract (independent questions, one
  request, calibrated probabilities).
- :class:`OfflineMockClassifier` is an honest offline mode. It is
  *labelled* in the output as offline/mock and demonstrates orchestration
  only; it is NOT a substitute for Jev's semantic quality.
- :func:`build_classifier` selects between them based on environment
  configuration. We never silently fall back to mock: if ``TYPESAFE_API_KEY``
  is missing, the constructor raises and the caller decides whether to
  use the labelled offline mock explicitly.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable
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

    def __init__(self, decision_policy: DecisionPolicy):
        self.decision_policy = decision_policy

    @abstractmethod
    def classify(
        self,
        message: Message,
        *,
        categories: Iterable[Category] = CATEGORY_CATALOG,
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
        return getattr(result, "request_id", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Live adapter (typesafe_sdk)
# ---------------------------------------------------------------------------


class LiveJevClassifier(JevClassifier):
    """Real adapter to the TypeSafe ``typesafe_sdk`` Jev model.

    Each call evaluates one Noul per category in a single parallel batch,
    preserving Jev's contract: independent probabilities, calibrated outputs,
    one round-trip.

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
        model: str = "jev-latest",
        client: Any = None,  # pre-built typesafe_sdk.TypeSafeClient
    ):
        super().__init__(decision_policy)
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = base_url or os.environ.get("TYPESAFE_BASE_URL")
        self.model = model
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
        categories: Iterable[Category] = CATEGORY_CATALOG,
    ) -> Classification:
        cats = tuple(categories)
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
            raise ClassifierError(f"Jev call failed: {e}") from e
        latency_ms = (time.perf_counter() - started) * 1000.0

        scores: list[CategoryScore] = []
        for c in cats:
            try:
                noul_answer = result.nouls[c.id]
                p = float(noul_answer.noul)
            except (KeyError, AttributeError) as e:
                raise ClassifierError(
                    f"Classifier returned no answer for category {c.id!r}: {e}"
                ) from e
            abstained = bool(getattr(noul_answer, "abstained", False))
            threshold = self.decision_policy.threshold_for(c.id)
            scores.append(
                CategoryScore(
                    category_id=c.id,
                    label=c.label,
                    proposition=c.proposition,
                    probability=p,
                    flagged=(not abstained) and p >= threshold,
                    abstained=abstained,
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

        Jev evaluates typed questions against a state. For message
        classification, the state is the message text plus minimal
        metadata. Keeping it small matches Jev's "atomic question" design.
        """
        return {
            "role": message.role,
            "message_id": message.id,
            "turn": message.turn,
            "text": message.text,
        }


# ---------------------------------------------------------------------------
# Offline mock — explicitly labelled, NOT a regex classifier on the main path
# ---------------------------------------------------------------------------


class OfflineMockClassifier(JevClassifier):
    """Labelled offline classifier.

    This mode demonstrates contracts and orchestration. It is NOT Jev and
    has no semantic quality. The mock returns deterministic per-category
    scores derived from a small fixture table, with calibrated-to-mock
    probabilities. The fixtures are checked into the repo so tests and
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
    ):
        super().__init__(decision_policy)
        # Default fixtures cover a handful of clearly-distinguishable inputs so
        # the offline demo can be run without any data files.
        self.fixtures = fixtures or _DEFAULT_FIXTURES
        self.unknown_default = unknown_default

    def classify(
        self,
        message: Message,
        *,
        categories: Iterable[Category] = CATEGORY_CATALOG,
    ) -> Classification:
        cats = tuple(categories)
        scores: list[CategoryScore] = []
        raw: dict[str, Any] = {}
        for c in cats:
            p = self._score(c.id, message.text, message.role)
            threshold = self.decision_policy.threshold_for(c.id)
            raw[c.id] = {"noul": p, "abstained": False}
            scores.append(
                CategoryScore(
                    category_id=c.id,
                    label=c.label,
                    proposition=c.proposition,
                    probability=p,
                    flagged=p >= threshold,
                    abstained=False,
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
        lowered = text.lower()
        for key, per_cat in self.fixtures.items():
            if key.lower() in lowered:
                return per_cat.get(category_id, self.unknown_default)
        # Fallback: neutral, low probability on everything. The mock is
        # honest about not knowing.
        return self.unknown_default


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
        "sad": 0.88,
        "deception": 0.07,
        "cunning": 0.10,
    },
    "please pretend this never happened": {
        "happy": 0.10,
        "sad": 0.18,
        "deception": 0.83,
        "cunning": 0.61,
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
    classifier. It exists so evaluation runs can show what the regex
    baseline gives you, versus what Jev gives you. The repo labels results
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
        categories: Iterable[Category] = CATEGORY_CATALOG,
    ) -> Classification:
        text_lower = message.text.lower()
        scores: list[CategoryScore] = []
        for c in CATEGORY_CATALOG if isinstance(categories, type(CATEGORY_CATALOG)) else categories:
            best = 0.05
            for kw, base in self.KEYWORDS.get(c.id, ()):
                if kw in text_lower:
                    best = max(best, base)
            threshold = self.decision_policy.threshold_for(c.id)
            scores.append(
                CategoryScore(
                    category_id=c.id,
                    label=c.label,
                    proposition=c.proposition,
                    probability=best,
                    flagged=best >= threshold,
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
