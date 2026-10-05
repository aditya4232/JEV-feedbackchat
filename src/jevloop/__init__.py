"""jevloop: Jev-driven feedback middleware for an agent loop.

Public surface (importable):

- :class:`JevClassifier`  - pluggable classifier interface
- :class:`LiveJevClassifier` - real adapter to ``typesafe_sdk``
- :class:`OfflineMockClassifier` - deterministic labelled offline mode
- :class:`AdaptationPolicy`  - deterministic versioned prompt adaptation
- :func:`run_turn`  - one complete turn of the loop
- :class:`LoopResult`, :class:`EffectivePrompt`, :class:`FeedbackPayload`,
  :class:`Classification`, :class:`AdaptedPrompt`, :class:`GeneratorOutput`,
  :class:`Message` - typed contracts
- :data:`CATEGORY_CATALOG`, :class:`DecisionPolicy` - taxonomy

The agent-facing contract is small and explicit. See :mod:`jevloop.contracts`.
"""

from .adapter import (
    JevClassifier,
    LiveJevClassifier,
    OfflineMockClassifier,
    RulesBaselineClassifier,
    build_classifier,
)
from .contracts import (
    AdaptationStep,
    AdaptedPrompt,
    CategoryScore,
    Classification,
    EffectivePrompt,
    FeedbackPayload,
    GeneratorOutput,
    Message,
    Role,
)
from .loop import (
    FeedbackLoop,
    LoopResult,
    echo_generator,
    recorded_generator,
    run_turn,
)
from .prompt_policy import AdaptationPolicy, Rule, default_rules
from .taxonomy import CATEGORY_CATALOG, Category, DecisionPolicy, category_by_id

__all__ = [
    "CATEGORY_CATALOG",
    "Category",
    "DecisionPolicy",
    "category_by_id",
    "Message",
    "Role",
    "Classification",
    "CategoryScore",
    "AdaptedPrompt",
    "AdaptationStep",
    "EffectivePrompt",
    "FeedbackPayload",
    "GeneratorOutput",
    "JevClassifier",
    "LiveJevClassifier",
    "OfflineMockClassifier",
    "RulesBaselineClassifier",
    "build_classifier",
    "AdaptationPolicy",
    "Rule",
    "default_rules",
    "FeedbackLoop",
    "LoopResult",
    "echo_generator",
    "recorded_generator",
    "run_turn",
]
